# main.py

import time
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import threading
from transitions.extensions import HierarchicalMachine as Machine
import csv
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from login import initialize_mt5, shutdown_mt5

np.set_printoptions(threshold=np.inf)
# （Machine のimportは重複しているので必要に応じて整理してください）

# セッション管理用のグローバル WaveManager インスタンス
current_price_global = []

symbol = "USDJPY"  # デフォルト値
last_pivot_data = 999
sml_last_pivot_data = 999

states = [
    "created_base_arrow",
    "touched_20",
    "created_new_arrow",
    {"name": "infibos", "children": "has_determined_neck", 'initial': False},
    "has_position",
    "closed"
]

# 遷移定義
transitions = [
    {"trigger": "touch_20", "source": "created_base_arrow", "dest": "touched_20"},
    {"trigger": "create_new_arrow", "source": "touched_20", "dest": "created_new_arrow"},
    {"trigger": "touch_37", "source": ["created_new_arrow", "infibos_has_determined_neck"], "dest": "infibos"},
    {"trigger": "neck_determine", "source": "infibos", "dest": "infibos_has_determined_neck"},
    {"trigger": "build_position", "source": ["infibos", "infibos_has_determined_neck"], "dest": "has_position"},
    {"trigger": "close", "source": "has_position", "dest": "closed"}
]


#############################################
# セッションクラス（各セッションの状態を管理）
#############################################
class MyModel(object):
    def __init__(self, name, pivot_data, up_trend, tp_level ,stop_loss, check_no_SMA):
        self.name = name
        self.pivot_data = pivot_data[-2:]  # セッション開始時点のピボットデータのコピー
        self.start_pivot = pivot_data[-1] if pivot_data else datetime.now()
        self.machine = Machine(model=self, states=states, transitions=transitions, initial="created_base_arrow")
        self.new_arrow_pivot = None  # 推進波の終わりの最高（安）値を保管（以降調整波と考える）
        self.base_fibo37 = []  # 推進波に対する37%リトレースメントライン
        self.base_fibo70 = [] # 推進波に対する70%リトレースメントライン
        self.max_neck = []
        self.time_of_goldencross = []
        self.highlow_since_new_arrow = []  # 調整波の戻しの深さを把握
        self.sml_pivot_data = []  # touch20以降のsml_pivotを記録

        # ネックライン関連
        self.sml_pivots_after_goldencross = []
        self.potential_neck = []
        self.determined_neck = []

        self.destroy_reqest = False

        self.up_trend = True if up_trend == "True" else False

        self.state_times = {}  # 各状態移行時刻

        # 最後の2ピボットからフィボナッチラインを設定
        if self.up_trend is True:
            _, _, self.fibo_minus_20, self.fibo_minus_200 = detect_extension_reversal(
                self.pivot_data[-2:], None, None, 0.2, 2
            )
        else:
            self.fibo_minus_20, self.fibo_minus_200, _, _ = detect_extension_reversal(
                self.pivot_data[-2:], -0.2, -2, None, None
            )

        # 状態に応じた処理関数のディスパッチテーブル
        self.state_actions = {
            "created_base_arrow": self.handle_created_base_arrow,
            "touched_20": self.handle_touched_20,
            "created_new_arrow": self.handle_created_new_arrow,
            "infibos": self.handle_infibos,
            "infibos_has_determined_neck": self.handle_infibos_has_determined_neck,
            "has_position": self.handle_has_position,
            "closed": self.handle_closed
        }

    def execute_state_action(self, df, sml_df):
        """現在の状態に対応する処理関数を実行"""
        action = self.state_actions.get(self.state)
        if action:
            action(df.copy(), sml_df.copy())

    def handle_created_base_arrow(self, df, sml_df):
        if self.up_trend is True and check_touch_line(self.fibo_minus_20, df.iloc[-1]["high"]):
            self.touch_20()
        elif self.up_trend is False and check_touch_line(self.fibo_minus_20, df.iloc[-1]["low"]) is False:
            self.touch_20()

    def handle_touched_20(self, df, sml_df):
        if self.up_trend is True:
            result = watch_price_in_range(self.start_pivot[1], self.fibo_minus_200, df.iloc[-1]["high"])
        else:
            result = watch_price_in_range(self.fibo_minus_200, self.start_pivot[1], df.iloc[-1]["low"])

        if result is False:
            self.destroy_reqest = True
            print(f"{self.name}:200のラインに触れたので削除")

    def handle_created_new_arrow(self, df, sml_df):
        self.highlow_since_new_arrow = self.get_high_and_low_in_term(df, self.new_arrow_pivot[0])
        if self.up_trend is True and check_touch_line(self.base_fibo37, self.highlow_since_new_arrow[1]) is False:
            self.touch_37()
        elif self.up_trend is False and check_touch_line(self.base_fibo37, self.highlow_since_new_arrow[0]) is True:
            self.touch_37()
        self.price_in_range_while_adjustment(df)

    def handle_infibos(self, df, sml_df):
        if self.potential_neck:
            entry_result = self.potential_entry(df, self.potential_neck)
            if entry_result is True:
                self.build_position()
            elif entry_result is False:
                self.potential_neck = []
        elif len(self.determined_neck) > 0:
            self.neck_determine()
        self.price_in_range_while_adjustment(df)

    def handle_infibos_has_determined_neck(self, df, sml_df):
        if self.potential_neck:
            entry_result = self.potential_entry(df, self.potential_neck)
            if entry_result is True:
                self.build_position()
            elif entry_result is False:
                self.potential_neck = []

        if "has_position" not in self.state:
            self.highlow_since_new_arrow = self.get_high_and_low_in_term(df, self.new_arrow_pivot[0])
            if self.up_trend is True:
                for neckline in self.determined_neck[:]:
                    if self.state != 'infibos_has_determined_neck':
                        break
                    if df.iloc[-1]["high"] > neckline[1] and self.check_no_SMA(df.iloc[-1050:], neckline[1]):
                        self.stop_loss = self.highlow_since_new_arrow[1] - 0.006
                        pivots_data_to_get_take_profit = self.start_pivot, self.new_arrow_pivot
                        highlow = detect_extension_reversal(pivots_data_to_get_take_profit, higher1_percent=0.32)
                        self.take_profit = highlow[2]
                        self.entry_line = neckline[1] + 0.002
                        self.entry_pivot = df.iloc[-1]
                        self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                        self.point_to_stoploss = abs(self.entry_line - self.stop_loss)
                        self.build_position()
                    elif df.iloc[-1]["high"] > neckline[1] and self.check_no_SMA(df.iloc[-1050:], neckline[1]) is False:
                        self.determined_neck.remove(neckline)
                        if not self.determined_neck:
                            self.touch_37()
            else:
                for neckline in self.determined_neck[:]:
                    if self.state != 'infibos_has_determined_neck':
                        break
                    if df.iloc[-1]["low"] < neckline[1] and self.check_no_SMA(df.iloc[-1050:], neckline[1]) is False:
                        self.stop_loss = self.highlow_since_new_arrow[0] + 0.006
                        pivots_data_to_get_take_profit = self.start_pivot, self.new_arrow_pivot
                        highlow = detect_extension_reversal(pivots_data_to_get_take_profit, lower1_percent=-0.32)
                        self.take_profit = highlow[0]
                        self.entry_line = neckline[1] - 0.002
                        self.entry_pivot = df.iloc[-1]
                        self.point_to_stoploss = abs(self.entry_line - self.stop_loss)
                        self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                        self.build_position()
                    elif df.iloc[-1]["low"] < neckline[1] and self.check_no_SMA(df.iloc[-1050:], neckline[1]) is False:
                        self.determined_neck.remove(neckline)
                        if not self.determined_neck:
                            self.touch_37()

        self.price_in_range_while_adjustment(df)

    def handle_has_position(self, df, sml_df):
        if self.up_trend is True:
            if df.iloc[-1]["low"] < self.stop_loss:
                self.win = False
                self.result = -1
                self.close()
            elif df.iloc[-1]["high"] > self.take_profit:
                self.win = True
                self.result = self.point_to_take_profit / self.point_to_stoploss
                self.close()
        else:
            if df.iloc[-1]["high"] > self.stop_loss:
                self.win = False
                self.result = -1
                self.close()
            elif df.iloc[-1]["low"] < self.take_profit:
                self.win = True
                self.result = self.point_to_take_profit / self.point_to_stoploss
                self.close()

    def handle_closed(self, df, sml_df):
        self.destroy_reqest = True

    # on_enter 系
    def record_state_time(self):
        state_name = self.state
        self.state_times[state_name] = current_df.iloc[-1]["time"]

    def on_enter_created_new_arrow(self):
        self.record_state_time()
        pvts = self.start_pivot, self.new_arrow_pivot
        if self.up_trend is True:
            self.base_fibo70, _, self.base_fibo37, _ = detect_extension_reversal(pvts, lower1_percent=0.3, higher1_percent=-0.37)
        else:
            self.base_fibo37, _, self.base_fibo70, _ = detect_extension_reversal(pvts, lower1_percent=0.37, higher1_percent=-0.3)

    def on_enter_infibos(self):
        self.record_state_time()
        self.get_potential_neck_wheninto_newarrow()

    def on_enter_has_position(self):
        self.record_state_time()
        print(f"名前：{self.name}, エントリーライン：{self.entry_line}、テイクプロフィット：{self.take_profit}、"
              f"ストップロス：{self.stop_loss}、エントリーピボット：{self.entry_pivot}、"
              f"ポイント：{self.point_to_stoploss}、ポイント：{self.point_to_take_profit}、"
              f"ステート時間：{self.state_times}")

    def on_enter_closed(self):
        self.record_state_time()
        print(f"{self.name}: Entered 'くろーず' state.")

    def __repr__(self):
        return f"MyModel(name={self.name}, state={self.state})"


        

#---------------------------------------------------------------------
#その他の今後も特に使いそうな機能
#---------------------------------------------------------------------

    def get_high_and_low_in_term(self, df,time):
        """
        dfから、指定した期間以降の最高値と最安値を検出しreturnする
        """
        if df.index.name != "time":
            df = df.set_index("time", drop=False)
        required_df = df[time:].copy()
        highest_price = required_df[["open", "high", "low", "close"]].max().max()
        lowest_price = required_df[['open', 'high', 'low', 'close']].min().min()
        highest_close = required_df["close"].max()
        lowest_close = required_df["close"].min()

        return highest_price, lowest_price, highest_close, lowest_close

    def get_golden_cross_time(self,df,sml_df):
        """
        dfは過去100本のローソクのデータ(datetime型のtime,open,close,high,low,20MAの値など)
        sml_dfは過去100本のローソクのデータ(datetime型のtime,open,close,high,low,4MAの値など)
        基準となるBASE_SMAと一つ下のフラクタル構造のSML_SMAのゴールデンクロスを起こした時間を検出
        調整波の始まり時間をゴールデンクロス基準で把握し、それ以降の最も深い調整位置を知るためのメソッド。
        このゴールデンクロスを起こした後にBASE_SMAが調整方向に転換していてtouch37を満たせば
        調整波として完全に基準を満たしていると判断することができる
        USDJPY 2/19 13:22付近でのエントリーみたいなのをなくすための措置
        """
        df_indexed = df.copy().set_index("time", drop=False)
        sml_df_indexed = sml_df.copy().set_index("time", drop=False)
        base_sma_since_new_arrow = df_indexed.loc[self.new_arrow_pivot[0]:].copy()
        sml_sma_since_new_arrow = sml_df_indexed.loc[self.new_arrow_pivot[0]:].copy()

        if self.up_trend is True:
            for i in range(0, len(base_sma_since_new_arrow)):
                if base_sma_since_new_arrow.iloc[i]["BASE_SMA"] > sml_sma_since_new_arrow.iloc[i]["SML_SMA"]:
                    return base_sma_since_new_arrow.iloc[i]["time"] - timedelta(minutes=1)
            return df.iloc[-1]["time"]
            
        elif self.up_trend is False:
            for i in range(0, len(base_sma_since_new_arrow)):
                if base_sma_since_new_arrow.iloc[i]["BASE_SMA"] < sml_sma_since_new_arrow.iloc[i]["SML_SMA"]:
                    return base_sma_since_new_arrow.iloc[i]["time"] - timedelta(minutes=1)
            return df.iloc[-1]["time"]

        
    def check_no_SMA(self,df,neckline):
        """
        指定シンボルについて、1分足と5分足の過去 n_bars 本の確定済みデータから、
        SMA (close) の各期間（25,75,100,150,200）の最新値を計算し、
        その中で最も低い値と最も高い値を返す。
        
        Args:
            symbol (str): 例 "USDJPY"
            n_bars (int): 取得するバー数。デフォルトは200
            
        Returns:
            tuple: (lowest_SMA, highest_SMA)
                もしデータが取得できない場合は None を返す。
        """
        df = df.copy()
        periods = [25, 75, 100, 150, 200, 375, 625, 1000]
        sma_values = []
            
        # それぞれの SMA を計算する。ローリング平均は直近の確定足（in-progressの足は含まれない）
        for period in periods:
            if len(df) < period:
                continue  # 十分なデータがない場合はスキップ
            sma_col = f"SMA_{period}"
            df.loc[:, sma_col] = df['close'].rolling(window=period).mean()
            # dropna() しても良いが、最後の行は十分なデータがあればNaNにならないので、そのまま取得
            sma_value = df[sma_col].iloc[-1]
            sma_values.append(sma_value)



        if self.up_trend is True and neckline >= max(sma_values):
            return True
        
        elif self.up_trend is False and neckline <= min(sma_values):
            return False
        
        else:
            return None
        


    def get_sml_pivots_after_goldencross(self,sml_pivots):
        """
        goldencross以降のsmall_pivotsのデータを取得するメソッド
        """
        for idx, pivot in enumerate(sml_pivots):
            if pivot[0] > self.time_of_goldencross - timedelta(minutes=1):
                self.sml_pivots_after_goldencross = sml_pivots[idx:]

    def get_potential_neck_wheninto_newarrow(self):
        """
        ゴールデンクロス以降のsml_pivots(sml_pivots_after_goldencross)を
        sml_pvtsに格納し、その中にネックラインになりうるpivot(上昇トレンド中の調整のhigh
        下降トレンド中の調整のlow)があれば、その次のsml_pvtsの戻しの深さ次第で
        determined_neckに入れる。その価格を超えたらエントリーできる点。
        ネックラインになりうるpivotの次のpivotが生成されてなければ
        potential_neckに格納。(この場合次のpivot確定待ち)
        """
        print(self.up_trend,self.time_of_goldencross)
        sml_pvts = self.sml_pivots_after_goldencross
        if len(sml_pvts) >= 2 and self.up_trend is True:
            # print(f"名前：{self.name},ステート時間：{self.state_times},ゴールデンクロス：{self.time_of_goldencross}")
            for i in range(1, len(sml_pvts)):
                # print(f"名前：{self.name},ステート時間：{self.state_times}")
                if sml_pvts[i][2] == "high" and sml_pvts[i][0] > self.state_times["infibos"]:
                    
                    if i + 1 < len(sml_pvts):
                        pvts_to_get_32level = [sml_pvts[i],sml_pvts[i-1]]
                        fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level,-0.32,0.32,None,None)
                        if watch_price_in_range(fibo32_of_ptl_neck[0],fibo32_of_ptl_neck[1],sml_pvts[i+1]):
                            
                            self.determined_neck.append(sml_pvts[i])
                            self.organize_determined_neck()
                    else:
                        pvts_to_get_32level = [sml_pvts[i],sml_pvts[i-1]]
                        fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level,-0.32,0.32,None,None)
                        self.potential_neck.append(sml_pvts[i])

        if len(sml_pvts) >= 2 and self.up_trend is False:
            for i in range(1, len(sml_pvts)):
                if sml_pvts[i][2] == "low" and sml_pvts[i][0] > self.state_times["infibos"]:
                    if i + 1 < len(sml_pvts):
                        pvts_to_get_32level = [sml_pvts[i],sml_pvts[i-1]]
                        fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level,None,None,-0.32,0.32)
                        if watch_price_in_range(fibo32_of_ptl_neck[2],fibo32_of_ptl_neck[3],sml_pvts[i+1]):
                            self.determined_neck.append(sml_pvts[i])
                            self.organize_determined_neck()
                    else:
                        pvts_to_get_32level = [sml_pvts[i],sml_pvts[i-1]]
                        fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level,None,None,-0.32,0.32)
                        self.potential_neck.append(sml_pvts[i])

        

    def check_potential_to_determine_neck(self):
        """
        self.sml_pivots_after_goldencrossに新しいsml_pivotが追加されてから実行しなければならない関数
        """
        sml_pvts = self.sml_pivots_after_goldencross

        if self.potential_neck:
            if len(sml_pvts) < 3:
                self.destroy_reqest = True
                return None

            pvts_to_get_32level = [sml_pvts[-3], self.potential_neck[-1]]

            if self.up_trend is True:

                fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level,-0.32,0.32,None,None)
                if watch_price_in_range(fibo32_of_ptl_neck[0],fibo32_of_ptl_neck[1],sml_pvts[-1][1]):
                    self.determined_neck.append(self.potential_neck[-1])
                    self.potential_neck.clear()
                    self.organize_determined_neck()
                    

            elif self.up_trend is False:
                fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level,None,None,-0.32,0.32)
                if watch_price_in_range(fibo32_of_ptl_neck[2],fibo32_of_ptl_neck[3],sml_pvts[-1][1]):
                    self.determined_neck.append(self.potential_neck[-1])
                    self.potential_neck.clear()
                    self.organize_determined_neck()

    def potential_entry(self, df, neckline):
        # print(f"ネックライン[-1][1]:{neckline[-1][1]}")
    # up_trendがTrueの場合
        if self.up_trend is True:
            if df.iloc[-1]["high"] > neckline[-1][1] and self.check_no_SMA(df.iloc[-1050:],neckline[-1][1]):
                self.stop_loss = self.highlow_since_new_arrow[1] - 0.006
                pivots_data_to_get_take_profit = self.start_pivot, self.new_arrow_pivot
                highlow = detect_extension_reversal(pivots_data_to_get_take_profit, higher1_percent=0.32)
                self.take_profit = highlow[2]
                self.entry_line = neckline[-1][1] + 0.002
                self.entry_pivot = df.iloc[-1]
                self.point_to_stoploss = abs(self.entry_line - self.stop_loss)
                self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                # print(f"エントリー記録：：：　アプトれ{self.up_trend}, {self.name}、ネック：{neckline}、エントリーライン：{self.entry_line}、エントリーピボット：{self.entry_pivot}, テイクプロフィット：{self.take_profit}")
                return True
            elif df.iloc[-1]["high"] > neckline[-1][1] and self.check_no_SMA(df.iloc[-1050:],neckline[-1][1]) is False:
                return False

        # up_trendがFalseの場合
        if self.up_trend is False:
            if df.iloc[-1]["low"] < neckline[-1][1] and self.check_no_SMA(df.iloc[-1050:],neckline[-1][1]) is False:
                self.stop_loss = self.highlow_since_new_arrow[0] + 0.006
                pivots_data_to_get_take_profit = self.start_pivot, self.new_arrow_pivot
                highlow = detect_extension_reversal(pivots_data_to_get_take_profit, lower1_percent=-0.32)
                self.take_profit = highlow[0]
                self.entry_line = neckline[-1][1] - 0.002
                self.entry_pivot = df.iloc[-1]
                self.point_to_stoploss = abs(self.entry_line - self.stop_loss)
                self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                # print(f"エントリー記録：：：　アプトれ{self.up_trend}, {self.name}、ネック：{neckline}、エントリーライン：{self.entry_line}、エントリーピボット：{self.entry_pivot}, テイクプロフィット：{self.take_profit}")
                return True
            elif df.iloc[-1]["low"] < neckline[-1][1] and self.check_no_SMA(df.iloc[-1050:],neckline[-1][1]) is False:
                return False

        # 条件に該当しない場合は明示的にFalseを返すなど
        return None

    def organize_determined_neck(self):
        result = []
        # print(f"名前：{self.name},開始:{self.pivot_data},アプトレ{self.up_trend},ポテンシャル{self.determined_neck},デタマインド{self.determined_neck}")
        # print(f"200ライン：{self.fibo_minus_200}")
        for item in self.determined_neck:
        # item[1] が数値であると仮定
            while result and item[1] > result[-1][1]:
                result.pop()
            result.append(item)
        self.determined_neck = result


    def price_in_range_while_adjustment(self, df):
        if self.up_trend:
            high, low = self.new_arrow_pivot[1], self.base_fibo70
        else:
            high, low = self.base_fibo70, self.new_arrow_pivot[1]
        
        judged_price = self.get_high_and_low_in_term(df, self.new_arrow_pivot[0])
        if high < judged_price[0] or low > judged_price[1]:
            self.destroy_reqest = True
        


#############################################
# セッションマネージャークラス
#############################################
class WaveManager(object):
    def __init__(self):
        self.sessions = {}  # セッションは session_id をキーに管理
        self.next_session_id = 1
        self.trade_logs = []


    def add_session(self, pivot_data, up_trend, tp_level ,stop_loss, check_no_SMA):
        """
        新しいセッションを生成して管理リストに追加する。
        """
        session = MyModel(f"Session_{self.next_session_id}", pivot_data, up_trend, tp_level ,stop_loss, check_no_SMA)
        self.sessions[self.next_session_id] = session
        print(f"New session created: {session},時間:{pivot_data}、アプトレ{session.up_trend}")
        self.next_session_id += 1
        return session

    def append_pivot_data(self, new_pivot_data, df, sml_df):
        """
        mainでlast_pivot_dataが更新されたら受け取って各セッションのpivot_dataに追加
        touched_20の場合推進波の形成が終わったサインとしてcreated_new_arrowに移る
        "created_base_arrow"の場合touched_20に移る前 (推進波になる前)に波終了でセッション削除
        """
        sessions_to_delete = []

        for session_id, session in self.sessions.items():
            session.pivot_data.append(new_pivot_data)

            if session.state == "touched_20":
                session.new_arrow_pivot = new_pivot_data
                session.time_of_goldencross = session.get_golden_cross_time(df, sml_df)
                session.get_sml_pivots_after_goldencross(session.sml_pivot_data)
                # if session.name == "Session_2":
                #     print(f"ここでセッション２テスト！セッション名:{session.name},トレンド：{session.up_trend},開始：{session.start_pivot}、スモール：{session.sml_pivot_data}")
                #     print("ゴールデンクロス後のぴぼと",session.sml_pivots_after_goldencross)
                
                session.create_new_arrow()
                
                """
                stateがtouched_20で矢印が出現ということは推進波が終わり20SMAまで調整方向に
                傾いている合図。SMAが傾いているころにはsml_smaは間違いなく
                調整方向へのゴールデンクロス状態。
                ゴールデンクロスが起きた時からBASE_SMAが調整方向に傾いたと検出されるまでにも
                調整の一番深い場所やネックラインとなりうるsml_pivotが生成されている可能性もあるので
                この段階でsml_pivots_after_goldencrossを取得しておく
                """
                
            elif session.state == "created_base_arrow":
                # print(f"{session_id}:20到達前に新しい矢印形成で削除,{df.iloc[-1]["time"]}")
                sessions_to_delete.append(session_id)
        
        for session_id in sessions_to_delete:
            # print(f"Session {session_id} deleted due to non trade posibility")
            # print(f"ステート:{session.state},ポテンシャル：{session.potential_neck},デターミンド：{session.determined_neck}")
            del self.sessions[session_id]
            

    def append_sml_pivot_data(self, new_sml_pivot_data):
        """
        mainでlast_pivot_dataが更新されたら受け取って各セッションのpivot_dataに追加
        touched_20の場合推進波の形成が終わったサインとしてcreated_new_arrowに移る
        "created_base_arrow"の場合touched_20に移る前 (推進波になる前)に波終了でセッション削除
        """
        state_list = ["infibos", "infibos_has_determined_neck"]
        avoid_list = ["created_base_arrow", "has_position", "closed"]
        for session in self.sessions.values():
            if session.state not in avoid_list:
                session.sml_pivot_data.append(new_sml_pivot_data)

            if session.new_arrow_pivot is not None:
                session.sml_pivots_after_goldencross.append(new_sml_pivot_data)

            if session.state in state_list:
                if session.potential_neck:
                    session.check_potential_to_determine_neck()
                if not session.potential_neck and session.up_trend is True and new_sml_pivot_data[2] == "high":
                    session.potential_neck.append(new_sml_pivot_data)
                elif not session.potential_neck and session.up_trend is False and new_sml_pivot_data[2] == "low":
                    session.potential_neck.append(new_sml_pivot_data)
            
        


    def send_candle_data_tosession(self,df,sml_df):#dfは直近100件渡されるようになってます
        """
        mainからローソク足データが送信された時に各セッションがstate次に進めないか確認
        """
        sessions_to_delete = []

        for session_id, session in self.sessions.items():
            if self.sessions[session_id].state == "closed":
                self.trade_logs.append(session.entry_pivot)
                print(f"トレードログテスト：名前：{session.name}{session.entry_pivot}")

            session.execute_state_action(df, sml_df)
            if session.destroy_reqest is True:
                sessions_to_delete.append(session_id)

            

        for session_id in sessions_to_delete:
            del self.sessions[session_id]
            print(f"Session {session_id} 削除、最終的にちゃんと実行.")
        


    def check_in_range(self):
        avoid_state = "created_base_arrow", "build_position","position_reached161","position_reached200"
        sessions_to_delete = []
        for session_id, session in self.sessions.items():
            if session.state not in avoid_state:
                if session.up_trend == True:    
                    result = watch_price_in_range(session.pivot_data[1],session.high150)
                    if result is False:
                        sessions_to_delete.append(session_id)
                        # print(f"{self.name}:check_in_rangeで範囲外で削除")
                else:
                    result = watch_price_in_range(session.pivot_data[0],session.low150)
                    if result is False:
                        sessions_to_delete.append(session_id)
                        # print(f"{self.name}:check_in_rangeで範囲外で削除")
        for session_id in sessions_to_delete:
            del self.sessions[session_id]

    def export_trade_logs_to_csv(trade_logs, filename="trade_logs.csv"):
        import csv

        total_trades = len(trade_logs)
        wins = sum(1 for trade in trade_logs if trade.get("win") is True)
        win_rate = wins / total_trades if total_trades > 0 else 0

        # 勝ち・負けの金額を集計する例（ここでは result が正なら利益、負なら損失と仮定）
        total_profit = sum(trade.get("result", 0) for trade in trade_logs if trade.get("result", 0) > 0)
        total_loss = -sum(trade.get("result", 0) for trade in trade_logs if trade.get("result", 0) < 0)
        profit_factor = total_profit / total_loss if total_loss > 0 else None

        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            # ヘッダ行：各トレードのピボットデータ（datetime, price, type）
            writer.writerow(["time", "price", "type"])
            for trade in trade_logs:
                # trade には例として、{"time": ..., "price": ..., "type": ..., "win": ..., "result": ...} と記録されているとする
                time_str = trade["time"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(trade["time"], "strftime") else trade["time"]
                writer.writerow([time_str, trade["price"], trade["type"]])
            
            # 空行を入れてからサマリー行を追加
            writer.writerow([])
            writer.writerow(["Total Trades", total_trades])
            writer.writerow(["Wins", wins])
            writer.writerow(["Win Rate", win_rate])
            writer.writerow(["Total Profit", total_profit])
            writer.writerow(["Total Loss", total_loss])
            writer.writerow(["Profit Factor", profit_factor])
        
        print(f"Trade logs exported to {filename}")


        


    def __repr__(self):
        return f"WaveManager(sessions={list(self.sessions.values())})"


def detect_extension_reversal(pivot_data, lower1_percent=None, lower2_percent=None, higher1_percent=None, higher2_percent=None):
    """
    low1はフィボナッチあてる2点のうち低い方の価格を0として考える。
    high1はフィボナッチあてる2点のうち高い方の価格を0として考える。
    例えば150と160のフィボの場合、low1に-0.2を入れると152
    low2に0.4を入れると154、high1に-0.2を入れると158、high2に0.2を入れると162
    """    
    
    if len(pivot_data) < 2:
        return (None, None)
    
    # 前回と直近のピボットの価格を取り出す
    price1 = pivot_data[-2][1]
    price2 = pivot_data[-1][1]
    
    # 波の低い方と高い方を求める
    low_val = min(price1, price2)
    high_val = max(price1, price2)
    
    # 波幅の計算
    wave_range = high_val - low_val

    if lower1_percent is not None:
        low1 = low_val - (-wave_range * lower1_percent)
    else:
        low1 = None

    if higher1_percent is not None:
        high1 = high_val - (-wave_range * higher1_percent)
    else:
        high1 =None

    if lower2_percent is not None:
        low2 = low_val - (-wave_range * lower2_percent)
    else:
        low2 = None

    if higher2_percent is not None:
        high2 = high_val - (-wave_range * higher2_percent)
    else:
        high2 = None
    
    return (low1, low2, high1, high2)


def detect_small_reversal(base_p,end_adjustment_p):
    low_val = min(base_p, end_adjustment_p)
    high_val = max(base_p, end_adjustment_p)
    
def get_out_of_range(low, high):
    """
    セッションのトレンド開始を判断する20%のラインを突破したか確認する関数
    True → 上に設定した価格を上抜けたという合図
    True → 下に設定した価格を下抜けたという合図
    None → どちらの価格も突破せず、lowとhighの間にいるという合図
    """
    if current_price_global["high"] >= high:
        return True
    elif current_price_global["low"] <= low:
        return False
    else:
        return None
    
def check_touch_line(center_price, tested_price):
    if center_price <= tested_price:
        return True
    elif center_price >= tested_price:
        return False

    
def watch_price_in_range(low,high,judged_price = current_price_global):
    low = min(low, high)
    high = max(low, high)
    if low <= judged_price <= high:
        return True
    else:
        return False
    

    



#-----------------------------------------------

def calculate_sma(np_arr, window=20):
    """
    NumPy配列に対してSMA（単純移動平均）を計算します。

    Args:
        np_arr (numpy.ndarray): 入力データ。少なくとも4列目（インデックス3）が 'close' 値として使われる。
        window (int): 移動平均のウィンドウサイズ（デフォルト20）。

    Returns:
        numpy.ndarray: 計算されたSMAの1次元配列。最初の window-1 要素はNaNになります。
    """
    np_arr = np_arr.copy()  # 入力配列のコピー作成
    close = np_arr[:, 4]
    kernel = np.ones(window, dtype=np.float64) / window #過去window分を合計してwindowで割ればＳＭＡの値とれる
    sma_valid = np.convolve(close, kernel, mode="valid") #closeに対してkarnelを当てていき、完全にwindow分取得できるところから格納する
    # 最初の window-1 行は計算できないので NaN で埋める
    sma_arr = np.empty_like(close)
    sma_arr[:window-1] = np.nan #windowが4としたら[3]
    sma_arr[window-1:] = sma_valid

    return sma_arr


    



# def determine_trend(sma):
#     """
#     SMAの時系列データから、前の値と比較して上昇しているかを判定するリストを作成する。
    
#     Args:
#         sma_series (Series): SMAの時系列データ。
    
#     Returns:
#         list: 各インデックスでの上昇(True) or 非上昇(False)のリスト。
#     """
#     trend_array = np.empty(sma.shape, dtype = np.float64)

#     trend_array[0] = False
#     trend_array[1:] = sma[:-1] < sma[1:]
#     print(f"トレンド{trend_array}")
#     return trend_array

def update_determine_trend(df,name):
    if df.loc[df.index[-2], name] < df.loc[df.index[-1], name]:
        df.loc[df.index[-1], "UP_DOWN"] = True
    elif df.loc[df.index[-2], name] > df.loc[df.index[-1], name]:
        df.loc[df.index[-1], "UP_DOWN"] = False
    else:
        df.loc[df.index[-1], "UP_DOWN"] = None
    # print(f"最新df:{df.loc[df.index[-1]]}")
    return df

# def detect_pivots(df, name, POINT_THRESHOLD=0.01, LOOKBACK_BARS=15, consecutive_bars=3, arrow_spacing=10):
#     """
#     指定したSMA列（name）の値とUP_DOWNカラムをもとに、トレンド転換のピボット（高値または安値）を検出する。
#     連続して同じUP_DOWNが続くバー数がconsecutive_bars以上であり、かつ前回のピボットからarrow_spacing以上離れている場合に
#     ピボットを検出するようにしています。

#     Args:
#         df (DataFrame): 時系列データ（'time', 'close', 'high', 'low', <name>、"UP_DOWN"などが含まれる）
#         name (str): SMA計算済みのカラム名（例："BASE_SMA" や "SML_SMA"）
#         POINT_THRESHOLD (float): 転換判定の閾値（SMAの極値との差がこの値以上なら転換とみなす）
#         LOOKBACK_BARS (int): 遡るバー数。直近のこの期間でのSMAの最高値／最低値を求める
#         consecutive_bars (int): 現在のトレンドを示す連続したバーの数がこの値以上でないとピボット検出を行わない
#         arrow_spacing (int): 前回ピボット検出後、次のピボット検出までに最低このバー数は間隔を空ける

#     Returns:
#         list: 各ピボット情報を (time, pivot_value, pivot_type) のタプルでまとめたリスト
#     """
#     df = df.copy()
#     pivot_data = []
#     last_pivot_index = -arrow_spacing  # 最初は十分離れていると仮定
#     run_counter = 1  # 現在の連続数。最初のバーは1とする

#     # up_down_listは計算済みの"UP_DOWN"カラム（True:上昇、False:下降）のリスト
#     up_down_list = df["UP_DOWN"].tolist()

#     # 初期のトレンドは、最初の連続が足りなければ仮にFalse（下降）と設定
#     # ※より厳密には初期の連続状態で決定すべきですが、ここではシンプルにFalseとする
#     up_trend = False  
#     # prev_h_or_l_index = None

#     # ループはインデックス1から開始（最初はrun_counterを更新するため）
#     for i in range(1, len(df)):
#         # 連続して同じ方向かどうかチェック
#         if up_down_list[i] == up_down_list[i - 1]:
#             run_counter += 1
#         else:
#             run_counter = 1

#         # 連続期間が十分でなければ、ピボット検出は行わない
#         if run_counter < consecutive_bars:
#             continue

#         # 前回ピボットからarrow_spacing以上離れていなければスキップ
#         if i - last_pivot_index < arrow_spacing:
#             continue

#         # 転換検出処理（上昇→下降の場合と下降→上昇の場合で分岐）
#         # ※ここでは「転換」が起こったかどうかをSMAの極値との差で判断する
#         if up_trend and not up_down_list[i]:
#             # 直前の上昇トレンド中から下落に転じた場合（高値ピボット）
#             pivot_index = i
#             # LOOKBACK期間内のSMAを抽出
#             sma_slice = df[name][pivot_index - LOOKBACK_BARS: pivot_index + 1]
#             sma_highest = sma_slice.max()
#             current_sma = df[name].iloc[pivot_index]

#             if (sma_highest - current_sma) >= POINT_THRESHOLD:
#                 # ピボットとする場合：該当期間中の高値（"high"カラム）も調べる
#                 hs = df['high'][pivot_index - LOOKBACK_BARS: pivot_index + 1]
#                 highest_index = hs.idxmax()
#                 highest = hs.max()
#                 highest_datetime = df["time"].iloc[highest_index]
#                 pivot_data.append({"time": highest_datetime, "pivot_value": highest, "pivot_type": "high"})
#                 last_pivot_index = i
#                 up_trend = False
#                 # オプション：prev_h_or_l_index を更新（今回は単純化のため割愛）
#         elif (not up_trend) and up_down_list[i]:
#             # 下降トレンドから上昇に転じた場合（安値ピボット）
#             pivot_index = i
#             if pivot_index - LOOKBACK_BARS < 0:
#                 continue
#             sma_slice = df[name][pivot_index - LOOKBACK_BARS: pivot_index + 1]
#             sma_lowest = sma_slice.min()
#             current_sma = df[name].iloc[pivot_index]

#             if (current_sma - sma_lowest) >= POINT_THRESHOLD:
#                 ls = df['low'][pivot_index - LOOKBACK_BARS: pivot_index + 1]
#                 lowest_index = ls.idxmin()
#                 lowest = ls.min()
#                 lowest_datetime = df["time"].iloc[lowest_index]
#                 pivot_data.append({"time": lowest_datetime, "pivot_value": lowest, "pivot_type": "low"})
#                 last_pivot_index = i
#                 up_trend = True
#                 # 同様に、prev_h_or_l_index の更新を行う場合はここで設定

#     pivots_df = pd.DataFrame(pivot_data)
#     return pivots_df


def detect_pivots(np_arr, sma_arr,time_df,
                             POINT_THRESHOLD=0.01,
                             LOOKBACK_BARS=15,
                             arrow_spacing=10):
    """
    SMAの転換点をベクトル演算＋ループで検出し、検出時刻も含めたピボット情報を返します。
    戻り値の各行は (detection_time, pivot_time, pivot_value, pivot_type) となります。
      - detection_time: 条件を満たした時刻（Unix Timestamp: float）
      - pivot_time: LOOKBACKウィンドウ内の最高値／最安値の時刻（Unix Timestamp: float）
      - pivot_value: 検出したピボットの値
      - pivot_type: 1 (high) または 2 (low)
      
    np_arr: 元の時系列データ（列0が time, 列2が high, 列3が low）
    sma_arr: 1次元のSMA配列
    """
    # 差分からトレンド（1:上昇, -1:下降, 0:変化なし）を計算
    
    diff = np.diff(sma_arr)
    trend = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    
    # トレンド変化箇所の候補インデックス（ベクトル演算で一括検出）
    change = np.diff(trend) != 0
    candidate_idx = np.where(change)[0] + 1  # 差分のため +1
    
    pivot_data = []
    last_pivot = -arrow_spacing  # 前回ピボットのインデックス（arrow_spacing のため）
    run_counter = 1

    for idx in candidate_idx:
        
        # arrow_spacing 条件のチェック
        if idx - last_pivot < arrow_spacing:
            continue

        # 現在の候補の検出時刻
        
        # 上昇→下降の場合：高値ピボット候補
        if trend[idx - 1] == 1 and trend[idx] == -1:
            start = max(0, idx - LOOKBACK_BARS)
            window = sma_arr[start: idx+1]
            sma_max = np.nanmax(window)
            # 閾値条件のチェック
            if (sma_max - sma_arr[idx]) >= POINT_THRESHOLD:
                # ウィンドウ内での最高値が現れた時刻を取得
                max_idx = np.argmax(window) + start
                detection_time = np_arr[idx, 0]
                pivot_time = np_arr[max_idx, 0]
                pivot_data.append((detection_time, pivot_time, sma_max, 1))
                last_pivot = idx
                if POINT_THRESHOLD == 0.009:
                    print(time_df[max_idx],time_df[idx])
                
        # 下降→上昇の場合：安値ピボット候補
        elif trend[idx - 1] == -1 and trend[idx] == 1:
            start = max(0, idx - LOOKBACK_BARS)
            window = sma_arr[start: idx+1]
            sma_min = np.nanmin(window)
            if (sma_arr[idx] - sma_min) >= POINT_THRESHOLD:
                min_idx = np.argmin(window) + start
                detection_time = np_arr[idx, 0]
                pivot_time = np_arr[min_idx, 0]
                pivot_data.append((detection_time, pivot_time, sma_min, 2))
                last_pivot = idx
                # if POINT_THRESHOLD == 0.009:
                #     print(time_df[min_idx],time_df[idx])
    pivot_time_str = np.datetime_as_string(np_arr[max_idx, 0].astype('datetime64[s]'), unit='s')
    if pivot_data:
        return np.array(pivot_data)
    else:
        return np.empty((0, 4), dtype=np.float64)
# def detect_pivots(np_arr, sma_arr, 
#                              POINT_THRESHOLD=0.01, LOOKBACK_BARS=15, 
#                              consecutive_bars=3, arrow_spacing=10):
#     """
#     SMAの変化と連続性に基づいてピボットを検出し、各ピボット情報をすべて float64 の
#     2 次元配列として返します。
    
#     戻り値の各行は [time, pivot_value, pivot_type] となり、
#     time は Unix Timestamp (秒単位)、
#     pivot_value は float64、
#     pivot_type は 1 (high) または 2 (low) となります。
    
#     np_arr: 元の時系列データ（例：列0が time, 列2が high, 列3が low）
#     sma_arr: 1次元のSMA配列
#     """
#     n = len(sma_arr)
#     pivot_data = []  # ピボット情報のリスト (time, pivot_value, pivot_type)
#     last_pivot_index = -arrow_spacing
#     run_counter = 1
#     # 初期のトレンドは sma_arr[1] と sma_arr[0] で決定
#     current_trend = sma_arr[1] > sma_arr[0]
#     up_trend = current_trend

#     # インデックス 2 からループ
#     for i in range(2, n):
#         new_trend = sma_arr[i] > sma_arr[i-1]
#         if new_trend == current_trend:
#             run_counter += 1
#         else:
#             run_counter = 1
#         current_trend = new_trend
        
#         if run_counter < consecutive_bars or (i - last_pivot_index) < arrow_spacing:
#             continue

#         if up_trend and not new_trend:
#             pivot_index = i
#             start_idx = max(0, pivot_index - LOOKBACK_BARS)
#             sma_window = sma_arr[start_idx: pivot_index+1]
#             sma_max = np.nanmax(sma_window)
#             current_sma = sma_arr[pivot_index]
#             if (sma_max - current_sma) >= POINT_THRESHOLD:
#                 # 高値ピボット：np_arr の列2 を使うと仮定
#                 high_window = np_arr[start_idx: pivot_index+1, 2]
#                 highest = np.nanmax(high_window)
#                 max_idx = np.argmax(high_window) + start_idx
#                 pivot_time = np_arr[max_idx, 0]  # 時間列（np.datetime64）
#                 # 時間を float64 の Unix Timestamp (秒) に変換
#                 time_float = pivot_time.astype('datetime64[s]').astype(float)
#                 pivot_data.append((time_float, highest, 1))  # 1 = high
#                 last_pivot_index = i
#                 up_trend = False
#                 run_counter = 0
#         elif (not up_trend) and new_trend:
#             pivot_index = i
#             if pivot_index - LOOKBACK_BARS < 0:
#                 continue
#             start_idx = pivot_index - LOOKBACK_BARS
#             sma_window = sma_arr[start_idx: pivot_index+1]
#             sma_min = np.nanmin(sma_window)
#             current_sma = sma_arr[pivot_index]
#             if (current_sma - sma_min) >= POINT_THRESHOLD:
#                 # 安値ピボット：np_arr の列3 を使うと仮定
#                 low_window = np_arr[start_idx: pivot_index+1, 3]
#                 lowest = np.nanmin(low_window)
#                 min_idx = np.argmin(low_window) + start_idx
#                 pivot_time = np_arr[min_idx, 0]
#                 time_float = pivot_time.astype('datetime64[s]').astype(float)
#                 pivot_data.append((time_float, lowest, 2))  # 2 = low
#                 last_pivot_index = i
#                 up_trend = True
#                 run_counter = 0

#     # pivot_data はリストのタプルなので、2 次元 float64 の ndarray に変換します。
#     if pivot_data:
#         pivot_arr = np.array(pivot_data, dtype=np.float64)
#     else:
#         pivot_arr = np.empty((0, 3), dtype=np.float64)
#     return pivot_arr

# def detect_pivots(np_array, sma_arr,POINT_THRESHOLD=0.01, LOOKBACK_BARS=15, consecutive_bars=3, arrow_spacing=10):
#     """
#     指定したSMA列（name）の値とUP_DOWNカラムをもとに、トレンド転換のピボット（高値または安値）を検出する。
#     連続して同じUP_DOWNが続くバー数がconsecutive_bars以上であり、かつ前回のピボットからarrow_spacing以上離れている場合に
#     ピボットを検出するようにしています。

#     Args:
#         df (DataFrame): 時系列データ（'time', 'close', 'high', 'low', <name>、"UP_DOWN"などが含まれる）
#         name (str): SMA計算済みのカラム名（例："BASE_SMA" や "SML_SMA"）
#         POINT_THRESHOLD (float): 転換判定の閾値（SMAの極値との差がこの値以上なら転換とみなす）
#         LOOKBACK_BARS (int): 遡るバー数。直近のこの期間でのSMAの最高値／最低値を求める
#         consecutive_bars (int): 現在のトレンドを示す連続したバーの数がこの値以上でないとピボット検出を行わない
#         arrow_spacing (int): 前回ピボット検出後、次のピボット検出までに最低このバー数は間隔を空ける

#     Returns:
#         list: 各ピボット情報を (time, pivot_value, pivot_type) のタプルでまとめたリスト
#     """
#     wave_managers = [WaveManager() for _ in range(8)]
#     session_counter = 0
#     n = len(sma_arr)

#     pivot_arr = np.empty(n, dtype=np.dtype([
#                         ("time", "datetime64[ns]"),
#                         ("pivot_value", "f8"),
#                         ("pivot_type", "U10")]))

#     last_pivot_index = -arrow_spacing  # 最初は十分離れていると仮定
#     run_counter = 1

#     # 初期のトレンドは sma_arr[1] と sma_arr[0] の比較で決定
#     current_trend = sma_arr[1] > sma_arr[0]
#     up_trend = current_trend  # Trueなら上昇、Falseなら下降
    



def pre_data_process(np_arr,conditions,name,time_df):
    if name == "BASE_SMA":
        window = conditions.get("BASE_SMA", 20)
        col_name = "BASE_SMA"
        point_threshold = conditions.get("BASE_threshold", 0)
        lookback_bars = conditions.get("BASE_lookback", 15)
        consecutive_bars = conditions.get("BASE_consecutive", 3)
        arrow_spacing = conditions.get("BASE_arrow_spacing", 8)
    elif name == "SML_SMA":
        window = conditions.get("SML_SMA", 4)
        col_name = "SML_SMA"
        point_threshold = conditions.get("SML_threshold", 0.002)
        lookback_bars = conditions.get("SML_lookback", 3)
        consecutive_bars = conditions.get("SML_consecutive", 1)
        arrow_spacing = conditions.get("SML_arrow_spacing", 1)

    

    sma_arr = calculate_sma(np_arr, window=window)
    np_arr = np.column_stack((np_arr, sma_arr.reshape(-1, 1)))
    pivot_arr = detect_pivots(np_arr,sma_arr,time_df,point_threshold,lookback_bars,arrow_spacing)
    
# グローバルに8つの WaveManager インスタンスを作成


def assign_session_to_manager(session):
    # ラウンドロビン方式などでWaveManagerを選択してセッションを追加する
    global session_counter
    manager = wave_managers[session_counter % len(wave_managers)]
    manager.add_session(session.pivot_data, session.up_trend, session.tp_level, session.stop_loss, session.check_no_SMA)
    session_counter += 1





import pytz



def process_data(conditions):
    global last_pivot_data, sml_last_pivot_data, current_price_global, current_df
    
    


    # conditions からパラメータを取得
    symbol = conditions.get("symbol", "USDJPY")
    fromdate = conditions.get("fromdate", datetime(2025, 2, 15, 20, 0))
    todate = conditions.get("todate", datetime(2025, 2, 20, 6, 50))
    BASE_SMA = conditions.get("BASE_SMA", 20)
    BASE_threshold = conditions.get("BASE_threshold", 0.009)
    BASE_lookback = conditions.get("BASE_lookback", 15)
    BASE_consecutive = conditions.get("BASE_consecutive", 3)
    BASE_arrow_spacing = conditions.get("BASE_arrow_spacing", 8)
    SML_SMA = conditions.get("SML_SMA", 4)
    SML_threshold = conditions.get("SML_threshold", 0.002)
    SML_lookback = conditions.get("SML_lookback", 3)
    SML_consecutive = conditions.get("SML_consecutive", 1)
    SML_arrow_spacing = conditions.get("SML_arrow_spacing", 1)
    tp_level = conditions.get("tp_level", 138)
    check_no_SMA = conditions.get("check_no_sma", True)
    output_file = conditions.get("output_file", "trade_logs.csv")
    # range, stop は将来のための仮パラメータ
    range_param = conditions.get("range", 80)
    stop_loss = conditions.get("stop", "sml")

    print("実行中")
    timezone = pytz.timezone("Etc/UTC")


    df = pd.read_pickle("currency_data/USDJPY_1M.pkl").loc[fromdate:todate].reset_index()
    time_df = df["time"]
    np_array = df.to_numpy(dtype=np.float64)
    print(len(np_array))

    tasks = [
        (np_array, conditions, "BASE_SMA",time_df),  # BASE用のタスク
        (np_array, conditions, "SML_SMA",time_df)       # SML用のタスク
    ]

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(pre_data_process, *task) for task in tasks]
        # 結果をfuture.result()で取得
        results = [future.result() for future in futures]
        


    


    last_pivot_data = pivot_data[-1]
    sml_last_pivot_data = sml_pivot_data[-1]
    print(f"テスト開始時間{datetime.now()}")
    print(f"開始時間：{df.iloc[-1]['time']}")

    for idx in range(1440, len(original_df)):
        new_row = original_df.copy().iloc[idx:idx+1]
        df = pd.concat([df, new_row], ignore_index=True)
        sml_df = pd.concat([sml_df, new_row], ignore_index=True)

        # SMA再計算
        df = calculate_sma(df, window=BASE_SMA, name="BASE_SMA")
        sml_df = calculate_sma(sml_df, window=SML_SMA, name="SML_SMA")

        update_determine_trend(df, "BASE_SMA")
        update_determine_trend(sml_df, "SML_SMA")


        new_pivot = update_detect_pivot(df, point_threshold=BASE_threshold, lookback_bars=BASE_lookback,
                                        consecutive_bars=BASE_consecutive, arrow_spacing=BASE_arrow_spacing,
                                        name="BASE_SMA")

        if new_pivot is not None and new_pivot != last_pivot_data:
            last_pivot_data = new_pivot
            pivot_data.append(new_pivot)
            wm.append_pivot_data(last_pivot_data, df, sml_df)
            if new_pivot[2] == "high":
                wm.add_session(pivot_data[-2:], up_trend="False", tp_level=tp_level, stop_loss=stop_loss, check_no_SMA=check_no_SMA)
            else:
                wm.add_session(pivot_data[-2:], up_trend="True", tp_level=tp_level, stop_loss=stop_loss, check_no_SMA=check_no_SMA)

        sml_new_pivot = update_detect_pivot(sml_df,
                                            point_threshold=SML_threshold,
                                            lookback_bars=SML_lookback,
                                            consecutive_bars=SML_consecutive,
                                            name="SML_SMA",
                                            arrow_spacing=SML_arrow_spacing)
        
        if sml_new_pivot is not None and sml_new_pivot != sml_last_pivot_data:
            sml_last_pivot_data = sml_new_pivot
            sml_pivot_data.append(sml_new_pivot)
            wm.append_sml_pivot_data(sml_last_pivot_data)

        if not df.empty:
            current_df = df.tail(1)
        else:
            continue

        current_price_global.clear()
        current_price_global.append(current_df.iloc[-1])

        wm.send_candle_data_tosession(df.iloc[-1100:].copy(), sml_df.iloc[-1100:].copy())
    
    print("処理終了")
    print(f"終了時間{datetime.now()}")

    # 出力処理（ここでは例としてコメントアウト）
    wm.export_trade_logs_to_csv(wm.trade_logs, filename=output_file)

if __name__ == "__main__":
    conditions = {
        "symbol": "USDJPY",
        "fromdate": datetime(2015, 6, 19, 0, 0, tzinfo=pytz.UTC), #始まる日時
        "todate": datetime(2025, 2, 20, 0, 0, tzinfo=pytz.UTC), #終わる日時
        "BASE_SMA": 20, #BASE_SMAの期間
        "BASE_threshold": 0.009, #BASE_SMAの閾値
        "BASE_lookback": 15, #BASE_SMAの遡る期間
        "BASE_consecutive": 3, #BASE_SMAの上昇下降を判断する連続期間
        "BASE_arrow_spacing": 8, #BASE_SMAの矢印間隔
        "SML_SMA": 4, #SML_SMAの期間
        "SML_threshold": 0.002, #SML_SMAの閾値
        "SML_lookback": 3, #SML_SMAの遡る期間
        "SML_consecutive": 1, #SML_SMAの上昇下降を判断する連続期間
        "SML_arrow_spacing": 1, #SML_SMAの矢印間隔
        "range" : 80, #一旦無視でいい
        "stop" : "sml", #一旦無視でいい
        "tp_level": 138, #利確ライン
        "check_no_sma" : True, #Trueの場合check_no_smaを実行。Falseの場合は実行しない
        "output_file": "USDJPY_138_trade_logs.csv" #出力ファイル名
        

    }
    process_data(conditions)



            #     print(f"名前：{self.name}\n"
        #         f"ここでテスト！セッション名：{self.name}\n"
        #         f"カレント：{current_df['time']}\n"
        #         f"開始：{self.start_pivot}\n"
        #         f"トレンド：{self.up_trend}\n"
        #         f"ステート：{self.state}\n"
        #         f"スターと：{self.new_arrow_pivot}\n"
        #         f"ニューアロー：{self.new_arrow_pivot}\n"
        #         f"ポテンシャル：{self.potential_neck}\n"
        #         f"デタマイン：{self.determined_neck}\n"
        #         f"ベース37：{self.base_fibo37}\n"
        #         f"ベース70：{self.base_fibo70}\n"
        #         f"ゴールデンクロス：{self.time_of_goldencross}\n"
        #         f"スモール：{self.sml_pivot_data}\n"
        #         f"ゴールデンクロス後のスモール：{self.sml_pivots_after_goldencross}\n"

        # )