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
import os

from login import initialize_mt5, shutdown_mt5

# np.set_printoptions(threshold=np.inf)
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
    def __init__(self, name, start_index, prev_index, up_trend):
        self.name = name
        self.start_index = start_index
        self.prev_index = prev_index
        # self.pivot_data = pivot_data[-2:]  # セッション開始時点のピボットデータのコピー
        # self.start_pivot = pivot_data[-1] if pivot_data else datetime.now()
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
        # if self.up_trend is True:
        #     _, _, self.fibo_minus_20, self.fibo_minus_200 = detect_extension_reversal(
        #         self.pivot_data[-2:], None, None, 0.2, 2
        #     )
        # else:
        #     self.fibo_minus_20, self.fibo_minus_200, _, _ = detect_extension_reversal(
        #         self.pivot_data[-2:], -0.2, -2, None, None
        #     )

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


    def add_session(self, start_index, prev_index, up_trend):
        """
        新しいセッションを生成して管理リストに追加する。
        """
        session = MyModel(f"Session_{self.next_session_id}", start_index, prev_index, up_trend)
        self.sessions[self.next_session_id] = session

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


    



def determine_trend(sma):
    """
    SMAの時系列データから、前の値と比較して上昇しているかを判定するリストを作成する。
    
    Args:
        sma_series (Series): SMAの時系列データ。
    
    Returns:
        list: 各インデックスでの上昇(True) or 非上昇(False)のリスト。
    """
    trend_array = np.empty(sma.shape, dtype = np.float64)

    trend_array[0] = False
    trend_array[1:] = sma[:-1] < sma[1:]
    return trend_array



def detect_pivots(np_arr, time_df,name,POINT_THRESHOLD,
                  LOOKBACK_BARS, consecutive_bars,
                  arrow_spacing):

    """
    結合済み np_arr を用いて、SMA の転換点（ピボット）を検出する関数です。
    結果は各行 (detection_time, pivot_time, pivot_value, pivot_type)
    として返されます。すべて float64 型の数値（Unixタイムスタンプとしての時刻）です。
    
    入力 np_arr は以下の列構成を前提としています：
      - 列0: time (Unix タイムスタンプ、float64)
      - 列2: high
      - 列3: low
      - 列-2: SMA (calculate_sma の結果)
      - 列-1: trend (determine_trend の結果、True->1.0, False->0.0)
    
    パラメータ:
      POINT_THRESHOLD: SMA の変化量の閾値
      LOOKBACK_BARS: 転換判定のため遡るバー数
      consecutive_bars: 同一トレンドが連続している必要のあるバー数
      arrow_spacing: 前回ピボット検出からの最小間隔（バー数）
    
    戻り値:
      検出されたピボット情報を格納した numpy 配列（float64、shape=(n,4)）
      各行: (detection_time, pivot_time, pivot_value, pivot_type)
              pivot_type: 1.0=高値ピボット, 0.0=安値ピボット
    """

    if name == "BASE_SMA":
        wm = WaveManager()

    # np_arr の最後から2列目が SMA、最後の列が trend（1.0 or 0.0）
    sma = np_arr[:, -2]
    trend_arr = np_arr[:, -1]

    pivot_data = []
    last_pivot_index = -arrow_spacing  # 前回検出したピボットのインデックス
    run_counter = 1                   # 連続する同一トレンドのカウンター
    n = np_arr.shape[0]
    # 初期の up_trend 状態は、ここでは False（下降状態）として開始
    up_trend = False

    for i in range(1, n):
        # 前のバーと同じトレンドなら連続カウンターを増加させ、違えばリセット
        if trend_arr[i] == trend_arr[i-1]:
            run_counter += 1
        else:
            run_counter = 1

        # 連続していないと条件を満たさないのでスキップ
        if run_counter < consecutive_bars:
            continue

        # 前回ピボットから十分なバー数が経過しているかチェック
        if i - last_pivot_index < arrow_spacing:
            continue

        # ケース1: 上昇→下降の場合（高値ピボット検出）
        # ※ up_trend が True（上昇状態）で、trend_arr[i] が 0.0（False、下降へ転じたと仮定）
        if up_trend and (trend_arr[i] == 0.0):
            start = max(0, i - LOOKBACK_BARS)
            window_sma = sma[start: i+1]
            sma_max = np.nanmax(window_sma)
            # SMA の変化量が POINT_THRESHOLD 以上なら転換と判断
            if (sma_max - sma[i]) >= POINT_THRESHOLD:
                # ここでは、検出した index から LOOKBACK 分さかのぼったウィンドウ内の
                # 元データの high（列2）の中で、一番高い値を探します。
                window_high = np_arr[last_pivot_index: i+1, 2]
                local_high_idx = np.argmax(window_high) + last_pivot_index if last_pivot_index >= 0 else start
                detection_time = np_arr[i, 0]      # 転換条件が検出された時刻
                pivot_time = np_arr[local_high_idx, 0]  # ウィンドウ内で最も高い high の時刻
                pivot_value = np_arr[local_high_idx, 2]  # ウィンドウ内の最高 high 値
                pivot_type = 1.0  # 高値ピボット
                pivot_data.append((detection_time, pivot_time, pivot_value, pivot_type))
                up_trend = False  # 状態反転
                
                if name == "BASE_SMA":
                    wm.add_session(start_index=i, prev_index=last_pivot_index, up_trend="False")
                last_pivot_index = i
                # if POINT_THRESHOLD == 0.003:
                #     print(time_df[local_high_idx],time_df[i],pivot_value)
    

        # ケース2: 下降→上昇の場合（安値ピボット検出）
        # ※ up_trend が False（下降状態）で、trend_arr[i] が 1.0（上昇に転じたと仮定）
        elif (not up_trend) and (trend_arr[i] == 1.0):
            start = max(0, i - LOOKBACK_BARS)
            window = sma[start: i+1]
            window_min = np.nanmin(window)
            if (sma[i] - window_min) >= POINT_THRESHOLD:
                window_min = np_arr[last_pivot_index: i+1, 3]
                local_min_idx = np.argmin(window_min) + last_pivot_index if last_pivot_index >= 0 else start
                detection_time = np_arr[i, 0]
                pivot_time = np_arr[local_min_idx, 0]
                pivot_value = np_arr[local_min_idx, 3]  # 元データの low（列3）を利用
                pivot_type = 0.0  # 安値ピボット
                pivot_data.append((detection_time, pivot_time, pivot_value, pivot_type))
                up_trend = True  # 状態反転

                if name == "BASE_SMA":
                    wm.add_session(start_index=i, prev_index=last_pivot_index, up_trend="True")
                last_pivot_index = i
                # if POINT_THRESHOLD == 0.003:
                #     print(time_df[local_min_idx],time_df[i],pivot_value)
    # print(len(trend_arr),len(np_arr),len(sma))
    
    if pivot_data:
        if name == "BASE_SMA":
            print("base_ピボット終わり")
            return wm, np.array(pivot_data, dtype=np.float64)
        else:
            print("sml_ピボット終わり")
            return np.array(pivot_data, dtype=np.float64)
            
    else:
        return np.empty((0, 4), dtype=np.float64)
    
def merge_arr(base_arr, sml_arr):

    #↓結合したい順に並んでる
    base_np = base_arr["sma_arr"] #BASE_SMAと元のnp_arrの情報含んでる状態
    sml_sma_arr = sml_arr["sma_arr"]
    base_pivot_arr = base_arr["pivot_arr"]
    sml_pivot_arr = sml_arr["pivot_arr"]
    # print(base_np)
    # print(sml_sma_arr.shape)
    # print(base_pivot_arr.shape)
    # print(sml_pivot_arr.shape)
    # print(len(base_np))
    # print(len(sml_sma_arr))
    # print(len(base_pivot_arr))
    # print(len(sml_pivot_arr))
    # print(base_np.shape)
    
    np_arr_with_base_sml_sma = np.column_stack((base_np, sml_sma_arr.reshape(-1, 1)))
    print(np_arr_with_base_sml_sma.shape)
    columns = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume", "BASE_SMA", "SML_SMA"]
    df = pd.DataFrame(np_arr_with_base_sml_sma, columns=columns)

    

    # ここで、もともとの time 列は nanosecond 単位になっているので、unit="ns" を指定
    df["time"] = pd.to_datetime(df["time"], unit="ns", utc=True)

    pivot_columns = ["detection_time", "pivot_time", "pivot_value", "pivot_type"]
    df_pivot = pd.DataFrame(base_pivot_arr, columns=pivot_columns)
    df_pivot["detection_time"] = pd.to_datetime(df_pivot["detection_time"], unit="ns", utc=True)
    df_pivot["pivot_time"] = pd.to_datetime(df_pivot["pivot_time"], unit="ns", utc=True)

    sml_pivot_columns = ["sml_detection_time", "sml_pivot_time", "sml_pivot_value", "sml_pivot_type"]
    sml_df_pivot = pd.DataFrame(sml_pivot_arr, columns=sml_pivot_columns)
    sml_df_pivot["sml_detection_time"] = pd.to_datetime(sml_df_pivot["sml_detection_time"], unit="ns", utc=True)
    sml_df_pivot["sml_pivot_time"] = pd.to_datetime(sml_df_pivot["sml_pivot_time"], unit="ns", utc=True)

    df_sorted = df.sort_values("time")
    df_pivot_sorted = df_pivot.sort_values("detection_time")
    sml_df_pivot_sorted = sml_df_pivot.sort_values("sml_detection_time")
    
    # まず、df_sorted と df_pivot_sorted を結合
    merged_temp = pd.merge_asof(df_sorted, df_pivot_sorted,
                                left_on="time", right_on="detection_time",
                                direction="nearest", tolerance=pd.Timedelta("3sec"))
    
    

    # 次に、merged_temp と sml_df_pivot_sorted を結合
    merged = pd.merge_asof(merged_temp, sml_df_pivot_sorted,
                        left_on="time", right_on="sml_detection_time",
                        direction="nearest", tolerance=pd.Timedelta("3sec"))

    print("出力完了")
    return merged


def pre_data_process(np_arr,conditions,name,time_df):
    
    if name == "BASE_SMA":
        window = conditions.get("BASE_SMA", 20)
        point_threshold = conditions.get("BASE_threshold", 0)
        lookback_bars = conditions.get("BASE_lookback", 15)
        consecutive_bars = conditions.get("BASE_consecutive", 3)
        arrow_spacing = conditions.get("BASE_arrow_spacing", 8)
    elif name == "SML_SMA":
        window = conditions.get("SML_SMA", 4)
        point_threshold = conditions.get("SML_threshold", 0.002)
        lookback_bars = conditions.get("SML_lookback", 3)
        consecutive_bars = conditions.get("SML_consecutive", 1)
        arrow_spacing = conditions.get("SML_arrow_spacing", 1)



    sma_arr = calculate_sma(np_arr, window=window)
    trend_array = determine_trend(sma_arr)
    np_arr = np.column_stack((np_arr, sma_arr.reshape(-1, 1),trend_array.reshape(-1,1)))
    
    if name == "BASE_SMA":
        wm, pivot_arr = detect_pivots(np_arr,
                            time_df,name,
                            POINT_THRESHOLD=point_threshold,
                            LOOKBACK_BARS=lookback_bars,
                            consecutive_bars=consecutive_bars,
                            arrow_spacing=arrow_spacing)
        np_arr = np_arr[:, :-1]
        return (name, np_arr, pivot_arr, wm)
    
    elif name == "SML_SMA":
        pivot_arr = detect_pivots(np_arr,
                            time_df,name,
                            POINT_THRESHOLD=point_threshold,
                            LOOKBACK_BARS=lookback_bars,
                            consecutive_bars=consecutive_bars,
                            arrow_spacing=arrow_spacing)
        np_arr = np_arr[:, :-1]
        return (name, sma_arr, pivot_arr)
        
# グローバルに8つの WaveManager インスタンスを作成


def assign_session_to_manager(session):
    # ラウンドロビン方式などでWaveManagerを選択してセッションを追加する
    global session_counter
    manager = wave_managers[session_counter % len(wave_managers)]
    manager.add_session(session.pivot_data, session.up_trend, session.tp_level, session.stop_loss, session.check_no_SMA)
    session_counter += 1





import pytz



def process_data(conditions):
    global tp_level_global, check_no_SMA_global, range_param_global, stop_loss_global, time_df
    print("Current working directory:", os.getcwd())
    print(f"テスト開始時間{datetime.now()}")


    # conditions からパラメータを取得
    symbol = conditions.get("symbol", "USDJPY")
    fromdate = conditions.get("fromdate", datetime(2025, 2, 15, 20, 0))
    todate = conditions.get("todate", datetime(2025, 2, 20, 6, 50))
    BASE_SMA = conditions.get("BASE_SMA", 20)
    BASE_threshold = conditions.get("BASE_threshold", 0.01)
    BASE_lookback = conditions.get("BASE_lookback", 15)
    BASE_consecutive = conditions.get("BASE_consecutive", 3)
    BASE_arrow_spacing = conditions.get("BASE_arrow_spacing", 8)
    SML_SMA = conditions.get("SML_SMA", 4)
    SML_threshold = conditions.get("SML_threshold", 0.005)
    SML_lookback = conditions.get("SML_lookback", 3)
    SML_consecutive = conditions.get("SML_consecutive", 1)
    SML_arrow_spacing = conditions.get("SML_arrow_spacing", 1)
    tp_level_global = conditions.get("tp_level", 138)
    check_no_SMA_global = conditions.get("check_no_sma", True)
    output_file = conditions.get("output_file", "trade_logs.csv")
    # range, stop は将来のための仮パラメータ
    range_param_global = conditions.get("range", 80)
    stop_loss_global = conditions.get("stop", "sml")

    print("実行中")
    timezone = pytz.timezone("Etc/UTC")


    df = pd.read_pickle("currency_data/USDJPY_1M.pkl").loc[fromdate:todate].reset_index()
    
    time_df = df["time"]
    np_arr = df.to_numpy(dtype=np.float64)


    tasks = [
        (np_arr, conditions, "BASE_SMA",time_df),  # BASE用のタスク
        (np_arr, conditions, "SML_SMA",time_df)       # SML用のタスク
    ]

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(pre_data_process, *task) for task in tasks]
        # 結果をfuture.result()で取得
        results = [future.result() for future in futures]

    

    result_dict = {}
    for result in results:
        # BASE_SMA の場合はタプルの長さが 4 (name, np_arr, pivot_arr, wm)
        if len(result) == 4:
            name, arr, pivot_arr, wm = result
            result_dict[name] = {"sma_arr": arr, "pivot_arr": pivot_arr, "wm": wm}
        else:
            name, arr, pivot_arr = result
            result_dict[name] = {"sma_arr": arr, "pivot_arr": pivot_arr}
    base_arr = result_dict.get("BASE_SMA")
    sml_arr = result_dict.get("SML_SMA")

    wm = base_arr.get("wm")


    #ここではbase_arrはSMA結合済みの全体のnpとbase_pivot
    #sml_arrはsmaとpivot単体で入っている
    #既にどちらもnpとsma結合処理をpre_data_processで行っているため
    #このような形にすればsmaの結合を1回分削ることができる

    


    final_df = merge_arr(base_arr,sml_arr)


    print(f"終了時間{datetime.now()}")



    


    for idx in range(1440, len(original_df)):




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
        "fromdate": datetime(2020, 2, 16, 0, 0, tzinfo=pytz.UTC), #始まる日時
        "todate": datetime(2025, 2, 18, 7, 0, tzinfo=pytz.UTC), #終わる日時
        "BASE_SMA": 20, #BASE_SMAの期間
        "BASE_threshold": 0.009, #BASE_SMAの閾値
        "BASE_lookback": 15, #BASE_SMAの遡る期間
        "BASE_consecutive": 3, #BASE_SMAの上昇下降を判断する連続期間
        "BASE_arrow_spacing": 8, #BASE_SMAの矢印間隔
        "SML_SMA": 4, #SML_SMAの期間
        "SML_threshold": 0.003, #SML_SMAの閾値
        "SML_lookback": 3, #SML_SMAの遡る期間
        "SML_consecutive": 1, #SML_SMAの上昇下降を判断する連続期間
        "SML_arrow_spacing": 2, #SML_SMAの矢印間隔
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