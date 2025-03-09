# main.py

import time
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import threading
from transitions.extensions import HierarchicalMachine as Machine
import MetaTrader5 as mt5
import csv
import pandas as pd

from login import initialize_mt5, shutdown_mt5
from transitions.extensions import HierarchicalMachine as Machine


# セッション管理用のグローバル WaveManager インスタンス

current_price_global = []



symbol="USDJPY"

last_pivot_data = 999
sml_last_pivot_data = 999


states = [
    "created_base_arrow",
    "touched_20",
    "created_new_arrow",
    {"name":"infibos", "children":"has_determined_neck", 'initial': False},
    "has_position",
    "closed"
]

# 遷移定義
transitions = [
    {"trigger": "touch_20", "source": "created_base_arrow", "dest": "touched_20"},
    {"trigger": "create_new_arrow", "source": "touched_20", "dest": "created_new_arrow"},
    {"trigger": "touch_37", "source": ["created_new_arrow", "infibos_has_determined_neck"], "dest": "infibos"},
    {"trigger": "neck_determine", "source": "infibos", "dest": "infibos_has_determined_neck"},
    {"trigger": "build_position", "source": ["infibos","infibos_has_determined_neck"], "dest": "has_position"},
    {"trigger": "close", "source": "has_position", "dest": "closed"}
]

#############################################
# セッションクラス（各セッションの状態を管理）
#############################################
class MyModel(object):
    def __init__(self, name, pivot_data, up_trend):
        self.name = name
        self.pivot_data = pivot_data[-2:]  # セッション開始時点のピボットデータのコピー
        self.start_pivot = pivot_data[-1] if pivot_data else datetime.now()
        self.machine = Machine(model=self, states=states, transitions=transitions, initial="created_base_arrow")
        self.new_arrow_pivot = None #推進波の終わりの最高（安）値を保管。ここ以降を調整波と考えられる。append_pivot_dataで設定される
        self.base_fibo37 = [] #推進波に対してリトレースメント37%を知るための変数
        self.base_fibo70 = []
        self.time_of_goldencross = []
        self.highlow_since_new_arrow = [] #上昇トレンドの調整波の戻しの深さを知るための変数
        self.sml_pivot_data = [] #append_sml_pivot_dataでtouch20以降sml_pivotsを記録

#-------------ネックラインに関する制作-----------------
        self.sml_pivots_after_goldencross = []
        self.potential_neck = []
        self.determined_neck = []

        self.destroy_reqest = False


        if up_trend == "True":
            self.up_trend = True
        if up_trend == "False":
            self.up_trend = False

        self.state_times = {} #stateが移行した時の時刻を記録

        #base_arrowができるまでの最後の波に対してフィボナッチをあて
        #上(下)20%のライン(self.fibo_minus_20)に波が触れてこれば
        #そちらの方向への推進波ととらえることができる
        if up_trend == "True":
            _,_,self.fibo_minus_20, self.fibo_minus_200 = detect_extension_reversal(self.pivot_data[-2:],None, None, 0.2, 2)
        if up_trend == "False":
            self.fibo_minus_20, self.fibo_minus_200,_,_ = detect_extension_reversal(self.pivot_data[-2:],-0.2,-2,None, None)

        

        # 状態に応じた処理関数のディスパッチテーブルを定義
        self.state_actions = {
            "created_base_arrow": self.handle_created_base_arrow,
            "touched_20": self.handle_touched_20,
            "created_new_arrow": self.handle_created_new_arrow,
            "infibos": self.handle_infibos,
            "infibos_has_determined_neck":self.handle_infibos_has_determined_neck,
            "has_position": self.handle_has_position,
            "closed": self.handle_closed
        }

    def execute_state_action(self, df, sml_df): #現時点ではdf直近100件　df.iloc[-100:]
        """
        現在の状態に対応する処理関数を実行する
        """
        action = self.state_actions.get(self.state)
        if action:
            # 必要に応じて df を渡す
            action(df.copy(), sml_df.copy())
        else:
            pass

    # 各状態で実行される関数
    def handle_created_base_arrow(self, df, sml_df):
        # print(f"{self.name}: Handling 'created_base_arrow'")
        if self.up_trend is True and check_touch_line(self.fibo_minus_20,df.iloc[-1]["high"]):
            self.touch_20()
        elif self.up_trend is False and check_touch_line(self.fibo_minus_20,df.iloc[-1]["low"]) is False:
            self.touch_20()
        else:
            pass

    #矢印形成（append_pivot_data）で詳細条件作成
    def handle_touched_20(self, df, sml_df):
        if self.up_trend is True:
            result = watch_price_in_range(self.start_pivot[1], self.fibo_minus_200, df.iloc[-1]["high"])
        elif self.up_trend is False:
            result = watch_price_in_range(self.fibo_minus_200, self.start_pivot[1], df.iloc[-1]["low"])
        
        if result is False:
            self.destroy_reqest = True
            print(f"{self.name}:200のラインに触れたので削除")
        # print(f"名前：{self.name}、リザルト、{result},リクエスト:{self.destroy_reqest},カレント：{current_df.iloc[-1]["time"]}")
        # print(f"{self.name}: Handling 'touched_20'")
        


    def handle_created_new_arrow(self, df, sml_df):
        # print(f"{self.name}: Handling 'created_new_arrow'")
        self.highlow_since_new_arrow = self.get_high_and_low_in_term(df, self.new_arrow_pivot[0]) 
        # print(f"前のぴぼと：{df.iloc[-1]}")
        if self.up_trend is True and check_touch_line(self.base_fibo37, self.highlow_since_new_arrow[1]) is False:
            #調整の深さを判断するので、推進波に対して37%戻したラインより前回のローソクの最安値(df.iloc[-1]["low"])が低ければ調整完了と判断
            self.touch_37()
        elif self.up_trend is False and check_touch_line(self.base_fibo37, self.highlow_since_new_arrow[0]) is True:
            self.touch_37()
        else:
            pass
        
        self.price_in_range_while_adjustment(df)

    #↓を修正して↑になったけど、↓はエントリー時の戻しの深さの最終チェックに使えるかも
    # def handle_created_new_arrow(self, df):
    #     print(f"{self.name}: Handling 'created_new_arrow'")
    #     self.highlow_since_new_arrow = self.get_high_and_low_in_term(df, self.new_arrow_pivot[0])#推進波の終わり以降の最高（安）値（調整の最も深い価格）を取得
    #     if self.up_trend is True and check_touch_line(self.base_fibo37, self.highlow_since_new_arrow[1]) is False:
    #         self.touch_37()
    #     elif self.up_trend is False and check_touch_line(self.base_fibo37, self.highlow_since_new_arrow[0]) is True:
    #         self.touch_37()
    #     else:
    #         pass

#--------------------ここから------------------------------
    
    # def handle_infibos(self, df, sml_df):
    #     self.price_in_range_while_adjustment(df)
        
    #     if self.potential_neck:
    #         if self.check_potential_entry() is True:
    #             self.build_position()
    #         elif self.check_potential_entry() is False:
    #             self.potential_neck = []
    #     elif len(self.determined_neck) > 0:
    #         self.neck_determine()

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
        

    #if self.name == "Session_2":
    #     print(f"ここでセッション２テスト！セッション名:{self.name},トレンド：{self.up_trend},開始：{self.start_pivot}、スモール：{self.sml_pivot_data}")
    #     print("ゴールデンクロス後のぴぼと",self.sml_pivots_after_goldencross)
    #     print(f"{self.name}: トレンド：{self.up_trend},ゴールデンクロス{self.sml_pivots_after_goldencross},ポテンシャル{self.potential_neck},デターミンド{self.determined_neck}、カレント{current_df}ステートタイム{self.state_times}")


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
                        break  # 状態が変わっていたら処理中断
                    # print(f"ネックライン[1]：{neckline[1]}")
                    # print(f"ネックライン[-1][1]：{neckline[-1][1]}")
                    if df.iloc[-1]["high"] > neckline[1] and self.check_no_SMA(df.iloc[-1050:],neckline[1]):
                        
                        self.stop_loss = self.highlow_since_new_arrow[1] - 0.006 #推進波終了以降の最も調整の深い部分から0.006円低い価格を取得
                        pivots_data_to_get_take_profit = self.start_pivot, self.new_arrow_pivot
                        highlow = detect_extension_reversal(pivots_data_to_get_take_profit, higher1_percent=0.32)
                        self.take_profit = highlow[2]
                        self.entry_line = neckline[1] + 0.002
                        self.entry_pivot = df.iloc[-1]
                        self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                        self.point_to_stoploss = abs(self.entry_line - self.stop_loss)
                        self.build_position()
                    elif df.iloc[-1]["high"] > neckline[1] and self.check_no_SMA(df.iloc[-1050:],neckline[1]) is False:
                        self.determined_neck.remove(neckline)
                        if not self.determined_neck:
                            self.touch_37()

            if self.up_trend is False:
                for neckline in self.determined_neck[:]:
                    if self.state != 'infibos_has_determined_neck':
                        break  # 状態が変わっていたら処理中断
                    # print(f"ネックライン[1]：{neckline[1]}")
                    # print(f"ネックライン[-1][1]：{neckline[-1][1]}")

                    if df.iloc[-1]["low"] < neckline[1] and self.check_no_SMA(df.iloc[-1050:],neckline[1]) is False:
                        self.stop_loss = self.highlow_since_new_arrow[0] + 0.006 #推進波終了以降の最も調整の深い部分から0.006円高い価格を取得
                        pivots_data_to_get_take_profit = self.start_pivot, self.new_arrow_pivot
                        highlow = detect_extension_reversal(pivots_data_to_get_take_profit, lower1_percent= -0.32)
                        self.take_profit = highlow[0]
                        self.entry_line = neckline[1] - 0.002
                        self.entry_pivot = df.iloc[-1]
                        self.point_to_stoploss = abs(self.entry_line - self.stop_loss)
                        self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                        self.build_position()
                        
                    elif df.iloc[-1]["low"] < neckline[1] and self.check_no_SMA(df.iloc[-1050:],neckline[1]) is False:
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

        if self.up_trend is False:
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
        
                


    # on_enter-------------------------------------
    # on_enter-------------------------------------
    # on_enter-------------------------------------
    # on_enter-------------------------------------
    # on_enter-------------------------------------

    def record_state_time(self):
        """状態が遷移した際に状態名と時刻を記録"""
        state_name = self.state
        self.state_times[state_name] = current_df.iloc[-1]["time"]

    def on_enter_created_new_arrow(self):
        """
        ここで推進波に対しての37%の戻りを定義しておく。
        上昇波であればhighに対して37%の戻り
        下降波であればlowに対して37%の戻り
        """
        self.record_state_time()
        
        pvts = self.start_pivot, self.new_arrow_pivot
        if self.up_trend is True:
            self.base_fibo70,_,self.base_fibo37,_ = detect_extension_reversal(pvts,lower1_percent=0.3, higher1_percent=-0.37)
        else:
            self.base_fibo37,_,self.base_fibo70,_ = detect_extension_reversal(pvts,lower1_percent=0.37, higher1_percent=-0.3)



    def on_enter_infibos(self):
        self.record_state_time()
        self.get_potential_neck_wheninto_newarrow()

    def on_enter_has_position(self):
        self.record_state_time()
        print(f"名前：{self.name}, エントリーライン：{self.entry_line}、テイクプロフィット：{self.take_profit}、ストップロス：{self.stop_loss}、エントリーピボット：{self.entry_pivot}、ポイント：{self.point_to_stoploss}、ポイント：{self.point_to_take_profit}、ステート時間：{self.state_times}")
    
    def on_enter_closed(self):
        self.record_state_time()
        print(f"{self.name}: Entered 'くろーず' state.")
    def __repr__(self):
        return f"MyModel(name={self.name}, state={self.state})"


        

#---------------------------------------------------------------------
#その他の今後も特に使いそうな機能
#---------------------------------------------------------------------

    def should_check():
        """現在の秒数がチェック対象の範囲内か判定する。"""
        # 現在の秒を取得
        sec = datetime.now().second
        # 例えば、58秒〜59秒、もしくは0秒〜3秒の間をチェック対象にする
        return sec >= 54 or sec <= 2
    
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
        sml_pvts = self.sml_pivots_after_goldencross
        if len(sml_pvts) >= 2 and self.up_trend is True:
            
            for i in range(1, len(sml_pvts)):
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


    def price_in_range_while_adjustment(self,df):
        if self.up_trend is True:

            high = self.new_arrow_pivot[1]
            low = self.base_fibo70
            judged_price = self.get_high_and_low_in_term(df, self.new_arrow_pivot[0])
            if high < judged_price[0] or low > judged_price[1]:
                
                self.destroy_reqest = True
                # print(f"Session {self.name}：調整中に推進波と70ラインの範囲外に出て削除")
                # print(f"ステート：{self.state}、カレント：{current_df},ハイ：{judged_price[0]},ロー：{judged_price[1]}")

        elif self.up_trend is False:
            high = self.base_fibo70
            low = self.new_arrow_pivot[1]
            judged_price = self.get_high_and_low_in_term(df, self.new_arrow_pivot[0])
            if high < judged_price[0] or low > judged_price[1]:
                self.destroy_reqest = True
                # print(f"Session {self.name}：調整中に推進波と70ラインの範囲外に出て削除")
        


#############################################
# セッションマネージャークラス
#############################################
class WaveManager(object):
    def __init__(self):
        self.sessions = {}  # セッションは session_id をキーに管理
        self.next_session_id = 1
        self.trade_logs = []


    def add_session(self, pivot_data, up_trend):
        """
        新しいセッションを生成して管理リストに追加する。
        """
        session = MyModel(f"Session_{self.next_session_id}", pivot_data, up_trend)
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
    

def initialize_mt5():
    """
    MT5への接続を初期化する。
    接続に成功すればTrue、失敗すればFalseを返す。
    """
    if not mt5.initialize():
        print("MT5の初期化に失敗しました")
        return False
    return True

def shutdown_mt5():
    """
    MT5の接続をシャットダウンする。
    """
    mt5.shutdown()


def fetch_data_range(symbol,from_date, to_date, timeframe=mt5.TIMEFRAME_M1 ):
    """
    指定された期間のデータを取得して DataFrame として返す。
    
    Args:
        symbol (str): 通貨ペア（例: "USDJPY"）
        timeframe: MT5 のタイムフレーム（例: mt5.TIMEFRAME_M1）
        from_date (datetime): 取得開始日時
        to_date (datetime): 取得終了日時
        
    Returns:
        DataFrame または None
    """
    if from_date is None or to_date is None:
        print("from_date と to_date を指定してください")
        return None

    if not mt5.initialize():
        print("MT5 の初期化に失敗しました")
        return None

    rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)
    if rates is None:
        print("データが取得できませんでした")
        shutdown_mt5()
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    shutdown_mt5()
    return df
    

# def update_current_price(symbol, interval=0.1):
#     """スレッド関数内で MT5 を初期化し、一定間隔でティック価格を取得して global に格納。"""
#     if not mt5.initialize():
#         print("MT5の初期化に失敗しました")
#         return
#     print("MT5 initialized in thread.")

#     while True:
#         tick = mt5.symbol_info_tick(symbol)
#         if tick is not None:
#             global current_price_global
#             current_price_global.clear()
#             current_price_global.append(tick)
#             print(f"Updated price: {current_price_global}")
#         else:
#             print("ティックデータ取得失敗（update_current_price）")
#         time.sleep(interval)






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
    
        
# def check_price_reached(price_to_judge):
#     """
#     ネックライン超えたか判断するための関数
#     指定した価格より現在の価格が上にあるか下にあるかを返す関数
#     True → 現在価格がprice_to_judgeよりも上にある
#     False → 現在価格がprice_to_judgeよりも下にある
#     """
#     if current_price_global >= price_to_judge:
#         return True
#     elif current_price_global <= price_to_judge:
#         return False



    
def watch_price_in_range(low,high,judged_price = current_price_global):
    low = min(low, high)
    high = max(low, high)
    if low <= judged_price <= high:
        return True
    else:
        return False
    

    



#------------------------------------------------------------------------------------
#直接はトレードに関係ない系

def save_fibonacci_to_csv(fib_data, filename="fibs.csv"):
    """
    フィボナッチ矩形情報をCSV形式で保存する（デバッグ用）。
    
    Args:
        fib_data (list): (time_start, time_end, price_lower, price_upper)のタプルのリスト。
        filename (str): 出力先ファイル名。
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_start", "time_end", "price_lower", "price_upper"])
        for f in fib_data:
            time_start = f[0].strftime('%Y-%m-%d %H:%M') if hasattr(f[0], 'strftime') else f[0]
            time_end   = f[1].strftime('%Y-%m-%d %H:%M') if hasattr(f[1], 'strftime') else f[1]
            writer.writerow([time_start, time_end, f[2], f[3]])

def calculate_sma(df, window=20, name = "SMA?"):
    """
    DataFrameに対してSMA（単純移動平均）を計算し、'sma25'列として追加する。
    
    Args:
        df (DataFrame): 入力データ（少なくとも'close'列が必要）。
        window (int): 移動平均のウィンドウサイズ（デフォルト20）。
        
    Returns:
        DataFrame: SMA計算後のデータフレーム。
    """
    df = df.copy()  # 明示的なコピー作成
    df[name] = df['close'].rolling(window=window).mean()
    df[name] = df[name].ffill()
    df = df.reset_index(drop=True)
    return df


def update_sma(df, window, name):
    """
    df の最後の行のみ、直近 window 本の 'close' 値から SMA を計算し、name カラムを更新する。
    """
    # もしデータ数が window 未満なら、全体平均を計算
    df = df.copy()  # コピーを作成
    if len(df) < window:
        sma_value = df['close'].mean()
    else:
        sma_value = df['close'].iloc[-window:].mean()
    df.loc[df.index[-1], name] = sma_value
    return df

def determine_trend(df,name):
    """
    SMAの時系列データから、前の値と比較して上昇しているかを判定するリストを作成する。
    
    Args:
        sma_series (Series): SMAの時系列データ。
    
    Returns:
        list: 各インデックスでの上昇(True) or 非上昇(False)のリスト。
    """

    up_down_list = [False]  # 先頭は比較対象がないのでFalse
    for i in range(1, len(df)):
        up_down_list.append(df[name][i] > df[name][i-1])
    df["UP_DOWN"] = up_down_list
    return df

def update_determine_trend(df,name):
    if df.loc[df.index[-2], name] < df.loc[df.index[-1], name]:
        df.loc[df.index[-1], "UP_DOWN"] = True
    elif df.loc[df.index[-2], name] > df.loc[df.index[-1], name]:
        df.loc[df.index[-1], "UP_DOWN"] = False
    else:
        df.loc[df.index[-1], "UP_DOWN"] = None
    # print(f"最新df:{df.loc[df.index[-1]]}")
    return df

def detect_pivots(df, name, POINT_THRESHOLD=0.01, LOOKBACK_BARS=15, consecutive_bars=3, arrow_spacing=10):
    """
    指定したSMA列（name）の値とUP_DOWNカラムをもとに、トレンド転換のピボット（高値または安値）を検出する。
    連続して同じUP_DOWNが続くバー数がconsecutive_bars以上であり、かつ前回のピボットからarrow_spacing以上離れている場合に
    ピボットを検出するようにしています。

    Args:
        df (DataFrame): 時系列データ（'time', 'close', 'high', 'low', <name>、"UP_DOWN"などが含まれる）
        name (str): SMA計算済みのカラム名（例："BASE_SMA" や "SML_SMA"）
        POINT_THRESHOLD (float): 転換判定の閾値（SMAの極値との差がこの値以上なら転換とみなす）
        LOOKBACK_BARS (int): 遡るバー数。直近のこの期間でのSMAの最高値／最低値を求める
        consecutive_bars (int): 現在のトレンドを示す連続したバーの数がこの値以上でないとピボット検出を行わない
        arrow_spacing (int): 前回ピボット検出後、次のピボット検出までに最低このバー数は間隔を空ける

    Returns:
        list: 各ピボット情報を (time, pivot_value, pivot_type) のタプルでまとめたリスト
    """
    pivot_data = []
    last_pivot_index = -arrow_spacing  # 最初は十分離れていると仮定
    run_counter = 1  # 現在の連続数。最初のバーは1とする

    # up_down_listは計算済みの"UP_DOWN"カラム（True:上昇、False:下降）のリスト
    up_down_list = df["UP_DOWN"].tolist()

    # 初期のトレンドは、最初の連続が足りなければ仮にFalse（下降）と設定
    # ※より厳密には初期の連続状態で決定すべきですが、ここではシンプルにFalseとする
    up_trend = False  
    prev_h_or_l_index = None

    # ループはインデックス1から開始（最初はrun_counterを更新するため）
    for i in range(1, len(df)):
        # 連続して同じ方向かどうかチェック
        if up_down_list[i] == up_down_list[i - 1]:
            run_counter += 1
        else:
            run_counter = 1

        # 連続期間が十分でなければ、ピボット検出は行わない
        if run_counter < consecutive_bars:
            continue

        # 前回ピボットからarrow_spacing以上離れていなければスキップ
        if i - last_pivot_index < arrow_spacing:
            continue

        # 転換検出処理（上昇→下降の場合と下降→上昇の場合で分岐）
        # ※ここでは「転換」が起こったかどうかをSMAの極値との差で判断する
        if up_trend and not up_down_list[i]:
            # 直前の上昇トレンド中から下落に転じた場合（高値ピボット）
            pivot_index = i
            # LOOKBACK期間内のSMAを抽出
            sma_slice = df[name][pivot_index - LOOKBACK_BARS: pivot_index + 1]
            sma_highest = sma_slice.max()
            current_sma = df[name].iloc[pivot_index]

            if (sma_highest - current_sma) >= POINT_THRESHOLD:
                # ピボットとする場合：該当期間中の高値（"high"カラム）も調べる
                hs = df['high'][pivot_index - LOOKBACK_BARS: pivot_index + 1]
                highest_index = hs.idxmax()
                highest = hs.max()
                highest_datetime = df["time"].iloc[highest_index]
                pivot_data.append((highest_datetime, highest, "high"))
                last_pivot_index = i
                up_trend = False
                # オプション：prev_h_or_l_index を更新（今回は単純化のため割愛）
        elif (not up_trend) and up_down_list[i]:
            # 下降トレンドから上昇に転じた場合（安値ピボット）
            pivot_index = i
            if pivot_index - LOOKBACK_BARS < 0:
                continue
            sma_slice = df[name][pivot_index - LOOKBACK_BARS: pivot_index + 1]
            sma_lowest = sma_slice.min()
            current_sma = df[name].iloc[pivot_index]

            if (current_sma - sma_lowest) >= POINT_THRESHOLD:
                ls = df['low'][pivot_index - LOOKBACK_BARS: pivot_index + 1]
                lowest_index = ls.idxmin()
                lowest = ls.min()
                lowest_datetime = df["time"].iloc[lowest_index]
                pivot_data.append((lowest_datetime, lowest, "low"))
                last_pivot_index = i
                up_trend = True
                # 同様に、prev_h_or_l_index の更新を行う場合はここで設定

    return pivot_data




def update_detect_pivot(df, name, point_threshold, lookback_bars, consecutive_bars, arrow_spacing, window=1000):
    """
    df の最後 window 行のみを対象にピボット検出を行い、
    最新のピボットイベントがあれば、元の df の該当行の "Pivot" カラムを更新する。
    戻り値は検出された最新のピボットイベント（タプル）または None。
    """
    # 最新の window 行をコピーして subset_df を作成
    subset_df = df.iloc[-window:].copy().reset_index(drop=True)
    # subset_df の UP_DOWN リストを取得
    up_down_list = subset_df["UP_DOWN"].tolist()
    # detect_pivots() は (pivot_data, updated_subset_df) を返すので、pivot_data を抽出
    
    pivots = detect_pivots(subset_df, name, POINT_THRESHOLD=point_threshold, 
                              LOOKBACK_BARS=lookback_bars, consecutive_bars=consecutive_bars, 
                              arrow_spacing=arrow_spacing)
    if pivots:
        # 最新のピボットイベント
        last_pivot = pivots[-1]
        
        pivot_time, pivot_price, pivot_type = last_pivot
        # 元の df で pivot_time と一致する行のインデックスを取得
        idx = df.index[df["time"] == pivot_time]
        if len(idx) > 0:
            # "high" なら True、"low" なら False として記録（必要に応じて値を変更してください）
            df.loc[idx[0], "Pivot"] = True if pivot_type == "high" else False
        return last_pivot
    return None


def save_pivots_to_csv(pivot_data, filename="pivots.csv"):
    """
    ピボット情報をCSV形式で保存する（デバッグ用）。
    
    Args:
        pivot_data (list): (datetime, price, type)のタプルのリスト。
        filename (str): 出力ファイル名。
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "price", "type"])
        for row in pivot_data:
            # datetimeはstrftimeできる場合のみ
            time_str = row[0].strftime('%Y-%m-%d %H:%M') if hasattr(row[0], 'strftime') else row[0]
            writer.writerow([time_str, row[1], row[2]])

import pytz



def process_data(symbol="USDJPY", tp_level=160, output_file="trade_logs.csv"):
    global last_pivot_data, sml_last_pivot_data, current_price_global, current_df

    pivot_data =[]
    sml_pivot_data = []

    if not initialize_mt5():
        return
    
    print("実行中")
    timezone = pytz.timezone("Etc/UTC")
    fromdate = datetime(2023, 1, 18, 0, 0, tzinfo=timezone)
    todate   = datetime(2025, 2, 20, 6, 50, tzinfo=timezone)

    original_df = fetch_data_range(symbol,fromdate, todate)
    if original_df is None:
        shutdown_mt5()
        return

    wm = WaveManager()

    # 1. SMAを計算しSMAの行をdfに追加する
    
    df = original_df.iloc[:1600].copy()
    sml_df = original_df.iloc[:1600].copy()
    
# 初期のSMA計算（この時点では200本分）
    df = calculate_sma(df.copy(), window=20, name="BASE_SMA")
    sml_df = calculate_sma(sml_df.copy(), window=4, name="SML_SMA")

    determine_trend(df,"BASE_SMA")
    determine_trend(sml_df,"SML_SMA")

    pivot_data = detect_pivots(df.copy(), POINT_THRESHOLD=0.008, LOOKBACK_BARS=15, name = "BASE_SMA",arrow_spacing=8)
    sml_pivot_data = detect_pivots(sml_df.copy(), POINT_THRESHOLD=0.001, LOOKBACK_BARS=3,consecutive_bars=1,name="SML_SMA", arrow_spacing=1)

    last_pivot_data = pivot_data[-1]
    sml_last_pivot_data = sml_pivot_data[-1]

    print(f"開始時間：{df.iloc[-1]["time"]}")
    
    for idx in range(1600, len(original_df)):

    # 新しいローソク足データ（1行）を取得
        new_row = original_df.copy().iloc[idx:idx+1]
        # DataFrameに追加
        df = pd.concat([df, new_row], ignore_index=True)
        sml_df = pd.concat([sml_df, new_row], ignore_index=True)
        
        # SMAを再計算（全体のデータに対して計算する場合）
        df = calculate_sma(df, window=20, name="BASE_SMA")
        sml_df = calculate_sma(sml_df, window=4, name="SML_SMA")
        
        # トレンド判定
        update_determine_trend(df,"BASE_SMA")
        update_determine_trend(sml_df,"SML_SMA")

        # 3. 20MAのピボット検出
        # --- ピボット検出の更新（直近ウィンドウのみ） ---
        new_pivot = update_detect_pivot(df, point_threshold=0.009, lookback_bars=15, consecutive_bars=3, arrow_spacing=8, name = "BASE_SMA")
        if new_pivot is not None and new_pivot != last_pivot_data:
            last_pivot_data = new_pivot
            pivot_data.append(new_pivot)
            wm.append_pivot_data(last_pivot_data, df, sml_df)
            if new_pivot[2] == "high":
                wm.add_session(pivot_data[-2:], up_trend="False")
            else:
                wm.add_session(pivot_data[-2:], up_trend="True")


        # 4. 4MAのピボット検出
        sml_new_pivot = update_detect_pivot(
            sml_df,
            point_threshold=0.003,
            lookback_bars=3,
            consecutive_bars=1,
            name="SML_SMA", 
            arrow_spacing=1
        )    
        if sml_new_pivot is not None and sml_new_pivot != sml_last_pivot_data:
            sml_last_pivot_data = sml_new_pivot
            sml_pivot_data.append(sml_new_pivot)
            wm.append_sml_pivot_data(sml_last_pivot_data)
        
        # ここでは、過去の十分なデータ（例：直近100行）を含めたウィンドウを送るのではなく、
        # 最新の1行のみを取り出す場合の例です。
        if not df.empty:
            current_df = df.tail(1)
        else:
            continue  # 空なら次のループへ

        
        global current_price_global
        current_price_global.clear()
        current_price_global.append(current_df.iloc[-1])

 
        
        wm.send_candle_data_tosession(df.iloc[-1100:].copy(), sml_df.iloc[-1100:].copy())

    wm.export_trade_logs_to_csv(wm.trade_logs, filename=output_file)


if __name__ == "__main__":
    process_data()