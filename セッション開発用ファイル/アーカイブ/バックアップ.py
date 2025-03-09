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
import pytest

np.set_printoptions(threshold=np.inf)
# （Machine のimportは重複しているので必要に応じて整理してください）

# セッション管理用のグローバル WaveManager インスタンス
current_price_global = []


symbol = "USDJPY"  # デフォルト値
last_pivot_data = 999
sml_last_pivot_data = 999

states = [
    "created_base_arrow",
    "created_new_arrow",
    {"name": "infibos", "children": "has_determined_neck", 'initial': False},
    "has_position",
    "closed"
]

# 遷移定義
transitions = [
    {"trigger": "create_new_arrow", "source": "created_base_arrow", "dest": "created_new_arrow"},
    {"trigger": "touch_37", "source": ["created_new_arrow", "infibos_has_determined_neck"], "dest": "infibos"},
    {"trigger": "neck_determine", "source": "infibos", "dest": "infibos_has_determined_neck"},
    {"trigger": "build_position", "source": ["infibos", "infibos_has_determined_neck"], "dest": "has_position"},
    {"trigger": "close", "source": "has_position", "dest": "closed"}
]


#############################################
# セッションクラス（各セッションの状態を管理）
#############################################
class MyModel(object):
    def __init__(self, name, start_index, start_time_index,  prev_index, prev_time_index, up_trend="True"):
        self.name = name
        self.full_data = None
        self.start_index = start_index
        self.start_time_index = start_time_index
        self.prev_index = prev_index
        self.prev_time_index = prev_time_index
        # self.pivot_data = pivot_data[-2:]  # セッション開始時点のピボットデータのコピー
        # self.start_pivot = pivot_data[-1] if pivot_data else datetime.now()
        self.machine = Machine(model=self, states=states, transitions=transitions, initial="created_base_arrow")
        self.new_arrow_index = None  # 推進波の終わりの最高（安）値を保管（以降調整波と考える）
        self.next_new_arrow_index = None
        self.fibo_minus_20 = None
        self.fibo_minus_200 = None
        
        self.base_fibo37 = None  # 推進波に対する37%リトレースメントライン
        self.base_fibo70 = None # 推進波に対する70%リトレースメントライン
        self.index_of_fibo37 = None
        self.final_neckline_index = None
        self.time_of_goldencross = None
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
        

        # 状態に応じた処理関数のディスパッチテーブル
        self.state_actions = {
            "created_base_arrow": self.handle_created_base_arrow,
            "created_new_arrow": self.handle_created_new_arrow,
            "infibos": self.handle_infibos,
            "infibos_has_determined_neck": self.handle_infibos_has_determined_neck,
            "has_position": self.handle_has_position,
            "closed": self.handle_closed
        }


    np.set_printoptions(threshold=np.inf)
    def execute_state_action(self):
        """現在の状態に対応する処理関数を実行"""
        assert self.start_index == 231, f"start_index が不一致。期待値: 251, 実際: {self.start_index}"
        assert self.prev_index == 216, f"start_index が不一致。期待値: 219, 実際: {self.prev_index}"
        assert self.full_data[self.start_index,3] == 151.683, f"start_index の価格が不一致。実際: {self.full_data[self.start_index,3]}"
        while self.destroy_reqest == False:
            action = self.state_actions.get(self.state)
            if action:
                action()

    def handle_created_base_arrow(self):
        total_len = len(self.full_data)
        base_pivots_index = self.get_pivots_in_range(self.start_time_index+1, total_len-1,base_or_sml="base")
        indices_after = base_pivots_index[base_pivots_index > self.start_time_index]


        if indices_after.size >= 2:
            # new_time_ns = self.full_data[indices_after[1], 9]
            # new_time_readable = pd.to_datetime(new_time_ns, unit='ns', utc=True)
            # print("new_Arrowのいんでっくす", self.new_arrow_index, new_time_ns, "→", new_time_readable)
            self.new_arrow_index = self.find_detection_index(self.full_data[indices_after[0],9],0)
            self.new_arrow_detection_index = self.find_detection_index(self.full_data[self.new_arrow_index,0],9)
            self.next_new_arrow_index = self.find_detection_index(self.full_data[indices_after[1],9],0)

            assert self.full_data[self.new_arrow_index,[2]] == 151.854, f"new_arrowが正しくない。実際は{self.full_data[self.new_arrow_index,2]}"
            assert self.new_arrow_detection_index == 302, f"self.new_arrow_detection_indexが正しくない。実際は{self.new_arrow_detection_index}"
            assert self.full_data[self.next_new_arrow_index ,3] == 151.782, f"ねくすとnew_arrowが正しくない。実際は{self.full_data[self.next_new_arrow_index ,3]}"

        else:
            self.destroy_reqest = True
            return
        
        
        if self.up_trend:
            prices = []
            prices = self.full_data[self.prev_index,2], self.full_data[self.start_index,3]
            
            _,_,self.fibo_minus_20,self.fibo_minus_200, = detect_extension_reversal(prices,None, None, 0.2, 2)
            judged_price = self.full_data[self.new_arrow_index,2]
            assert judged_price == 151.854, f"ジャッジプライス(new_arrow)が正しくない。実際は{judged_price}"
        elif not self.up_trend:
            prev_price, start_price = self.full_data[self.prev_index,3], self.full_data[self.start_index,2]
            self.fibo_minus_20, self.fibo_minus_200,_,_ = detect_extension_reversal(prev_price, start_price,-0.2,-2,None, None)
            judged_price = self.full_data[self.new_arrow_index,3]

        if watch_price_in_range(self.fibo_minus_20,self.fibo_minus_200,judged_price):
            self.create_new_arrow()
        else:
            self.destroy_reqest == True

    def handle_created_new_arrow(self):
        #on_enterでtouch_37を発動できなかった時の処理
        required_data = self.full_data[self.new_arrow_detection_index:]
        while self.state == "created_new_arrow" and not self.destroy_reqest:
            for i in range(len(required_data)):
                pre_check = (not check_touch_line(self.base_fibo37, required_data[i, 3])
                            if self.up_trend else
                            check_touch_line(self.base_fibo37, required_data[i, 2]))
                if pre_check:
                    if self.price_in_range_while_adjustmen(self.new_arrow_detection_index,
                                                           self.new_arrow_detection_index + i):
                        self.index_of_fibo37 = i
                        self.touch_37()
                        return
                    else:
                        self.destroy_reqest = True
                        break
                if i == len(required_data) - 1:
                    self.destroy_reqest = True


    def handle_infibos(self):
        required_data = self.full_data[self.index_of_fibo37:self.index_of_fibo37+200]
        while self.state == "infibos":
            for i in range(len(required_data)):
                if self.potential_neck[0]:
                    neck_price = self.full_data[self.potential_neck[0],2]
                    entry_result = self.potential_entry(required_data[i], neck_price, self.index_of_fibo37+i+1)
                    if entry_result is True:
                        self.build_position()
                        break
                    elif entry_result is False:
                        self.potential_neck = []
                elif len(self.determined_neck) > 0:
                    self.neck_determine()
                    break

                if not self.price_in_range_while_adjustment():
                    break
                if 100 < required_data[i,13] <165:
                    self.append_sml_pivot_data(required_data, i)

    def handle_infibos_has_determined_neck(self):
        if self.potential_neck:
            neck_price = self.full_data[self.potential_neck]
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
                        self.entry_index = df.iloc[-1]
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
                        self.entry_index = df.iloc[-1]
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



    def on_enter_created_new_arrow(self):
        state_name = self.state
        self.state_times[state_name] = self.new_arrow_index

        if self.up_trend is True:
            # print(self.full_data[self.start_index])
            prices = [self.full_data[self.start_index,3],self.full_data[self.new_arrow_index,2]]
            print("プライス達",prices)
            self.base_fibo70, _, self.base_fibo37, _ = detect_extension_reversal(prices, lower1_percent=0.3, higher1_percent=-0.37)
        else:
            prices = [self.full_data[self.start_index,2],self.full_data[self.new_arrow_index,3]]
            self.base_fibo37, _, self.base_fibo70, _ = detect_extension_reversal(prices, lower1_percent=0.37, higher1_percent=-0.3)

        self.get_golden_cross_index()
        assert self.index_of_goldencross == 289, f"ゴールデンクロスインデックスが違う。実際は{self.index_of_goldencross}"
        self.sml_pivots_after_goldencross = self.get_pivots_in_range(self.index_of_goldencross,
                                                                    self.new_arrow_detection_index,
                                                                    base_or_sml="sml")
        assert self.sml_pivots_after_goldencross.size == 2, f"スモールpivotの数が違う。実際は{self.sml_pivots_after_goldencross.size}"
        highest, lowest = self.get_high_and_low_in_term(self.index_of_goldencross,
                                                        self.new_arrow_detection_index,
                                                        False)
        if self.up_trend:
            if not watch_price_in_range(self.base_fibo37, self.base_fibo70, lowest):
                print(self.start_time_index)
                self.index_of_fibo37 = self.get_touch37_index()
                assert self.index_of_fibo37 == 292, f"self.index_of_fibo37が違う。実際は{self.index_of_fibo37}"
                self.touch_37()
                return
        else:
            if watch_price_in_range(self.base_fibo37, self.base_fibo70, highest):
                self.index_of_fibo37 = self.get_touch37_index()
                self.touch_37()
                return
        

    def on_enter_infibos(self):
        self.get_potential_neck_wheninto_newarrow()

    def on_enter_has_position(self):
        assert self.neckline_index == 299, f"ネックラインが違う。実際は{self.neckline_index}"
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

    def find_detection_index(self, target_time,col):
        """
        np_arr の 0 列から、target_time と等しい値を持つ行のインデックスを返す関数です。
        target_time は int64 の Unix タイムスタンプ（ナノ秒）であるか、
        または np.datetime64 型の場合は自動で変換。
        """
        # target_time が np.datetime64 の場合、int64 に変換する
        if isinstance(target_time, np.datetime64):
            target_time = target_time.astype("int64")
        
        indices = np.where(self.full_data[:, col] == target_time)[0]
        if indices.size > 0:
            return indices[0]
        else:
            return None
        





    def get_high_and_low_in_term(self,start_index,end_index=None,close=None):
        """
        dfから、指定した期間以降の最高値と最安値を検出しreturnする
        """
        
        if end_index:
            required_data = self.full_data[start_index:end_index]
        else:
            required_data = self.full_data[start_index:start_index+1]
        highest_price = required_data[:,2].max()
        lowest_price = required_data[:,3].min()
        if close:
            highest_close = required_data[:,4].max()
            lowest_close = required_data[:,4].min()
            return highest_price, lowest_price, highest_close, lowest_close
        else:
            return highest_price, lowest_price

    def get_golden_cross_index(self):
        """
        dfは過去100本のローソクのデータ(datetime型のtime,open,close,high,low,20MAの値など)
        sml_dfは過去100本のローソクのデータ(datetime型のtime,open,close,high,low,4MAの値など)
        基準となるBASE_SMAと一つ下のフラクタル構造のSML_SMAのゴールデンクロスを起こした時間を検出
        調整波の始まり時間をゴールデンクロス基準で把握し、それ以降の最も深い調整位置を知るためのメソッド。
        このゴールデンクロスを起こした後にBASE_SMAが調整方向に転換していてtouch37を満たせば
        調整波として完全に基準を満たしていると判断することができる
        USDJPY 2/19 13:22付近でのエントリーみたいなのをなくすための措置
        """
        
        base_sma_since_new_arrow = self.full_data[self.new_arrow_index:self.next_new_arrow_index+1,7]
        sml_sma_since_new_arrow = self.full_data[self.new_arrow_index:self.next_new_arrow_index+1,8]
        print(base_sma_since_new_arrow,sml_sma_since_new_arrow)

        if self.up_trend is True:
            conditions = np.where(base_sma_since_new_arrow > sml_sma_since_new_arrow)
            
        elif self.up_trend is False:
            conditions = np.where(base_sma_since_new_arrow < sml_sma_since_new_arrow)
        
        if conditions[0].size > 0:
            print(conditions[0][0]+self.new_arrow_index)
            self.index_of_goldencross = conditions[0][0] + self.new_arrow_index
        else:
            self.index_of_goldencross = self.new_arrow_index


        
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
        


    def get_pivots_in_range(self,idx1,idx2,base_or_sml):
        """
        goldencross以降のsmall_pivotsのデータを取得するメソッド
        """
        pivot_arr = self.full_data[idx1:idx2+1, 10] if base_or_sml == "base" else self.full_data[idx1:idx2+1, 13]
        valid = ~np.isnan(pivot_arr)
        valid_indices = np.where(valid)[0] + idx1
        print(valid_indices)
        return valid_indices

    # self.start_index 以降の有効なインデックスだけ抽出して、最初の要素を選ぶ
        

    def get_potential_neck_wheninto_newarrow(self):
        """
        ゴールデンクロス以降のsml_pivots(sml_pivots_after_goldencross)を
        sml_pvtsに格納し、その中にネックラインになりうるpivot(上昇トレンド中の調整のhigh
        下降トレンド中の調整のlow)があれば、その次のsml_pvtsの戻しの深さ次第で
        determined_neckに入れる。その価格を超えたらエントリーできる点。
        ネックラインになりうるpivotの次のpivotが生成されてなければ
        potential_neckに格納。(この場合次のpivot確定待ち)
        """
        potential_neck_detection = None
        sml_pvts = self.sml_pivots_after_goldencross
        print(sml_pvts)
        if len(sml_pvts) >= 2:
            for i in range(1, len(sml_pvts)):
                    if self.up_trend is True:
                        if self.full_data[sml_pvts[i],14] == 1 and sml_pvts[i] > self.index_of_fibo37:
                            if i + 1 < len(sml_pvts):
                                prices = self.full_data[sml_pvts[i],2],self.full_data[sml_pvts[i-1],3]
                                fibo32_of_ptl_neck = detect_extension_reversal(prices,-0.32,0.32,None,None)
                                if watch_price_in_range(fibo32_of_ptl_neck[0],fibo32_of_ptl_neck[1],self.full_data[i+1,3]):
                                    self.determined_neck.append(sml_pvts[i])
                                    self.organize_determined_neck()
                            else:
                                potential_neck_detection = sml_pvts[i]
                    else:
                        if self.full_data[sml_pvts[i],14] == 0 and sml_pvts[i] > self.index_of_fibo37:
                            if i + 1 < len(sml_pvts):
                                prices = self.full_data[sml_pvts[i],3],self.full_data[sml_pvts[i-1],2]
                                fibo32_of_ptl_neck = detect_extension_reversal(prices,-0.32,0.32,None,None)
                                if watch_price_in_range(fibo32_of_ptl_neck[0],fibo32_of_ptl_neck[1],self.full_data[i+1,3]):
                                    self.determined_neck.append(sml_pvts[i])
                                    self.organize_determined_neck()
                            else:
                                potential_neck_detection = sml_pvts[i]
            if potential_neck_detection:
                neckline = self.find_detection_index(self.full_data[potential_neck_detection,12],0)
                self.potential_neck.append(neckline)

        assert len(self.potential_neck) == 1, f"ポテンシャルネックの数が違う。実際は{len(self.potential_neck)}"
        assert self.potential_neck[0] == 299 , f"ポテンシャルネックのインデックスが間違い。実際は{self.potential_neck}"
                

        

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
    #necklineは数字で入れる
    def potential_entry(self, arr, neckline, end_index):
        self.highlow_since_new_arrow = self.get_high_and_low_in_term(self.index_of_fibo37,
                                                                    end_index=end_index)
        if self.up_trend is True:
            if arr[2] > neckline and arr[2] > arr[7]:    #and self.check_no_SMA(df.iloc[-1050:],neckline[-1][1]):
                self.stop_loss = self.highlow_since_new_arrow[1] - 0.006
                prices_data_to_get_take_profit = self.start_index , self.new_arrow_detection_index
                highlow = detect_extension_reversal(prices_data_to_get_take_profit, higher1_percent=0.32)
                self.take_profit = highlow[2]
                self.entry_line = neckline + 0.002
                self.entry_index = end_index -1
                self.point_to_stoploss = abs(self.entry_line - self.stop_loss)
                self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                return True
            elif arr[2] > neckline and not arr[2] > arr[7]:
                return False
            #self.check_no_SMA(df.iloc[-1050:],neckline[-1][1]) is False:
                

        else:
            if arr[3] < neckline and arr[3] > arr[7]:
                self.stop_loss = self.highlow_since_new_arrow[0] + 0.006
                pivots_data_to_get_take_profit = self.start_pivot, self.new_arrow_pivot
                highlow = detect_extension_reversal(pivots_data_to_get_take_profit, lower1_percent=-0.32)
                self.take_profit = highlow[0]
                self.entry_line = neckline[-1][1] - 0.002
                self.entry_index = end_index -1
                self.point_to_stoploss = abs(self.entry_line - self.stop_loss)
                self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                # print(f"エントリー記録：：：　アプトれ{self.up_trend}, {self.name}、ネック：{neckline}、エントリーライン：{self.entry_line}、エントリーピボット：{self.entry_pivot}, テイクプロフィット：{self.take_profit}")
                return True
            elif arr[3] < neckline and not arr[3] > arr[7]:
                return False

    def append_sml_pivot_data(self, required_data, new_sml_pivot_index):
        """
        mainでlast_pivot_dataが更新されたら受け取って各セッションのpivot_dataに追加
        touched_20の場合推進波の形成が終わったサインとしてcreated_new_arrowに移る
        "created_base_arrow"の場合touched_20に移る前 (推進波になる前)に波終了でセッション削除
        """
        state_list = ["infibos", "infibos_has_determined_neck"]
        avoid_list = ["created_base_arrow", "has_position", "closed"]
        # if self.state not in avoid_list:
        #     self.sml_pivot_data.append(new_sml_pivot_index)

        if self.new_arrow_index is not None:
            self.sml_pivots_after_goldencross.append(new_sml_pivot_index)

        if self.state in state_list:
            if self.potential_neck:
                self.check_potential_to_determine_neck()
            if not self.potential_neck and self.up_trend is True and new_sml_pivot_index[14] == 1:
                new_neck_index = self.find_detection_index()
                self.potential_neck.append(new_sml_pivot_index)
            elif not self.potential_neck and self.up_trend is False and new_sml_pivot_index[14] == 0:
                self.potential_neck.append(new_sml_pivot_index)
            

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

    
        
    def get_touch37_index(self):
        required_data = self.full_data[self.index_of_goldencross:]
        if self.up_trend is True:
            required_data = required_data[:,3]
            conditions=np.where(required_data < self.base_fibo37)
        else:
            required_data = required_data[:,2]
            conditions=np.where(required_data > self.base_fibo37)

        if conditions[0].size > 0:
            return conditions[0][0] + self.index_of_goldencross
        else:
            return None

    def price_in_range_while_adjustment(self,start_index,end_index=None,close=None):
        """
        1本のバーのhigh_lowが範囲内か確認する関数
        """
        if self.up_trend:
            high, low = self.full_data[self.new_arrow_index,2], self.base_fibo70
        else:
            high, low = self.base_fibo70, self.full_data[self.new_arrow_index,3]
        
        judged_price = self.get_high_and_low_in_term(start_index,end_index,close)
        if high > judged_price[0] or low < judged_price[1]:
            return True
        else:
            self.destroy_reqest = True


def watch_price_in_range(low,high,judged_price):
    low = min(low, high)
    high = max(low, high)
    if low <= judged_price <= high:
        return True
    else:
        return False

def check_touch_line(center_price, tested_price):
    if center_price <= tested_price:
        return True
    elif center_price >= tested_price:
        return False

def detect_extension_reversal(prices,lower1_percent=None, lower2_percent=None, higher1_percent=None, higher2_percent=None):
    """
    pricesには2つの価格だけを入れる。
    low1はフィボナッチあてる2点のうち低い方の価格を0として考える。
    high1はフィボナッチあてる2点のうち高い方の価格を0として考える。
    例えば150と160のフィボの場合、low1に-0.2を入れると152
    low2に0.4を入れると154、high1に-0.2を入れると158、high2に0.2を入れると162
    """    
    
    # 前回と直近のピボットの価格を取り出す
    price1 = prices[0]
    price2 = prices[1]
    
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