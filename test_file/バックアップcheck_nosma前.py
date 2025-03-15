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
import pytz

from utils.login import initialize_mt5, shutdown_mt5
from utils.data_utils import calculate_sma, determine_trend

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
    {"trigger": "touch_37", "source": "created_new_arrow", "dest": "infibos"},
    {"trigger": "build_position", "source": "infibos", "dest": "has_position"},
    {"trigger": "close", "source": "has_position", "dest": "closed"}
]


#############################################
# セッションクラス（各セッションの状態を管理）
#############################################
class MyModel(object):
    def __init__(self, name, start_index, start_time_index, prev_index, prev_time_index, up_trend="True"):
        self.name = name
        self.start_origin = None
        self.original_offset = None
        self.full_data = None
        self.start_index = start_index
        self.start_time_index = start_time_index
        self.prev_index = prev_index
        self.prev_time_index = prev_time_index
        self.machine = Machine(model=self, states=states, transitions=transitions, initial="created_base_arrow")
        self.new_arrow_index = None  # 推進波の終わりの最高（安）値
        self.next_new_arrow_index = None
        
        self.fibo_minus_20 = None
        self.fibo_minus_200 = None
        
        self.base_fibo37 = None  # 37%リトレースメントライン
        self.base_fibo70 = None  # 70%リトレースメントライン
        self.index_of_fibo37 = None
        self.start_of_simulation = None
        self.final_neckline_index = None
        self.time_of_goldencross = None
        self.highlow_since_new_arrow = []  # 調整波の戻しの深さ
        self.sml_pivot_data = []  # touch20以降のsml_pivot記録

        # ネックライン関連
        # ここでは、各ピボットレコードを[original_index, detection_index, price, type]として保持する
        self.sml_pivots_after_goldencross = np.array([])  
        self.potential_neck = []  # この中には full_data のインデックス（scalar）を保持
        self.determined_neck = []

        self.destroy_reqest = False
        self.up_trend = True if up_trend == "True" else False
        self.state_times = {}  # 各状態移行時刻
        self.trade_log = None

        # 状態ごとの処理ディスパッチテーブル
        self.state_actions = {
            "created_base_arrow": self.handle_created_base_arrow,
            "created_new_arrow": self.handle_created_new_arrow,
            "infibos": self.handle_infibos,
            "has_position": self.handle_has_position,
            "closed": self.handle_closed
        }

    # np.set_printoptions(threshold=np.inf)
    def execute_state_action(self):
        """現在の状態に対応する処理関数を実行"""
        while not self.destroy_reqest:
            action = self.state_actions.get(self.state)
            if action:
                
                action()
        # if self.trade_log is not None and self.trade_log.size > 0:
        #     print(float(self.trade_log[6]))

    def handle_created_base_arrow(self):
        
        total_len = len(self.full_data)
        base_pivots_index = self.get_pivots_in_range(self.start_time_index + 1, total_len - 1, base_or_sml="base")
        indices_after = base_pivots_index[base_pivots_index > self.start_time_index]

        if indices_after.size >= 2:
            self.new_arrow_index = self.find_detection_index(self.full_data[indices_after[0], 9], 0)
            self.new_arrow_detection_index = self.find_detection_index(self.full_data[self.new_arrow_index, 0], 9)
            self.next_new_arrow_index = self.find_detection_index(self.full_data[indices_after[1], 9], 0)
        else:
            self.destroy_reqest = True
            return

        if self.up_trend:
            prices = (self.full_data[self.prev_index, 2], self.full_data[self.start_index, 3])
            _, _, self.fibo_minus_20, self.fibo_minus_200 = detect_extension_reversal(prices, None, None, 0.2, 2)
            judged_price = self.full_data[self.new_arrow_index, 2]
        else:
            # if self.name == "Session_3":
            #     import pdb; pdb.set_trace() 
            prices = (self.full_data[self.prev_index, 3], self.full_data[self.start_index, 2])
            self.fibo_minus_20, self.fibo_minus_200, _, _ = detect_extension_reversal(prices, -0.2, -2, None, None)
            judged_price = self.full_data[self.new_arrow_index, 3]

        if self.watch_price_in_range(self.fibo_minus_20, self.fibo_minus_200, judged_price):
            self.create_new_arrow()
            
        else:
            self.destroy_reqest = True


    def handle_created_new_arrow(self):
        # on_enterでtouch_37を発動できなかった場合の処理
        required_data = self.full_data[self.new_arrow_detection_index+1:]
        
        while self.state == "created_new_arrow" and not self.destroy_reqest:
            for i in range(len(required_data)):
                arr = required_data[i]
                # もしSMLピボットの条件（例として、arrの13列が特定のレンジにある）を満たすなら、
                # append_sml_pivot_dataを発動する
                if 100 < arr[13] < 165:
                    # if self.name == "Session_7":
                    #     import pdb; pdb.set_trace()
                    self.append_sml_pivot_data(required_data, i+1)
                
                pre_check = (not check_touch_line(self.base_fibo37, arr[3])
                            if self.up_trend else
                            check_touch_line(self.base_fibo37, arr[2]))
                
                if pre_check:
                    if self.price_in_range_while_adjustment(self.new_arrow_detection_index+i+1):
                        self.index_of_fibo37 = self.new_arrow_detection_index + i + 1
                        self.start_of_simulation = self.new_arrow_detection_index + i + 1
                        self.touch_37()
                        return
                    else:
                        self.destroy_reqest = True
                        break
                if i == len(required_data) - 1:
                    self.destroy_reqest = True
                    
    def handle_infibos(self):
        """
        infibos状態のメインループ：
        new_arrow_detection_index + 1 から最大200本分の足（required_data）を検証し、
        potential_entry や determined_neck に基づいてエントリー判定を行う。
        """
        # self.start_of_simulation = self.new_arrow_detection_index + 1
        end_of_infibos = self.start_of_simulation + 200
        required_data = self.full_data[self.start_of_simulation : end_of_infibos]
        while self.state == "infibos" and self.destroy_reqest == False:
            for local_index in range(len(required_data)):
                if self.state != "infibos":
                    break
                arr = required_data[local_index]
                # グローバルインデックス = new_arrow_detection_index+1 + local_index
                global_index = self.start_of_simulation + local_index
                
                if self.potential_neck:
                    # if global_index == 189:
                    #     import pdb; pdb.set_trace() 
                    entry_result = self.potential_entry(required_data, local_index)
                    if entry_result is True:
                        self.build_position()
                        break
                    elif entry_result is False:
                        self.potential_neck.clear()

                if self.determined_neck and "has_position" not in self.state:
                    self.highlow_since_new_arrow = self.get_high_and_low_in_term(self.start_of_simulation, global_index + 1)
                    for neckline in self.determined_neck[:]:
                        if self.state != "infibos":
                            break
                        if self.up_trend:
                            neck_price = neckline[2]
                            if arr[2] > neck_price and neck_price >= arr[7]:
                                self.highlow_stop_loss = self.highlow_since_new_arrow[1] - 0.006
                                self.sml_stop_loss = self.sml_pivots_after_goldencross[-1][2] - 0.006
                                self.final_neckline_index = neckline[1]
                                prices_data_to_get_take_profit = (self.full_data[self.start_index,3],
                                                                self.full_data[self.new_arrow_index,2])
                                highlow = detect_extension_reversal(prices_data_to_get_take_profit, higher1_percent=0.32)
                                self.take_profit = highlow[2]
                                self.entry_line = neck_price + 0.006
                                self.entry_index = global_index
                                self.point_to_stoploss = abs(self.entry_line - self.highlow_stop_loss)
                                self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                                self.build_position()
                                break
                            elif arr[2] > neck_price and neck_price < arr[7]:
                                self.determined_neck = [
                                item for item in self.determined_neck 
                                if not np.array_equal(item, neckline)
                            ]
                        else:
                            neck_price = neckline[2]
                            if arr[3] < neck_price and neck_price <= arr[7]:
                                self.highlow_stop_loss = self.highlow_since_new_arrow[0] + 0.006
                                self.sml_stop_loss = self.sml_pivots_after_goldencross[-1][2] + 0.006
                                self.final_neckline_index = neckline[1]
                                prices_data_to_get_take_profit = (self.full_data[self.start_index,2],
                                                                self.full_data[self.new_arrow_index,3])
                                highlow = detect_extension_reversal(prices_data_to_get_take_profit, lower1_percent=0.32)
                                self.take_profit = highlow[0]
                                self.entry_line = neck_price - 0.006
                                self.entry_index = global_index
                                self.point_to_stoploss = abs(self.entry_line - self.highlow_stop_loss)
                                self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                                self.build_position()
                                break
                            elif arr[3] < neck_price and neck_price > arr[7]:
                                self.determined_neck = [
                                item for item in self.determined_neck 
                                if not np.array_equal(item, neckline)
                            ]
                
                

                if not self.price_in_range_while_adjustment(global_index):
                    # if self.name == "Session_15":
                    
                    self.destroy_reqest = True
                    break
                # if self.name == "Session_3" and len(self.sml_pivots_after_goldencross) >2:
                #     import pdb; pdb.set_trace()
                if 100 < arr[13] < 165:
                    self.append_sml_pivot_data(required_data, local_index)
                    
                if len(required_data) - local_index == 1:
                    self.destroy_reqest = True
                            

    def handle_has_position(self):
        pass

    def handle_closed(self):
        if not self.destroy_reqest:
            self.destroy_reqest = True

    def on_enter_created_new_arrow(self):
        
        state_name = self.state
        self.state_times[state_name] = self.new_arrow_index
        
        if self.up_trend:
            prices = (self.full_data[self.start_index, 3], self.full_data[self.new_arrow_index, 2])
            
            self.base_fibo70, _, self.base_fibo37, _ = detect_extension_reversal(prices, lower1_percent=0.3, higher1_percent=-0.37)
        else:
            prices = (self.full_data[self.start_index, 2], self.full_data[self.new_arrow_index, 3])
            self.base_fibo37, _, self.base_fibo70, _ = detect_extension_reversal(prices, lower1_percent=0.37, higher1_percent=-0.3)

        if self.base_fibo37 is None or self.base_fibo70 is None:
            self.destroy_reqest = True
            return
        
        
        
        self.get_golden_cross_index()
        
        # sml_pivots_after_goldencross を[original_index, detection_index, price, type]で保持
        sml_indices = self.get_pivots_in_range(self.index_of_goldencross, self.new_arrow_detection_index, base_or_sml="sml")
        self.sml_pivots_after_goldencross = np.array([
            [index, actual_index, self.full_data[index, 13], self.full_data[index, 14]]
            for index in sml_indices
            if (actual_index := self.find_detection_index(self.full_data[index, 12], 0)) and self.index_of_goldencross < actual_index
        ])

        
        highest, lowest = self.get_high_and_low_in_term(self.index_of_goldencross, self.new_arrow_detection_index, False)
        
        if self.up_trend:
            if self.watch_price_in_range(self.base_fibo37, self.base_fibo70, lowest):
                self.index_of_fibo37 = self.get_touch37_index()
                self.start_of_simulation = self.new_arrow_detection_index
                self.touch_37()
            else:
                return
        else:
            if self.watch_price_in_range(self.base_fibo37, self.base_fibo70, highest):
                self.index_of_fibo37 = self.get_touch37_index()
                self.start_of_simulation = self.new_arrow_detection_index
                self.touch_37()
                return
            else:
                return

    def on_enter_infibos(self):
        
        self.get_potential_neck_wheninto_newarrow()

    def on_enter_has_position(self):
        # print(f"ぽじった,{self.name},{self.start_origin},えんとりー{self.entry_index},{self.entry_line}、ネック{float(self.final_neckline_index)}")
        self.close()
        
    def on_enter_closed(self):
        global_entry_index = self.original_offset + self.entry_index
        entry_time = self.full_data[self.entry_index,0]
        self.trade_log = np.array([entry_time,
                                self.up_trend,
                                global_entry_index,
                                self.entry_line,
                                self.take_profit,
                                self.highlow_stop_loss,
                                self.sml_stop_loss,
                                self.point_to_stoploss,
                                self.point_to_take_profit,
                                ])
        self.destroy_reqest = True

    def __repr__(self):
        return f"MyModel(name={self.name}, state={self.state})"

    #---------------------------------------------------------------------
    # その他の今後も特に使いそうな機能
    #---------------------------------------------------------------------

    def find_detection_index(self, target_time, col):
        """
        full_dataの指定列からtarget_timeと一致する行のインデックスを返す関数
        """
        if isinstance(target_time, np.datetime64):
            target_time = target_time.astype("int64")
        indices = np.where(self.full_data[:, col] == target_time)[0]
        if indices.size > 0:
            return indices[0]
        else:
            return None

    def get_high_and_low_in_term(self, start_index, end_index=None, close=None):
        start_index = int(start_index)
        if end_index is not None:
            end_index = int(end_index)
            # もしend_indexがstart_indexと同じか、それより小さいなら、1行だけ取得する
            if end_index <= start_index:
                end_index = start_index + 1
            required_data = self.full_data[start_index:end_index]
            if required_data.size == 0:
                print(f"空の配列が検出されました。start_index={start_index}, end_index={end_index}")
        else:
            required_data = self.full_data[start_index:start_index+1]
        
        highest_price = required_data[:, 2].max()
        lowest_price = required_data[:, 3].min()
        if close:
            highest_close = required_data[:, 4].max()
            lowest_close = required_data[:, 4].min()
            return highest_price, lowest_price, highest_close, lowest_close
        else:
            return highest_price, lowest_price


    def get_golden_cross_index(self):
        base_sma_since_new_arrow = self.full_data[self.new_arrow_index:self.next_new_arrow_index + 1, 7]
        sml_sma_since_new_arrow = self.full_data[self.new_arrow_index:self.next_new_arrow_index + 1, 8]
        if self.up_trend is True:
            conditions = np.where(base_sma_since_new_arrow > sml_sma_since_new_arrow)
        else:
            conditions = np.where(base_sma_since_new_arrow < sml_sma_since_new_arrow)
        if conditions[0].size > 0:
            self.index_of_goldencross = conditions[0][0] + self.new_arrow_index
        else:
            self.index_of_goldencross = self.new_arrow_index

    def check_no_SMA(self, df, neckline):
        df = df.copy()
        periods = [25, 75, 100, 150, 200, 375, 625, 1000]
        sma_values = []
        for period in periods:
            if len(df) < period:
                continue
            sma_col = f"SMA_{period}"
            df.loc[:, sma_col] = df['close'].rolling(window=period).mean()
            sma_value = df[sma_col].iloc[-1]
            sma_values.append(sma_value)
        if self.up_trend and neckline >= max(sma_values):
            return True
        elif not self.up_trend and neckline <= min(sma_values):
            return False
        else:
            return None 

    def get_pivots_in_range(self, idx1, idx2, base_or_sml):
        pivot_arr = self.full_data[idx1:idx2+1, 10] if base_or_sml == "base" else self.full_data[idx1:idx2+1, 13]
        valid = ~np.isnan(pivot_arr)
        valid_indices = np.where(valid)[0] + idx1
        return valid_indices

    def get_potential_neck_wheninto_newarrow(self):
        potential_neck = None
        sml_pvts = self.sml_pivots_after_goldencross
        determined_neck = None
        
        if sml_pvts.shape[0] >= 2:
            for i in range(1, sml_pvts.shape[0]):
                if self.up_trend:
                    if sml_pvts[i, 3] == 1 and sml_pvts[i, 1] > self.index_of_fibo37:
                        if i + 1 < sml_pvts.shape[0]: #determined作るには最低でも3つピボット必要
                            prices = (sml_pvts[i-1,2], sml_pvts[i,2])
                            fibo32_of_ptl_neck = detect_extension_reversal(prices, -0.32, 0.32, None, None)
                            if self.watch_price_in_range(fibo32_of_ptl_neck[0], fibo32_of_ptl_neck[1], sml_pvts[i+1, 3]):
                                determined_neck = sml_pvts[i]
                        else:
                            potential_neck = sml_pvts[i]
                else:
                    if sml_pvts[i, 3] == 0 and sml_pvts[i, 2] > self.index_of_fibo37:
                        if i + 1 < sml_pvts.shape[0]:
                            prices = (sml_pvts[i-1,2], sml_pvts[i,2])
                            fibo32_of_ptl_neck = detect_extension_reversal(prices,None, None,0.32 , -0.32)
                            if self.watch_price_in_range(fibo32_of_ptl_neck[2], fibo32_of_ptl_neck[3], sml_pvts[i+1, 2]):
                                determined_neck = sml_pvts[i]
                        else:
                            potential_neck = sml_pvts[i]
                if np.asarray(determined_neck).size > 0:
                    self.determined_neck.append(determined_neck)
                    self.organize_determined_neck()
            if potential_neck is not None:
                self.potential_neck.append(potential_neck)
                
    def check_potential_to_determine_neck(self):
        sml_pvts = self.sml_pivots_after_goldencross
        
        if self.potential_neck:
            # sml_pvtsはNumPy配列で、少なくとも3行必要とする
            if sml_pvts.shape[0] < 3:
                self.destroy_reqest = True
                return None
            # ここでFibonacci計算に使う2点を取得する
            # ・sml_pvts[-3, 0]：3つ前のpivotの検出時間（この値をfull_dataの行インデックスとして使い、列13から値を取得）
            # ・self.potential_neck[-1][1]：potential_neck候補の最新の要素の「ピボットのインデックス」（full_dataの列2から値を取得）
            pvts_to_get_32level = [
                sml_pvts[-3, 2],
                self.potential_neck[-1][2]
            ]

            
            if self.up_trend:
                
                judged_price = sml_pvts[-1][2]
                fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level, -0.32, 0.32, None, None)
                result = self.watch_price_in_range(fibo32_of_ptl_neck[0], fibo32_of_ptl_neck[1], judged_price)
                if result:
                    self.determined_neck.append(self.potential_neck[-1])
                    self.potential_neck.clear()
                    self.organize_determined_neck()
                else:
                    self.potential_neck.clear()
            else:
                judged_price = sml_pvts[-1][2]
                fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level, None, None, -0.32, 0.32)
                result = self.watch_price_in_range(fibo32_of_ptl_neck[2], fibo32_of_ptl_neck[3], judged_price)
                if result:
                    self.determined_neck.append(self.potential_neck[-1])
                    self.potential_neck.clear()
                    self.organize_determined_neck()
                else:
                    self.potential_neck.clear()


    # necklineは数字で入れる
    def potential_entry(self, required_data, local_index):
        """
        required_data:
        handle_infibos で切り出した配列（例: self.full_data[self.new_arrow_detection_index+1 : self.new_arrow_detection_index+200]）
        local_index:
        required_data 内での行番号 (0,1,2,...)
        
        この関数内でグローバルインデックスを計算し、そこでエントリー成立などの処理を行う。
        """
        
        
        global_index = self.start_of_simulation + local_index
        
        arr = required_data[local_index]
        
        
        
        sml_pvts = self.sml_pivots_after_goldencross
        if self.potential_neck:
            if sml_pvts.shape[0] < 2:
                self.potential_neck.clear()
                return None
            
        neck_price = self.potential_neck[-1][2]

        if self.up_trend:
            if arr[2] > neck_price and neck_price >= arr[7]:
                sml_index_to_get32 = (neck_price, sml_pvts[-2][2])
                fibo32_of_ptl_neck = detect_extension_reversal(sml_index_to_get32, -0.382, 0.382, None, None)
                last_highlow = self.get_high_and_low_in_term(self.potential_neck[-1][0], global_index)
                if self.watch_price_in_range(fibo32_of_ptl_neck[0], fibo32_of_ptl_neck[1], last_highlow[1]):
                    self.highlow_since_new_arrow = self.get_high_and_low_in_term(self.index_of_fibo37, global_index+1)
                    self.highlow_stop_loss = self.highlow_since_new_arrow[1] - 0.006
                    self.sml_stop_loss = last_highlow[1] - 0.006
                    self.final_neckline_index = self.potential_neck[-1][1]
                    prices_data_to_get_take_profit = (self.full_data[self.start_index,3],
                                                    self.full_data[self.new_arrow_index,2])
                    highlow = detect_extension_reversal(prices_data_to_get_take_profit, higher1_percent=0.32)
                    self.take_profit = highlow[2]
                    self.entry_line = neck_price + 0.006
                    self.entry_index = global_index
                    self.point_to_stoploss = abs(self.entry_line - self.highlow_stop_loss)
                    self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                    return True

            # ネックラインは超えてるが、SMAなどの条件を満たさない場合
            elif arr[2] > neck_price and neck_price < arr[7]:
                return False

        # --- 下降トレンドの場合 ---
        else:
            if arr[3] < neck_price and neck_price <= arr[7]:
                sml_index_to_get32 = (neck_price, sml_pvts[-2][2])
                fibo32_of_ptl_neck = detect_extension_reversal(sml_index_to_get32, None, None, -0.382, 0.382)
                last_highlow = self.get_high_and_low_in_term(self.potential_neck[-1][0], global_index)
                if self.watch_price_in_range(fibo32_of_ptl_neck[2], fibo32_of_ptl_neck[3], last_highlow[0]):
                    self.highlow_since_new_arrow = self.get_high_and_low_in_term(self.index_of_fibo37, global_index+1)
                    self.highlow_stop_loss = self.highlow_since_new_arrow[0] + 0.006
                    self.sml_stop_loss = last_highlow[0] + 0.006
                    self.final_neckline_index = self.potential_neck[-1][1]
                    prices_data_to_get_take_profit = (self.full_data[self.start_index,2],
                                                    self.full_data[self.new_arrow_index,3])
                    highlow = detect_extension_reversal(prices_data_to_get_take_profit, lower1_percent=0.32)
                    self.take_profit = highlow[0]
                    self.entry_line = neck_price - 0.006
                    self.entry_index = global_index
                    self.point_to_stoploss = abs(self.entry_line - self.highlow_stop_loss)
                    self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                    return True
            elif arr[3] < neck_price and neck_price > arr[7]:
                return False

        # 条件に当てはまらなければ None
        return None


    def append_sml_pivot_data(self, required_data, new_sml_pivot_index):
        """
        last_pivot_data更新時に、pivotレコードを追加。
        ピボットレコードは [original_index, detection_index, pivot_price, type] の4要素。
        """
        state_list = ["infibos"]
        
        if self.state == "created_new_arrow":
            actual_index = self.new_arrow_detection_index + new_sml_pivot_index
        else:
            actual_index = self.start_of_simulation + new_sml_pivot_index  # 検出したrow
        row = self.full_data[actual_index]  # キャッシュしてアクセス回数を削減
        pivot_index = self.find_detection_index(row[12], 0)  # 実際のピボットがあるindex
        price = row[13]  # ピボット価格
        type_val = row[14]  # ピボットタイプ

        new_row = np.array([actual_index, pivot_index, price, type_val])
        
        # sml_pivots_after_goldencross に対して重複チェック
        duplicate_in_sml = False
        if self.sml_pivots_after_goldencross.size != 0:
            for existing in self.sml_pivots_after_goldencross:
                if np.array_equal(existing, new_row):
                    duplicate_in_sml = True
                    break
        if not duplicate_in_sml:
            if self.sml_pivots_after_goldencross.size == 0:
                self.sml_pivots_after_goldencross = new_row.reshape(1, -1)
            else:
                self.sml_pivots_after_goldencross = np.vstack((self.sml_pivots_after_goldencross, new_row))
        
        
        # potential_neck に対しても重複チェック
        duplicate_in_potential = any(np.array_equal(existing, new_row) for existing in self.potential_neck)
        if not duplicate_in_potential:
            if self.state in state_list:
                if self.potential_neck:
                    self.check_potential_to_determine_neck()
                if not self.potential_neck:
                    if self.up_trend and type_val == 1:
                        self.potential_neck.append(new_row)
                    elif (not self.up_trend) and type_val == 0:
                        self.potential_neck.append(new_row)



    def organize_determined_neck(self):
        # None 以外の要素のみで再構成
        self.determined_neck = [item for item in self.determined_neck if item is not None]
        result = []
        if self.up_trend:
            # 上昇トレンドの場合：価格が大きいものを優先
            for item in self.determined_neck:
                while result and item[2] > result[-1][2]:
                    result.pop()
                result.append(item)
        else:
            # 下降トレンドの場合：価格が小さいものを優先
            for item in self.determined_neck:
                while result and item[2] < result[-1][2]:
                    result.pop()
                result.append(item)
        self.determined_neck = result
        
    def get_touch37_index(self):
        
        required_data = self.full_data[self.index_of_goldencross:]
        if self.up_trend:
            required_data = required_data[:, 3]
            conditions = np.where(required_data < self.base_fibo37)
        else:
            required_data = required_data[:, 2]
            conditions = np.where(required_data > self.base_fibo37)
        if conditions[0].size > 0:
            return conditions[0][0] + self.index_of_goldencross
        else:
            return None

    def price_in_range_while_adjustment(self, start_index, end_index=None, close=None):
        
            
        if self.up_trend:
            high = self.full_data[self.new_arrow_index, 2]
            low = self.base_fibo70
        else:
            high = self.base_fibo70
            low =  self.full_data[self.new_arrow_index, 3]
        judged_price = self.get_high_and_low_in_term(start_index, end_index, close)
        if high > judged_price[0] and low < judged_price[1]:
            return True
        else:
            self.destroy_reqest = True
            
    def watch_price_in_range(self, low, high, judged_price):
        
        low_val = min(low, high)
        high_val = max(low, high)
        return True if low_val <= judged_price <= high_val else False
        


#############################################
# マネージャークラス
#############################################
class WaveManager(object):
    def __init__(self):
        self.sessions = {}  # セッションは session_id をキーに管理
        self.next_session_id = 1
        self.trade_logs = []
        self.full_data = []


    def analyze_sessions(self):
        for session_id in list(self.sessions.keys()):
            session = self.sessions[session_id]
            session.start_origin = session.start_index
            session.full_data = self.full_data[session.prev_index-100:session.start_index+1500, :]
            difference = session.start_time_index - session.start_index
            session.start_index = session.start_index - session.prev_index + 100
            session.start_time_index = session.start_index + difference
            session.prev_index = 100
            session.execute_state_action()
            if session.trade_log is not None and session.trade_log.size > 0:
                self.trade_logs.append(session.trade_log)  # trade_log をリストに追加
            del self.sessions[session_id]
        # 全セッションの処理が終わったら、リストを NumPy 配列に変換
        if self.trade_logs:
            #  trade_logs は各行の最初の要素が entry_time だと仮定
            self.trade_logs = sorted(self.trade_logs, key=lambda log: log[0])
            self.trade_logs = np.vstack(self.trade_logs)
            self.organize_trade_logs()
            self.trade_logs.to_csv("test_result/usdjpyもういっかい.csv")
            print("ログ",self.trade_logs)
            print(len(self.trade_logs))
        
    def organize_trade_logs(self):
        columns = ["entry_time","up_trend","global_entry_index", "entry_line", "take_profit", "highlow_stop_loss", "sml_stop_loss", "point_to_stoploss", "point_to_take_profit"]
        time_columns = ["entry_time"]
        trade_logs = pd.DataFrame(self.trade_logs, columns=columns)
        time_columns = ["entry_time", "exit_time", "order_time"]  # 変換したい時間カラム名のリスト
        for col in time_columns:
            if col in trade_logs.columns:
                trade_logs[col] = pd.to_datetime(trade_logs[col], unit="ns", utc=True)
        self.trade_logs = trade_logs


    def add_session(self, start_index, start_time_index, prev_index, prev_time_index, up_trend):
        """
        新しいセッションを生成して管理リストに追加する。
        """
        
        session = MyModel(f"Session_{self.next_session_id}", start_index, start_time_index, prev_index, prev_time_index, up_trend)
        session.original_offset = prev_index - 100
        self.sessions[self.next_session_id] = session
        self.next_session_id += 1
        return session

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


def detect_extension_reversal(prices, lower1_percent=None, lower2_percent=None, higher1_percent=None, higher2_percent=None):
    """
    pricesには2つの価格のみを入れる。
    """
    price1 = prices[0]
    price2 = prices[1]
    low_val = min(price1, price2)
    high_val = max(price1, price2)
    wave_range = high_val - low_val
    if lower1_percent is not None:
        low1 = low_val - (-wave_range * lower1_percent)
    else:
        low1 = None
    if higher1_percent is not None:
        high1 = high_val - (-wave_range * higher1_percent)
    else:
        high1 = None
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


#-----------------------------------------------




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
    last_pivot_index = 0  # 前回検出したピボットのインデックス
    last_detect_index = 0
    run_counter = 1                   # 連続する同一トレンドのカウンター
    n = np_arr.shape[0]
    # 初期の up_trend 状態は、ここでは False（下降状態）として開始
    up_trend = None

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
        
        if up_trend is None:
            if trend_arr[i] == 0.0:
                # もし転換が下降に向かっているなら、これまで上昇していたとみなし
                up_trend = True
            elif trend_arr[i] == 1.0:
                # もし転換が上昇に向かっているなら、これまで下降していたとみなす
                up_trend = False

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
                
                if name == "BASE_SMA" and last_pivot_index > 150 and n - i > 1500:
                    wm.add_session(start_index=local_high_idx, start_time_index = i,  prev_index=last_pivot_index, prev_time_index = last_detect_index, up_trend="False")
                last_pivot_index = local_high_idx
                last_detect_index = i

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

                if name == "BASE_SMA" and  last_pivot_index > 100 and n - i > 1500:
                    wm.add_session(start_index=local_min_idx, start_time_index = i,  prev_index=last_pivot_index, prev_time_index = last_detect_index , up_trend="True")
                #     print(local_min_idx)
                last_pivot_index = local_min_idx
                last_detect_index = i
                # if POINT_THRESHOLD == 0.003:
                #     print(time_df[local_min_idx],time_df[i],pivot_value)
    # print(len(trend_arr),len(np_arr),len(sma))
    # 
    if pivot_data:
        if name == "BASE_SMA":
            return wm, np.array(pivot_data, dtype=np.float64)
        else:
            return np.array(pivot_data, dtype=np.float64)
    else:
        return np.empty((0, 4), dtype=np.float64)

def merge_arr(base_arr, sml_arr, additional_sma_df):

    base_np = base_arr["sma_arr"]
    sml_sma_arr = sml_arr["sma_arr"]
    base_pivot_arr = base_arr["pivot_arr"]
    sml_pivot_arr = sml_arr["pivot_arr"]

    # BASEデータとSML_SMAの結合
    np_arr_with_base_sml_sma = np.column_stack((base_np, sml_sma_arr.reshape(-1, 1)))
    columns = ["time", "open", "high", "low", "close", "tick_volume", "spread", "BASE_SMA", "SML_SMA"]
    df = pd.DataFrame(np_arr_with_base_sml_sma, columns=columns)
    df["time"] = pd.to_datetime(df["time"], unit="ns", utc=True)
    
    # pivotのDataFrame化と時刻調整
    pivot_columns = ["detection_time", "pivot_time", "pivot_value", "pivot_type"]
    df_pivot = pd.DataFrame(base_pivot_arr, columns=pivot_columns)
    df_pivot["detection_time"] = pd.to_datetime(df_pivot["detection_time"], unit="ns", utc=True)
    df_pivot["pivot_time"] = pd.to_datetime(df_pivot["pivot_time"], unit="ns", utc=True)
    
    # sml_pivotのDataFrame化と時刻調整
    sml_pivot_columns = ["sml_detection_time", "sml_pivot_time", "sml_pivot_value", "sml_pivot_type"]
    sml_df_pivot = pd.DataFrame(sml_pivot_arr, columns=sml_pivot_columns)
    sml_df_pivot["sml_detection_time"] = pd.to_datetime(sml_df_pivot["sml_detection_time"], unit="ns", utc=True)
    sml_df_pivot["sml_pivot_time"] = pd.to_datetime(sml_df_pivot["sml_pivot_time"], unit="ns", utc=True)

    # ソート
    df_sorted = df.sort_values("time")
    df_pivot_sorted = df_pivot.sort_values("detection_time")
    sml_df_pivot_sorted = sml_df_pivot.sort_values("sml_detection_time")

    # pivotデータのマージ
    merged_temp = pd.merge_asof(df_sorted, df_pivot_sorted,
                                left_on="time", right_on="detection_time",
                                direction="nearest", tolerance=pd.Timedelta("2sec"))
    
    # sml_pivotデータのマージ
    merged_temp2 = pd.merge_asof(merged_temp, sml_df_pivot_sorted,
                                 left_on="time", right_on="sml_detection_time",
                                 direction="nearest", tolerance=pd.Timedelta("2sec"))
    
    # 追加したいSMAデータのマージ（最後尾に追加）
    # final_merged = pd.merge_asof(merged_temp2, additional_sma_df_sorted,
    #                              on="time", direction="nearest", tolerance=pd.Timedelta("2sec"))
    merged_temp2.reset_index(drop=True, inplace=True)
    additional_sma_df.reset_index(drop=True, inplace=True)
    final_merged = pd.concat([merged_temp2, additional_sma_df], axis=1)

    # 不要な時刻列の削除
    final_merged = final_merged.drop(columns=["detection_time", "sml_detection_time"])

    print("最終出力データのマージが完了しました。")
    print(final_merged.head())

    return final_merged

def merge_all_results(final_df, merged_tf_df):
    # merged_tf_df には各時間足の追加SMA列が含まれているので、重複する基本列を除外して結合
    base_columns = ["time", "open", "high", "low", "close", "tick_volume", "spread", "BASE_SMA", "SML_SMA"]
    extra_cols = merged_tf_df.drop(columns=base_columns)
    combined_df = pd.merge(final_df, extra_cols, on="time", how="left")
    return combined_df

# ---------------------------
# pre_data_process: 1分足 np_arr に対して BASE_SMA / SML_SMA の計算とピボット検出
# ---------------------------
def pre_data_process(np_arr, conditions, name, time_df):
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
    
    sma_arr = calculate_sma(np_arr[:, 4], window=window)
    trend_array = determine_trend(sma_arr)
    np_arr = np.column_stack((np_arr, sma_arr.reshape(-1, 1), trend_array.reshape(-1, 1)))
    
    if name == "BASE_SMA":
        wm, pivot_arr = detect_pivots(np_arr, time_df, name,
                                       POINT_THRESHOLD=point_threshold,
                                       LOOKBACK_BARS=lookback_bars,
                                       consecutive_bars=consecutive_bars,
                                       arrow_spacing=arrow_spacing)
        np_arr = np_arr[:, :-1]
        return (name, np_arr, pivot_arr, wm)
    elif name == "SML_SMA":
        pivot_arr = detect_pivots(np_arr, time_df, name,
                                  POINT_THRESHOLD=point_threshold,
                                  LOOKBACK_BARS=lookback_bars,
                                  consecutive_bars=consecutive_bars,
                                  arrow_spacing=arrow_spacing)
        np_arr = np_arr[:, :-1]
        return (name, sma_arr, pivot_arr)
# グローバルに8つの WaveManager インスタンスを作成


# def assign_session_to_manager(session):
#     # ラウンドロビン方式などでWaveManagerを選択してセッションを追加する
#     global session_counter
#     manager = wave_managers[session_counter % len(wave_managers)]
#     manager.add_session(session.pivot_data, session.up_trend, session.tp_level, session.stop_loss, session.check_no_SMA)
#     session_counter += 1

def process_data(conditions):
    global tp_level_global, check_no_SMA_global, range_param_global, stop_loss_global, time_df
    print("Current working directory:", os.getcwd())
    print(f"テスト開始時間 {datetime.now()}")
    
    symbol = conditions.get("symbol", "USDJPY")
    fromdate = conditions.get("fromdate", datetime(2024, 2, 17, 20, 0, tzinfo=pytz.UTC))
    todate = conditions.get("todate", datetime(2025, 2, 23, 6, 50, tzinfo=pytz.UTC))
    
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
    range_param_global = conditions.get("range", 80)
    stop_loss_global = conditions.get("stop", "sml")
    
    
    
    print("実行中")
    base_path = os.path.join("pickle_data", symbol, "USDJPY_1M.pkl")
    
    origin_df = pd.read_pickle(base_path)
    origin_df = origin_df.loc[:, ~origin_df.columns.str.contains('^Unnamed')]
    print(len(origin_df))
    origin_df["time"] = pd.to_datetime(origin_df["time"], utc=True)
    origin_df = origin_df.set_index("time")
    origin_df = origin_df.loc[fromdate:todate].reset_index()

    # 基本データ部分 (最初の7列) とSMA関連の列 (7列目以降) に分割
    base_df = origin_df.iloc[:, :7]
    print(len(base_df))
    sma_df = origin_df.iloc[:, 7:]
    print(len(sma_df))

    print("base_df:")
    print(base_df.head())
    print("sma_df:")
    print(sma_df.head())
    print(sma_df.columns)
    time_df = base_df["time"]
    np_arr = base_df.to_numpy(dtype=np.float64)
    
    base_result = pre_data_process(np_arr, conditions, "BASE_SMA", time_df)
    sml_result = pre_data_process(np_arr, conditions, "SML_SMA", time_df)
    results = [base_result, sml_result]
    
    result_dict = {}
    for result in results:
        if len(result) == 4:
            name, arr, pivot_arr, wm = result
            result_dict[name] = {"sma_arr": arr, "pivot_arr": pivot_arr, "wm": wm}
        else:
            name, arr, pivot_arr = result
            result_dict[name] = {"sma_arr": arr, "pivot_arr": pivot_arr}
    base_arr = result_dict.get("BASE_SMA")
    sml_arr = result_dict.get("SML_SMA")
    wm = base_arr.get("wm")
    
    # 各時間足のSMA・トレンド情報を1分足のbase_dfにマージ
    final_df = merge_arr(base_arr, sml_arr, sma_df)
    print(final_df)
    base_df.to_csv("main_data.csv", index=False)
    wm.full_data = final_df.to_numpy(dtype=np.float64)
    wm.analyze_sessions()
    
    print("処理終了")
    print(f"終了時間 {datetime.now()}")


if __name__ == "__main__":
    conditions = {
        "symbol": "USDJPY",
        "fromdate": datetime(2024, 11, 17, 0, 0, tzinfo=pytz.UTC), #始まる日時
        "todate": datetime(2025, 2, 21, 18, 0, tzinfo=pytz.UTC), #終わる日時
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
    start = time.time()
    process_data(conditions)
    
    end = time.time()  # 現在時刻（処理完了後）を取得

    time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
    print(time_diff)  # 処理にかかった時間データを使用



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


# 0: time
# 1: open
# 2: high
# 3: low
# 4: close
# 5: tick_volume
# 6: spread
# 7: BASE_SMA
# 8: SML_SMA
# 9: pivot_time
# 10: pivot_value
# 11: pivot_type
# 12: sml_pivot_time
# 13: sml_pivot_value
# 14: sml_pivot_type