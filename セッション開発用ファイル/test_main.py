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
# （必要に応じてMachineのimportなど整理してください）

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
        assert self.start_index == 231, f"start_index が不一致。期待値: 251, 実際: {self.start_index}"
        assert self.prev_index == 216, f"prev_index が不一致。期待値: 219, 実際: {self.prev_index}"
        assert self.full_data[self.start_index, 3] == 151.683, f"start_index の価格が不一致。実際: {self.full_data[self.start_index,3]}"
        while not self.destroy_reqest:
            action = self.state_actions.get(self.state)
            if action:
                action()
        # if self.trade_log:
        # print(float(self.trade_log[6]))

    def handle_created_base_arrow(self):
        total_len = len(self.full_data)
        base_pivots_index = self.get_pivots_in_range(self.start_time_index + 1, total_len - 1, base_or_sml="base")
        indices_after = base_pivots_index[base_pivots_index > self.start_time_index]

        if indices_after.size >= 2:
            self.new_arrow_index = self.find_detection_index(self.full_data[indices_after[0], 9], 0)
            self.new_arrow_detection_index = self.find_detection_index(self.full_data[self.new_arrow_index, 0], 9)
            self.next_new_arrow_index = self.find_detection_index(self.full_data[indices_after[1], 9], 0)

            assert self.full_data[self.new_arrow_index, 2] == 151.854, f"new_arrowが正しくない。実際は{self.full_data[self.new_arrow_index,2]}"
            assert self.new_arrow_detection_index == 302, f"new_arrow_detection_indexが正しくない。実際は{self.new_arrow_detection_index}"
            assert self.full_data[self.next_new_arrow_index, 3] == 151.782, f"next_new_arrowが正しくない。実際は{self.full_data[self.next_new_arrow_index,3]}"
        else:
            self.destroy_reqest = True
            return

        if self.up_trend:
            prices = (self.full_data[self.prev_index, 2], self.full_data[self.start_index, 3])
            _, _, self.fibo_minus_20, self.fibo_minus_200 = detect_extension_reversal(prices, None, None, 0.2, 2)
            judged_price = self.full_data[self.new_arrow_index, 2]
            assert judged_price == 151.854, f"ジャッジプライス(new_arrow)が正しくない。実際は{judged_price}"
        else:
            prev_price, start_price = self.full_data[self.prev_index, 3], self.full_data[self.start_index, 2]
            self.fibo_minus_20, self.fibo_minus_200, _, _ = detect_extension_reversal(prev_price, start_price, -0.2, -2, None, None)
            judged_price = self.full_data[self.new_arrow_index, 3]

        if watch_price_in_range(self.fibo_minus_20, self.fibo_minus_200, judged_price):
            
            self.create_new_arrow()
            
        else:
            self.destroy_reqest = True

    def handle_created_new_arrow(self):
        
        # on_enterでtouch_37を発動できなかった場合の処理
        required_data = self.full_data[self.new_arrow_detection_index+1:]
        while self.state == "created_new_arrow" and not self.destroy_reqest:
            for i in range(len(required_data)):
                pre_check = (not check_touch_line(self.base_fibo37, required_data[i, 3])
                             if self.up_trend else
                             check_touch_line(self.base_fibo37, required_data[i, 2]))
                if pre_check:
                    if self.price_in_range_while_adjustment(self.new_arrow_detection_index,
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
        """
        infibos状態のメインループ：
        new_arrow_detection_index + 1 から最大200本分の足（required_data）を検証し、
        potential_entry や determined_neck に基づいてエントリー判定を行う。
        """
        start_of_infibos = self.new_arrow_detection_index + 1
        end_of_infibos = self.new_arrow_detection_index + 200
        required_data = self.full_data[start_of_infibos : end_of_infibos]

        while self.state == "infibos":
            for local_index in range(len(required_data)):
                arr = required_data[local_index]
                # グローバルインデックス = new_arrow_detection_index+1 + local_index
                global_index = start_of_infibos + local_index

                if self.potential_neck:
                    entry_result = self.potential_entry(required_data, local_index)
                    if entry_result is True:
                        # import pdb; pdb.set_trace()
                        self.build_position()
                        break
                    elif entry_result is False:
                        self.potential_neck.clear()

                if self.determined_neck and "has_position" not in self.state:
                    self.highlow_since_new_arrow = self.get_high_and_low_in_term(self.index_of_fibo37, global_index + 1)
                    for neckline in self.determined_neck[:]:
                        if self.state != 'infibos_has_determined_neck':
                            break
                        if self.up_trend:
                            neck_price = self.full_data[neckline, 2]
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
                                self.determined_neck.remove(neckline)
                        else:
                            neck_price = self.full_data[neckline, 3]
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
                                self.determined_neck.remove(neckline)

                if not self.price_in_range_while_adjustment(global_index):
                    self.destroy_reqest = True
                    break

                if 100 < arr[13] < 165:
                    self.append_sml_pivot_data(required_data, local_index)
                            
                if not self.price_in_range_while_adjustment(global_index):
                    self.destroy_reqest = True
                    break

                if 100 < arr[13] < 165:
                    self.append_sml_pivot_data(required_data, i)

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
            print("プライス達", prices)
            self.base_fibo70, _, self.base_fibo37, _ = detect_extension_reversal(prices, lower1_percent=0.3, higher1_percent=-0.37)
        else:
            prices = (self.full_data[self.start_index, 2], self.full_data[self.new_arrow_index, 3])
            self.base_fibo37, _, self.base_fibo70, _ = detect_extension_reversal(prices, lower1_percent=0.37, higher1_percent=-0.3)

        self.get_golden_cross_index()
        assert self.index_of_goldencross == 289, f"ゴールデンクロスインデックスが違う。実際は{self.index_of_goldencross}"
        # sml_pivots_after_goldencross を[original_index, detection_index, price, type]で保持
        sml_indices = self.get_pivots_in_range(self.index_of_goldencross, self.new_arrow_detection_index, base_or_sml="sml")
        self.sml_pivots_after_goldencross = np.array([
            [index, #検出したindex
             self.find_detection_index(self.full_data[index, 12], 0),#実際のピボットがあるindex
             self.full_data[index, 13], #ピボット価格
             self.full_data[index, 14]] #ピボットタイプ
            for index in sml_indices
        ])
        
        # ※assertのチェックは、必要に応じてshape[0]で確認する
        assert self.sml_pivots_after_goldencross.shape[0] == 2, f"スモールpivotの数が違う。実際は{self.sml_pivots_after_goldencross.shape[0]}"
        highest, lowest = self.get_high_and_low_in_term(self.index_of_goldencross, self.new_arrow_detection_index, False)
        if self.up_trend:
            if not watch_price_in_range(self.base_fibo37, self.base_fibo70, lowest):
                print(self.start_time_index)
                self.index_of_fibo37 = self.get_touch37_index()
                assert self.index_of_fibo37 == 292, f"index_of_goldencrossが違う。実際は{self.index_of_goldencross}"
                self.touch_37()
                return
        else:
            if watch_price_in_range(self.base_fibo37, self.base_fibo70, highest):
                self.index_of_fibo37 = self.get_touch37_index()
                self.touch_37()
                return

    def on_enter_infibos(self):
        print("ここだよ")
        self.get_potential_neck_wheninto_newarrow()

    def on_enter_has_position(self):
        print("ここだよ")
        assert self.final_neckline_index == 299, f"ネックラインが違う。実際は{self.final_neckline_index}"
        self.close()
        
    def on_enter_closed(self):
        print("ここだよ")
        self.trade_log = np.array([self.entry_index,
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
        if end_index:
            end_index = int(end_index)
            required_data = self.full_data[start_index:end_index]
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
        print(base_sma_since_new_arrow, sml_sma_since_new_arrow)
        if self.up_trend:
            conditions = np.where(base_sma_since_new_arrow > sml_sma_since_new_arrow)
        else:
            conditions = np.where(base_sma_since_new_arrow < sml_sma_since_new_arrow)
        if conditions[0].size > 0:
            print(conditions[0][0] + self.new_arrow_index)
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
        print(valid_indices)
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
                            prices = (sml_pvts[i-1,3], sml_pvts[i,2])
                            fibo32_of_ptl_neck = detect_extension_reversal(prices, -0.32, 0.32, None, None)
                            if watch_price_in_range(fibo32_of_ptl_neck[0], fibo32_of_ptl_neck[1], sml_pvts[i+1, 3]):
                                determined_neck = sml_pvts[i]
                        else:
                            potential_neck = sml_pvts[i]
                else:
                    if self.full_data[sml_pvts[i, 0], 14] == 0 and sml_pvts[i, 0] > self.index_of_fibo37:
                        if i + 1 < sml_pvts.shape[0]:
                            prices = (sml_pvts[i-1,2], sml_pvts[i,3])
                            fibo32_of_ptl_neck = detect_extension_reversal(prices, -0.32, 0.32, None, None)
                            if watch_price_in_range(fibo32_of_ptl_neck[0], fibo32_of_ptl_neck[1], sml_pvts[i+1, 2]):
                                determined_neck = sml_pvts[i]
                        else:
                            potential_neck = sml_pvts[i]
                if determined_neck:
                    self.determined_neck.append(determined_neck)
                    self.organize_determined_neck()
            if potential_neck is not None:
                self.potential_neck.append(potential_neck)
        assert len(self.potential_neck) == 1, f"ポテンシャルネックの数が違う。実際は{len(self.potential_neck)}"
        assert self.potential_neck[0][1] == 299, f"ポテンシャルネックのインデックスが間違い。実際は{self.potential_neck[0][1]}"

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

            judged_price = sml_pvts[-1][2]
            if self.up_trend:
                fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level, -0.32, 0.32, None, None)
                if watch_price_in_range(fibo32_of_ptl_neck[0], fibo32_of_ptl_neck[1], judged_price):
                    self.determined_neck.append(self.potential_neck[-1])
                    self.potential_neck.clear()
                    self.organize_determined_neck()
            else:
                fibo32_of_ptl_neck = detect_extension_reversal(pvts_to_get_32level, None, None, -0.32, 0.32)
                if watch_price_in_range(fibo32_of_ptl_neck[2], fibo32_of_ptl_neck[3], judged_price):
                    self.determined_neck.append(self.potential_neck[-1])
                    self.potential_neck.clear()
                    self.organize_determined_neck()


    # necklineは数字で入れる
    def potential_entry(self, required_data, local_index):
        """
        required_data:
        handle_infibos で切り出した配列（例: self.full_data[self.new_arrow_detection_index+1 : self.new_arrow_detection_index+200]）
        local_index:
        required_data 内での行番号 (0,1,2,...)
        
        この関数内でグローバルインデックスを計算し、そこでエントリー成立などの処理を行う。
        """
        global_index = self.new_arrow_detection_index + 1 + local_index
        
        arr = required_data[local_index]
        
        sml_pvts = self.sml_pivots_after_goldencross
        if self.potential_neck:
            if sml_pvts.shape[0] < 2:
                self.potential_neck.clear()
                return None
            
        neck_price = self.potential_neck[-1][2]

        if self.up_trend:
            print("いまだあ (global, local) =", global_index, local_index)
            if arr[2] > neck_price and neck_price >= arr[7]:
                sml_index_to_get32 = (neck_price, sml_pvts[-2][2])
                fibo32_of_ptl_neck = detect_extension_reversal(sml_index_to_get32, -0.32, 0.32, None, None)
                last_highlow = self.get_high_and_low_in_term(self.potential_neck[-1][0], global_index)
                if watch_price_in_range(fibo32_of_ptl_neck[0], fibo32_of_ptl_neck[1], last_highlow[1]):
                    self.highlow_since_new_arrow = self.get_high_and_low_in_term(self.index_of_fibo37, global_index)
                    self.highlow_stop_loss = self.highlow_since_new_arrow[1] - 0.006
                    self.sml_stop_loss = last_highlow[1] - 0.006
                    self.final_neckline_index = self.potential_neck[-1][1]
                    prices_data_to_get_take_profit = (self.full_data[self.start_index,3],
                                                    self.full_data[self.new_arrow_index,2])
                    highlow = detect_extension_reversal(prices_data_to_get_take_profit, higher1_percent=0.32)
                    print(prices_data_to_get_take_profit)
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
                fibo32_of_ptl_neck = detect_extension_reversal(sml_index_to_get32, None, None, -0.32, 0.32)
                last_highlow = self.get_high_and_low_in_term(self.potential_neck[-1][0], global_index)
                if watch_price_in_range(fibo32_of_ptl_neck[0], fibo32_of_ptl_neck[1], last_highlow[0]):
                    self.highlow_since_new_arrow = self.get_high_and_low_in_term(self.index_of_fibo37, global_index)
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
        ピボットレコードは [original_index, detection_index, detection_time, pivot_time, price, type] の6要素。
        """
        state_list = ["infibos", "infibos_has_determined_neck"]

        actual_index = self.new_arrow_detection_index + new_sml_pivot_index #検出したrow
        row = self.full_data[actual_index]  # キャッシュしてアクセス回数を削減
        pivot_index = self.find_detection_index(row[12], 0) #実際のピボットがあるindex
        price = row[13] #ピボット価格
        type_val = row[14] #ピボットタイプ

        new_row = np.array([actual_index, pivot_index, price, type_val])
        if self.sml_pivots_after_goldencross.size == 0:
            self.sml_pivots_after_goldencross = new_row.reshape(1, -1)
        else:
            self.sml_pivots_after_goldencross = np.vstack((self.sml_pivots_after_goldencross, new_row))

        if self.state in state_list:
            if self.potential_neck:
                self.check_potential_to_determine_neck()
            if not self.potential_neck:
                if self.up_trend and type_val == 1:
                    self.potential_neck.append(new_row)
                elif (not self.up_trend) and type_val == 0:
                    self.potential_neck.append(new_row)

    def organize_determined_neck(self):
        result = []
        for item in self.determined_neck:
            while result and item[2] > result[-1][2]:
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
            high, low = self.full_data[self.new_arrow_index, 2], self.base_fibo70
        else:
            high, low = self.base_fibo70, self.full_data[self.new_arrow_index, 3]
        judged_price = self.get_high_and_low_in_term(start_index, end_index, close)
        if high > judged_price[0] or low < judged_price[1]:
            return True
        else:
            self.destroy_reqest = True

def watch_price_in_range(low, high, judged_price):
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
