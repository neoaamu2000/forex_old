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

# グローバル変数
current_price_global = []
symbol = "USDJPY"  # デフォルト値
last_pivot_data = 999
sml_last_pivot_data = 999

# 状態定義
states = [
    "created_base_arrow",
    "created_new_arrow",
    "infibos",
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
        self.start_pivot_time = None
        self.original_offset = None
        self.full_data = None
        self.start_index = start_index  # 始まったピボットのインデックス
        self.start_time_index = start_time_index
        self.prev_index = prev_index
        self.prev_time_index = prev_time_index
        self.machine = Machine(model=self, states=states, transitions=transitions, initial="created_base_arrow")
        self.new_arrow_index = None         # 推進波の終わりの最高（安）値
        self.next_new_arrow_index = None
        self.new_arrow_detection_index = None
        self.next_next_new_arrow_index = None
        
        self.fibo_minus_20 = None
        self.fibo_minus_200 = None
        
        self.base_fibo37 = None  # 37%リトレースメントライン
        self.base_fibo70 = None  # 70%リトレースメントライン
        self.index_of_fibo37 = None
        self.start_of_simulation = None
        self.final_neckline_index = None
        self.time_of_goldencross = None
        self.highlow_since_new_arrow = []  # 調整波の戻しの深さ
        self.sml_pivot_data = []           # touch20以降のsml_pivot記録

        # ネックライン関連（各ピボットレコードを [original_index, detection_index, price, type] として保持）
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

    def execute_state_action(self):
        """現在の状態に対応する処理関数を実行"""
        while not self.destroy_reqest:
            action = self.state_actions.get(self.state)
            if action:
                action()

    def handle_created_base_arrow(self):
        total_len = len(self.full_data)
        base_pivots_index = self.get_pivots_in_range(self.start_time_index + 1, total_len - 1, base_or_sml="base")
        indices_after = base_pivots_index[base_pivots_index > self.start_time_index]

        if indices_after.size >= 3:
            self.new_arrow_index = self.find_detection_index(self.full_data[indices_after[0], 9], 0)
            self.new_arrow_detection_index = self.find_detection_index(self.full_data[self.new_arrow_index, 0], 9)
            self.next_new_arrow_index = self.find_detection_index(self.full_data[indices_after[1], 9], 0)
            self.next_next_new_arrow_index = self.find_detection_index(self.full_data[indices_after[2], 9], 0)
        else:
            self.destroy_reqest = True
            return

        if self.up_trend:
            prices = (self.full_data[self.prev_index, 2], self.full_data[self.start_index, 3])
            _, _, self.fibo_minus_20, self.fibo_minus_200 = detect_extension_reversal(prices, None, None, 0.2, 2)
            judged_price = self.full_data[self.new_arrow_index, 2]
        else:
            prices = (self.full_data[self.prev_index, 3], self.full_data[self.start_index, 2])
            self.fibo_minus_20, self.fibo_minus_200, _, _ = detect_extension_reversal(prices, -0.2, -2, None, None)
            judged_price = self.full_data[self.new_arrow_index, 3]

        if self.watch_price_in_range(self.fibo_minus_20, self.fibo_minus_200, judged_price):
            self.create_new_arrow()
        else:
            self.destroy_reqest = True

    def handle_created_new_arrow(self):
        required_data = self.full_data[self.new_arrow_detection_index+1:]
        while self.state == "created_new_arrow" and not self.destroy_reqest:
            for i in range(len(required_data)):
                if self.new_arrow_detection_index+1+1 > self.next_next_new_arrow_index:
                    self.destroy_reqest = True
                    break
                arr = required_data[i]
                if 100 < arr[13] < 165:
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
        new_arrow_detection_index 以降から最大200本分の足（required_data）を検証し、
        potential_entry や determined_neck に基づいてエントリー判定を行う。
        エントリー時、リスクリワード比率が2以下の場合はエントリーせず候補を削除する。
        """
        end_of_infibos = self.start_of_simulation + 200
        required_data = self.full_data[self.start_of_simulation : end_of_infibos]
        while self.state == "infibos" and not self.destroy_reqest:
            for local_index in range(len(required_data)):
                arr = required_data[local_index]
                global_index = self.start_of_simulation + local_index
                if global_index > self.next_next_new_arrow_index:
                    self.destroy_reqest = True
                if self.state != "infibos" or self.destroy_reqest:
                    break

                # potential_neck をもとにエントリー判定
                if self.potential_neck:
                    entry_result = self.potential_entry(required_data, local_index)
                    if entry_result is True:
                        self.build_position()
                        break
                    elif entry_result is False:
                        # 候補削除前に、リストが空でないか確認してから pop で削除
                        if self.potential_neck:
                            self.potential_neck.pop()
                
                # determined_neck に対してエントリー条件のチェックを実施
                if self.determined_neck and "has_position" not in self.state:
                    self.highlow_since_new_arrow = self.get_high_and_low_in_term(self.start_of_simulation, global_index + 1)
                    # 現在の determined_neck のコピーでループ
                    for neckline in self.determined_neck[:]:
                        if self.state != "infibos":
                            break
                        if self.up_trend:
                            neck_price = neckline[2]
                            if arr[2] > neck_price and self.check_no_SMA(global_index, neck_price):
                                self.highlow_stop_loss = self.highlow_since_new_arrow[1] - 0.006
                                self.sml_stop_loss = self.sml_pivots_after_goldencross[-1][2] - 0.006
                                self.final_neckline_index = neckline[1]
                                prices_data_to_get_take_profit = (
                                    #self.full_data[self.start_index, 3],
                                    self.full_data[self.new_arrow_index, 2],
                                    self.highlow_since_new_arrow[1]
                                )
                                highlow = detect_extension_reversal(prices_data_to_get_take_profit, higher1_percent=0.9)
                                self.take_profit = highlow[2]
                                self.entry_line = neck_price + 0.006
                                self.entry_index = global_index
                                self.point_to_stoploss = abs(self.entry_line - self.highlow_stop_loss)
                                self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                                # リスクリワード比率が2以下なら候補を削除（np.array_equal を利用）
                                if (self.point_to_take_profit / self.point_to_stoploss) <= 1.2:
                                    self.determined_neck = [
                                        n for n in self.determined_neck
                                        if not np.array_equal(n, neckline)
                                    ]
                                else:
                                    self.build_position()
                                    break
                            elif arr[2] > neck_price and not self.check_no_SMA(global_index, neck_price):
                                self.determined_neck = [
                                    n for n in self.determined_neck
                                    if not np.array_equal(n, neckline)
                                ]
                        else:
                            neck_price = neckline[2]
                            if arr[3] < neck_price and self.check_no_SMA(global_index, neck_price):
                                self.highlow_stop_loss = self.highlow_since_new_arrow[0] + 0.006
                                self.sml_stop_loss = self.sml_pivots_after_goldencross[-1][2] + 0.006
                                self.final_neckline_index = neckline[1]
                                prices_data_to_get_take_profit = (
                                    #self.full_data[self.start_index, 2],
                                    self.full_data[self.new_arrow_index,3],
                                    self.highlow_since_new_arrow[0]
                                )
                                highlow = detect_extension_reversal(prices_data_to_get_take_profit, lower1_percent=-0.9)
                                self.take_profit = highlow[0]
                                self.entry_line = neck_price - 0.006
                                self.entry_index = global_index
                                self.point_to_stoploss = abs(self.entry_line - self.highlow_stop_loss)
                                self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                                if (self.point_to_take_profit / self.point_to_stoploss) <= 1.2:
                                    self.determined_neck = [
                                        n for n in self.determined_neck
                                        if not np.array_equal(n, neckline)
                                    ]
                                else:
                                    self.build_position()
                                    break
                            elif arr[3] < neck_price and not self.check_no_SMA(global_index, neck_price):
                                self.determined_neck = [
                                    n for n in self.determined_neck
                                    if not np.array_equal(n, neckline)
                                ]
                if not self.price_in_range_while_adjustment(global_index):
                    self.destroy_reqest = True
                    break
                if 100 < arr[13] < 165:
                    self.append_sml_pivot_data(required_data, local_index)
                if len(required_data) - local_index == 1:
                    self.destroy_reqest = True

    def handle_has_position(self):
        # エントリー時刻をセットしてからトレード結果を算出
        self.entry_time = self.full_data[self.entry_index, 0]
        self.global_entry_index = self.original_offset + self.entry_index
        self.close()

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
        self.entry_time = self.full_data[self.entry_index, 0]
        self.global_entry_index = self.original_offset + self.entry_index
        self.close()

    def on_enter_closed(self):
        entry_slice = self.full_data[self.entry_index:]
        self.trade_log = self.calculate_trade_result(self,entry_slice)
        # self.trade_log = {
        #     "entry_time": self.entry_time,
        #     "up_trend": self.up_trend,
        #     "start_pivot_time": self.start_pivot_time,
        #     "global_entry_index": self.global_entry_index,
        #     "entry_line": self.entry_line,
        #     "take_profit": self.take_profit,
        #     "highlow_stop_loss": self.highlow_stop_loss,
        #     "sml_stop_loss": self.sml_stop_loss,
        #     "point_to_stoploss": self.point_to_stoploss,
        #     "point_to_take_profit": self.point_to_take_profit,
        #     "name": self.name
        # }
        self.destroy_reqest = True

    def __repr__(self):
        return f"MyModel(name={self.name}, state={self.state})"

    def find_detection_index(self, target_time, col):
        if isinstance(target_time, np.datetime64):
            target_time = target_time.astype("int64")
        indices = np.where(self.full_data[:, col] == target_time)[0]
        return indices[0] if indices.size > 0 else None

    def get_high_and_low_in_term(self, start_index, end_index=None, close=None):
        start_index = int(start_index)
        if end_index is not None:
            end_index = int(end_index)
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

    def check_no_SMA(self, index, neck_price):
        sma_index = (15, 17, 19, 21, 23, 25, 27) 
        sma_value = [self.full_data[index, smas] for smas in sma_index]
        if self.up_trend:
            return neck_price > max(sma_value) - 0.005
        else:
            return self.full_data[index, 3] < min(sma_value) + 0.005

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
                        if i + 1 < sml_pvts.shape[0]:
                            prices = (sml_pvts[i-1, 2], sml_pvts[i, 2])
                            fibo32_of_ptl_neck = detect_extension_reversal(prices, -0.32, 0.32, None, None)
                            if self.watch_price_in_range(fibo32_of_ptl_neck[0], fibo32_of_ptl_neck[1], sml_pvts[i+1, 3]):
                                determined_neck = sml_pvts[i]
                        else:
                            potential_neck = sml_pvts[i]
                else:
                    if sml_pvts[i, 3] == 0 and sml_pvts[i, 2] > self.index_of_fibo37:
                        if i + 1 < sml_pvts.shape[0]:
                            prices = (sml_pvts[i-1, 2], sml_pvts[i, 2])
                            fibo32_of_ptl_neck = detect_extension_reversal(prices, None, None, 0.32, -0.32)
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
            if sml_pvts.shape[0] < 3:
                self.destroy_reqest = True
                return None
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

    def potential_entry(self, required_data, local_index):
        """
        potential_entry のロジック（ポテンシャルエントリー候補からエントリー判定）
        
        ・潜在的なネックライン候補（potential_neck）があるかチェック
        ・エントリー時の条件（価格、SMAチェック、リトレースメントラインの範囲内）を確認
        ・エントリー前にリスクリワード比（point_to_take_profit / point_to_stoploss）が2以上であるかチェックする
        ※ 2未満なら候補を削除してエントリーしない
        """
        # potential_neckが空ならすぐにNoneを返す
        if not self.potential_neck:
            return None

        global_index = self.start_of_simulation + local_index
        arr = required_data[local_index]
        sml_pvts = self.sml_pivots_after_goldencross

        # sml_pvtsのサイズが2未満なら、potential_neckをクリアして終了
        if sml_pvts.shape[0] < 2:
            self.potential_neck.clear()
            return None

        # 再度、potential_neckが空になっていないかチェック
        if not self.potential_neck:
            return None

        # potential_neckの最後の要素のネック価格を取得
        neck_price = self.potential_neck[-1][2]

        if self.up_trend:
            # 上昇トレンドの場合の処理
            if arr[2] > neck_price and self.check_no_SMA(global_index, neck_price):
                # sml_pvtsのサイズが2未満でないか再チェック
                if sml_pvts.shape[0] < 2:
                    return None
                sml_index_to_get32 = (neck_price, sml_pvts[-2][2])
                fibo32_of_ptl_neck = detect_extension_reversal(sml_index_to_get32, -0.382, 0.382, None, None)
                last_highlow = self.get_high_and_low_in_term(self.potential_neck[-1][0], global_index)
                if self.watch_price_in_range(fibo32_of_ptl_neck[0], fibo32_of_ptl_neck[1], last_highlow[1]):
                    # エントリーラインおよびストップロス／テイクプロフィットの計算
                    self.highlow_since_new_arrow = self.get_high_and_low_in_term(self.index_of_fibo37, global_index + 1)
                    self.highlow_stop_loss = self.highlow_since_new_arrow[1] - 0.006
                    self.sml_stop_loss = last_highlow[1] - 0.006
                    self.final_neckline_index = self.potential_neck[-1][1]
                    prices_data_to_get_take_profit = (
                        # self.full_data[self.start_index, 3],
                        self.full_data[self.new_arrow_index, 2],
                        self.highlow_since_new_arrow[1]
                    )
                    highlow = detect_extension_reversal(prices_data_to_get_take_profit, higher1_percent=0.9)
                    self.take_profit = highlow[2]
                    self.entry_line = neck_price + 0.006
                    self.entry_index = global_index
                    self.point_to_stoploss = abs(self.entry_line - self.highlow_stop_loss)
                    self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                    # リスクリワード比のチェック：2未満なら候補を削除してFalseを返す
                    if (self.point_to_take_profit / self.point_to_stoploss) < 1.2:
                        # リスクリワード比が条件を満たさないので候補削除
                        self.potential_neck.pop()
                        return False
                    else:
                        return True
            elif arr[2] > neck_price and not self.check_no_SMA(global_index, neck_price):
                return False
        else:
            # 下降トレンドの場合の処理
            if arr[3] < neck_price and self.check_no_SMA(global_index, neck_price):
                if sml_pvts.shape[0] < 2:
                    return None
                sml_index_to_get32 = (neck_price, sml_pvts[-2][2])
                fibo32_of_ptl_neck = detect_extension_reversal(sml_index_to_get32, None, None, -0.382, 0.382)
                last_highlow = self.get_high_and_low_in_term(self.potential_neck[-1][0], global_index)
                if self.watch_price_in_range(fibo32_of_ptl_neck[2], fibo32_of_ptl_neck[3], last_highlow[0]):
                    self.highlow_since_new_arrow = self.get_high_and_low_in_term(self.index_of_fibo37, global_index + 1)
                    self.highlow_stop_loss = self.highlow_since_new_arrow[0] + 0.006
                    self.sml_stop_loss = last_highlow[0] + 0.006
                    self.final_neckline_index = self.potential_neck[-1][1]
                    prices_data_to_get_take_profit = (
                        #self.full_data[self.start_index, 2],
                        self.full_data[self.new_arrow_index, 3],
                        self.highlow_since_new_arrow[0]
                    )
                    highlow = detect_extension_reversal(prices_data_to_get_take_profit, lower1_percent=-0.9)
                    self.take_profit = highlow[0]
                    self.entry_line = neck_price - 0.006
                    self.entry_index = global_index
                    self.point_to_stoploss = abs(self.entry_line - self.highlow_stop_loss)
                    self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                    # リスクリワード比のチェック：2未満なら候補を削除してFalseを返す
                    if (self.point_to_take_profit / self.point_to_stoploss) < 1.2:
                        self.potential_neck.pop()
                        return False
                    else:
                        return True
            elif arr[3] < neck_price and not self.check_no_SMA(global_index, neck_price):
                return False
        return None



    def calculate_trade_result(self, session, data_slice):
        if data_slice.size == 0:
            # データが空の場合の特別処理
            if session.next_next_new_arrow_index and session.next_next_new_arrow_index > 0:
                exit_price = session.full_data[session.next_next_new_arrow_index - 1, 4]
                exit_time = session.full_data[session.next_next_new_arrow_index - 1, 0]
            else:
                exit_price = session.entry_line
                exit_time = session.entry_time
            exit_reason = "forced close"
            result = "forced"
        else:
            highs = data_slice[:, 2]
            lows = data_slice[:, 3]

            if session.up_trend:
                # 上昇トレンドの場合
                tp_hit = highs >= session.take_profit
                # highlow_stop_loss のみでチェック（下値がstoploss以下になったら損切り）
                sl_hit = lows <= session.highlow_stop_loss
            else:
                # 下降トレンドの場合
                tp_hit = lows <= session.take_profit
                # highlow_stop_loss のみでチェック（高値がstoploss以上になったら損切り）
                sl_hit = highs >= session.highlow_stop_loss

            tp_index = np.argmax(tp_hit) if np.any(tp_hit) else np.inf
            sl_index = np.argmax(sl_hit) if np.any(sl_hit) else np.inf

            # 最初に発生したイベントのインデックスを求める
            first_event_index = min(tp_index, sl_index)

            if first_event_index == np.inf:
                # どちらの条件も満たさなかった場合は最終バーで強制決済
                exit_index = len(data_slice) - 1
                exit_price = data_slice[exit_index, 4]
                exit_reason = "forced close"
                result = "forced"
            elif first_event_index == tp_index:
                # テイクプロフィットが先にヒットした場合
                exit_index = tp_index
                exit_price = session.take_profit
                exit_reason = "T/P"
                result = "win"
            else:
                # ストップロスが先にヒットした場合（ここでは highlow_stop_loss のみ）
                exit_index = sl_index
                exit_price = session.highlow_stop_loss
                exit_reason = "S/L (highlow)"
                result = "loss"

            exit_time = data_slice[exit_index, 0]

        profit_loss = (exit_price - session.entry_line if session.up_trend 
                    else session.entry_line - exit_price)

        trade_log = {
            "entry_time": pd.to_datetime(session.entry_time, unit="ns", utc=True),
            "exit_time": pd.to_datetime(exit_time, unit="ns", utc=True),
            "entry_price": session.entry_line,
            "exit_price": exit_price,
            "take_profit": session.take_profit,
            "highlow_stop_loss": session.highlow_stop_loss,
            # sml_stop_loss は今回使用しないため省略
            "exit_reason": exit_reason,
            "result": result,
            "profit_loss": profit_loss,
            "risk_reward_ratio": session.point_to_take_profit / session.point_to_stoploss
        }

        return trade_log




    def append_sml_pivot_data(self, required_data, new_sml_pivot_index):
        state_list = ["infibos"]
        if self.state == "created_new_arrow":
            actual_index = self.new_arrow_detection_index + new_sml_pivot_index
        else:
            actual_index = self.start_of_simulation + new_sml_pivot_index
        row = self.full_data[actual_index]
        pivot_index = self.find_detection_index(row[12], 0)
        price = row[13]
        type_val = row[14]
        new_row = np.array([actual_index, pivot_index, price, type_val])
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
        self.determined_neck = [item for item in self.determined_neck if item is not None]
        result = []
        if self.up_trend:
            for item in self.determined_neck:
                while result and item[2] > result[-1][2]:
                    result.pop()
                result.append(item)
        else:
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
            low = self.full_data[self.new_arrow_index, 3]
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
        self.risk_percentage = 5.0

    def analyze_sessions(self):
        for session_id in list(self.sessions.keys()):
            session = self.sessions[session_id]
            session.start_origin = session.start_index
            session.full_data = self.full_data[session.prev_index-100:session.start_index+1500, :]
            difference = session.start_time_index - session.start_index
            session.start_index = session.start_index - session.prev_index + 100
            session.start_pivot_time = session.full_data[session.start_index, 0]
            session.start_time_index = session.start_index + difference
            session.prev_index = 100
            session.execute_state_action()
            if session.trade_log is not None and len(session.trade_log) > 0:
                self.trade_logs.append(session.trade_log)
            del self.sessions[session_id]
        if self.trade_logs:
            trade_logs_df = pd.DataFrame(self.trade_logs)
            time_columns = ["entry_time", "start_pivot_time"]
            for col in time_columns:
                if col in trade_logs_df.columns:
                    trade_logs_df[col] = pd.to_datetime(trade_logs_df[col], unit="ns", utc=True, errors='coerce')
            trade_logs_df.sort_values(by="entry_time", inplace=True)
            trade_logs_df.to_csv("test_result/usdjpy_check_no_sma搭載1.csv", index=False)
            print("ログ出力完了", trade_logs_df.head())
            print(f"ログ数：{len(trade_logs_df)}")
            self.trade_logs = trade_logs_df
            self.summarize_and_export_results(filename=conditions.get("output_file", "final_trade_logs.csv"),
                                    initial_capital=10000,
                                    risk_percentage=self.risk_percentage)
    def organize_trade_logs(self):
        columns = ["entry_time", "up_trend", "start_pivot_time", "global_entry_index", 
                   "entry_line", "take_profit", "highlow_stop_loss",# "sml_stop_loss", 
                   "point_to_stoploss", "point_to_take_profit", "name"]
        trade_logs = pd.DataFrame(self.trade_logs, columns=columns)
        time_columns = ["entry_time", "start_pivot_time", "exit_time", "order_time"]
        for col in time_columns:
            if col in trade_logs.columns:
                trade_logs[col] = pd.to_datetime(trade_logs[col], unit="ns", utc=True, errors='coerce')
        self.trade_logs = trade_logs

    # @profile
    def add_session(self, start_index, start_time_index, prev_index, prev_time_index, up_trend):
        session = MyModel(f"Session_{self.next_session_id}", start_index, start_time_index, prev_index, prev_time_index, up_trend)
        session.original_offset = prev_index - 100
        self.sessions[self.next_session_id] = session
        self.next_session_id += 1
        return session

    def summarize_and_export_results(self, filename="final_trade_logs.csv", initial_capital=10000, risk_percentage=10.0):
        """
        trade_logs の統計情報を計算し、CSVの最後に以下の情報を追加する：
          ・プロフィットファクター：勝ちトレードの総利益 / 負けトレードの総損失
          ・平均リスクリワード比率：各トレードでの risk_reward_ratio の平均値
          ・最大連敗数：連続した負けトレードの最大回数
          ・最大ドローダウン：資金推移における最大の落ち込み
          ・初期資金からの資金推移：1万円スタート時のシミュレーション結果
          ・設定リスク（%）および1回あたりのリスク金額（初期資金×risk_percentage/100）
        
        ※動的リスク管理シミュレーション：各トレードごとに、エントリー時の資金に対して risk_percentage(例：3%)をリスク金額とし、
          勝ちの場合はそのリスク金額×reward_ratio、負けの場合はそのリスク金額分資金から差し引くシミュレーションを行う。
        """
        df = self.trade_logs.copy()
        if df.empty or "profit_loss" not in df.columns:
            print("トレードログが空か、必要な列がありません。")
            return

        # 既存の累積損益（固定値）ではなく、動的リスク管理シミュレーションを実施
        df = df.sort_values("entry_time").reset_index(drop=True)
        capital_list = []
        simulated_pl_list = []
        current_capital = initial_capital
        capital_list.append(current_capital)
        for idx, row in df.iterrows():
            # 各トレードのリスク金額は、エントリー時の資金の risk_percentage%
            risk = current_capital * (risk_percentage / 100.0)
            if row["result"] == "win":
                # 勝ちトレードの場合、利益 = risk × リスクリワード比率
                pl = risk * row["risk_reward_ratio"]
            else:
                # 負けトレードの場合、損失 = risk
                pl = -risk
            simulated_pl_list.append(pl)
            current_capital += pl
            capital_list.append(current_capital)
        df["simulated_pl"] = simulated_pl_list

        # シミュレーションによる資金推移（各トレード後の資金）
        simulated_capital = pd.Series(capital_list[1:], name="capital")
        df["capital"] = simulated_capital

        # シミュレーションによる最大ドローダウンの計算
        cummax = simulated_capital.cummax()
        drawdown = cummax - simulated_capital
        sim_max_drawdown = drawdown.max()

        # 最大連敗数の計算（シミュレーションではなくログ上の profit_loss を使用）
        df["is_loss"] = df["profit_loss"] < 0
        df["loss_group"] = (~df["is_loss"]).cumsum()
        losing_streak = df.groupby("loss_group").cumcount() + 1
        max_consecutive_losses = losing_streak.max()

        # プロフィットファクターの計算（ログの profit_loss を使用）
        total_win = df.loc[df["profit_loss"] > 0, "profit_loss"].sum()
        total_loss = -df.loc[df["profit_loss"] < 0, "profit_loss"].sum()
        profit_factor = total_win / total_loss if total_loss > 0 else np.inf

        # 平均リスクリワード比率（ログの risk_reward_ratio の平均）
        average_rr = df["risk_reward_ratio"].mean()

        risk_amount = initial_capital * (risk_percentage / 100.0)

        # 勝率の計算
        win_rate = (df["result"] == "win").mean() * 100

        # CSV出力
        df.to_csv(filename, index=False)
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["統計情報"])
            writer.writerow(["プロフィットファクター", profit_factor])
            writer.writerow(["平均リスクリワード比率", average_rr])
            writer.writerow(["最大連敗数", max_consecutive_losses])
            writer.writerow(["最大ドローダウン（シミュレーション）", sim_max_drawdown])
            writer.writerow(["初期資金", initial_capital])
            writer.writerow(["最終資金（シミュレーション）", simulated_capital.iloc[-1]])
            writer.writerow(["設定リスク（%）", risk_percentage])
            writer.writerow(["1回あたりのリスク金額（初期時）", risk_amount])
            writer.writerow(["勝率（%）", win_rate])
        print(f"トレードログと統計情報を {filename} に書き出しました。")

    def __repr__(self):
        return f"WaveManager(sessions={list(self.sessions.values())})"

def detect_extension_reversal(prices, lower1_percent=None, lower2_percent=None, higher1_percent=None, higher2_percent=None):
    price1 = prices[0]
    price2 = prices[1]
    low_val = min(price1, price2)
    high_val = max(price1, price2)
    wave_range = high_val - low_val
    low1 = low_val - (-wave_range * lower1_percent) if lower1_percent is not None else None
    high1 = high_val - (-wave_range * higher1_percent) if higher1_percent is not None else None
    low2 = low_val - (-wave_range * lower2_percent) if lower2_percent is not None else None
    high2 = high_val - (-wave_range * higher2_percent) if higher2_percent is not None else None
    return (low1, low2, high1, high2)

def detect_small_reversal(base_p, end_adjustment_p):
    low_val = min(base_p, end_adjustment_p)
    high_val = max(base_p, end_adjustment_p)

def get_out_of_range(low, high):
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

def detect_pivots(np_arr, time_df, name, POINT_THRESHOLD, LOOKBACK_BARS, consecutive_bars, arrow_spacing):
    if name == "BASE_SMA":
        wm = WaveManager()
    sma = np_arr[:, -2]
    trend_arr = np_arr[:, -1]
    pivot_data = []
    last_pivot_index = 0
    last_detect_index = 0
    run_counter = 1
    n = np_arr.shape[0]
    up_trend = None
    for i in range(1, n):
        if trend_arr[i] == trend_arr[i-1]:
            run_counter += 1
        else:
            run_counter = 1
        if run_counter < consecutive_bars:
            continue
        if i - last_pivot_index < arrow_spacing:
            continue
        if up_trend is None:
            if trend_arr[i] == 0.0:
                up_trend = True
            elif trend_arr[i] == 1.0:
                up_trend = False
        if up_trend and (trend_arr[i] == 0.0):
            start = max(0, i - LOOKBACK_BARS)
            window_sma = sma[start: i+1]
            sma_max = np.nanmax(window_sma)
            if (sma_max - sma[i]) >= POINT_THRESHOLD:
                window_high = np_arr[last_pivot_index: i+1, 2]
                local_high_idx = np.argmax(window_high) + last_pivot_index if last_pivot_index >= 0 else start
                detection_time = np_arr[i, 0]
                pivot_time = np_arr[local_high_idx, 0]
                pivot_value = np_arr[local_high_idx, 2]
                pivot_type = 1.0
                pivot_data.append((detection_time, pivot_time, pivot_value, pivot_type))
                up_trend = False
                if name == "BASE_SMA" and last_pivot_index > 150 and n - i > 1500:
                    wm.add_session(start_index=local_high_idx, start_time_index=i, prev_index=last_pivot_index, prev_time_index=last_detect_index, up_trend="False")
                last_pivot_index = local_high_idx
                last_detect_index = i
        elif (not up_trend) and (trend_arr[i] == 1.0):
            start = max(0, i - LOOKBACK_BARS)
            window = sma[start: i+1]
            window_min = np.nanmin(window)
            if (sma[i] - window_min) >= POINT_THRESHOLD:
                window_min = np_arr[last_pivot_index: i+1, 3]
                local_min_idx = np.argmin(window_min) + last_pivot_index if last_pivot_index >= 0 else start
                detection_time = np_arr[i, 0]
                pivot_time = np_arr[local_min_idx, 0]
                pivot_value = np_arr[local_min_idx, 3]
                pivot_type = 0.0
                pivot_data.append((detection_time, pivot_time, pivot_value, pivot_type))
                up_trend = True
                if name == "BASE_SMA" and last_pivot_index > 100 and n - i > 1500:
                    wm.add_session(start_index=local_min_idx, start_time_index=i, prev_index=last_pivot_index, prev_time_index=last_detect_index, up_trend="True")
                last_pivot_index = local_min_idx
                last_detect_index = i
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
    np_arr_with_base_sml_sma = np.column_stack((base_np, sml_sma_arr.reshape(-1, 1)))
    columns = ["time", "open", "high", "low", "close", "tick_volume", "spread", "BASE_SMA", "SML_SMA"]
    df = pd.DataFrame(np_arr_with_base_sml_sma, columns=columns)
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
    merged_temp = pd.merge_asof(df_sorted, df_pivot_sorted,
                                left_on="time", right_on="detection_time",
                                direction="nearest", tolerance=pd.Timedelta("2sec"))
    merged_temp2 = pd.merge_asof(merged_temp, sml_df_pivot_sorted,
                                 left_on="time", right_on="sml_detection_time",
                                 direction="nearest", tolerance=pd.Timedelta("2sec"))
    merged_temp2.reset_index(drop=True, inplace=True)
    additional_sma_df.reset_index(drop=True, inplace=True)
    final_merged = pd.concat([merged_temp2, additional_sma_df], axis=1)
    final_merged = final_merged.drop(columns=["detection_time", "sml_detection_time"])
    print("最終出力データのマージが完了しました。")
    print(final_merged.head())
    return final_merged

def merge_all_results(final_df, merged_tf_df):
    base_columns = ["time", "open", "high", "low", "close", "tick_volume", "spread", "BASE_SMA", "SML_SMA"]
    extra_cols = merged_tf_df.drop(columns=base_columns)
    combined_df = pd.merge(final_df, extra_cols, on="time", how="left")
    return combined_df

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

def process_data(conditions):
    global tp_level_global, check_no_SMA_global, range_param_global, stop_loss_global, time_df
    print("Current working directory:", os.getcwd())
    print(f"テスト開始時間 {datetime.now()}")
    symbol = conditions.get("symbol", "USDJPY")
    fromdate = conditions.get("fromdate", datetime(2023, 12, 1, 20, 0, tzinfo=pytz.UTC))
    todate = conditions.get("todate", datetime(2025, 2, 23, 6, 50, tzinfo=pytz.UTC))
    BASE_SMA = conditions.get("BASE_SMA", 20)
    BASE_threshold = conditions.get("BASE_threshold", 0.005)
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
    origin_df["time"] = pd.to_datetime(origin_df["time"], utc=True)
    origin_df = origin_df.set_index("time")
    origin_df = origin_df.loc[fromdate:todate].reset_index()
    base_df = origin_df.iloc[:, :7]
    sma_df = origin_df.iloc[:, 7:]
    print(base_df.head())
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
    final_df = merge_arr(base_arr, sml_arr, sma_df)
    wm.full_data = final_df.to_numpy(dtype=np.float64)
    wm.analyze_sessions()
    # risk_percentage を conditions から取得（例：3.0%）
    risk_percentage = conditions.get("risk_percentage", 3.0)
    
    print("処理終了")
    print(f"終了時間 {datetime.now()}")

if __name__ == "__main__":
    conditions = {
        "symbol": "USDJPY",
        "fromdate": datetime(2025, 1, 2, 0, 0, tzinfo=pytz.UTC),
        "todate": datetime(2025, 2, 20, 18, 0, tzinfo=pytz.UTC),
        "BASE_SMA": 20,
        "BASE_threshold": 0.009,
        "BASE_lookback": 15,
        "BASE_consecutive": 3,
        "BASE_arrow_spacing": 8,
        "SML_SMA": 4,
        "SML_threshold": 0.003,
        "SML_lookback": 3,
        "SML_consecutive": 1,
        "SML_arrow_spacing": 2,
        "range": 80,
        "stop": "sml",
        "tp_level": 138,
        "check_no_sma": True,
        "output_file": "USDJPY_138_trade_logs調整.csv",
        "risk_percentage": 3.0  # ここで1回あたりの損失許容額（%）を指定（例：3%）
    }
    start = time.time()
    process_data(conditions)
    end = time.time()
    time_diff = end - start
    print(time_diff)
