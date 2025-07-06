import numpy as np
import pandas as pd


#############################################
# セッションクラス（各セッションの状態を管理）
#############################################
class MyModel(object):
    def __init__(self, name, start_index, start_time_index, prev_index, prev_time_index, up_trend="True", stop_strategy="sml", tp_level=138):
        self.name = name
        self.state = "created_base_arrow"
        self.stop_strategy = stop_strategy  # stop戦略を保持
        self.tp_level = tp_level / 100.0  # tp_levelを割合に変換して保持 (例: 138 -> 1.38)
        self.start_origin = None
        self.start_pivot_time = None
        self.original_offset = None
        self.full_data = None
        self.start_index = start_index  # 始まったピボットのインデックス
        self.start_time_index = start_time_index
        self.prev_index = prev_index
        self.prev_time_index = prev_time_index
        # self.machine = Machine(model=self, states=states, transitions=transitions, initial="created_base_arrow")
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
        
    def set_state(self, new_state):
        self.state = new_state
        if new_state == "created_new_arrow":
            self.on_enter_created_new_arrow()
        elif new_state == "infibos":
            self.on_enter_infibos()
        elif new_state == "has_position":
            self.on_enter_has_position()
        elif new_state == "closed":
            self.on_enter_closed()

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
            _, _, self.fibo_minus_20, self.fibo_minus_200 = calculate_fibo_levels(prices, None, None, 0.2, 2)
            judged_price = self.full_data[self.new_arrow_index, 2]
        else:
            prices = (self.full_data[self.prev_index, 3], self.full_data[self.start_index, 2])
            self.fibo_minus_20, self.fibo_minus_200, _, _ = calculate_fibo_levels(prices, -0.2, -2, None, None)
            judged_price = self.full_data[self.new_arrow_index, 3]

        if self.watch_price_in_range(self.fibo_minus_20, self.fibo_minus_200, judged_price):
            self.set_state("created_new_arrow")
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
                        self.set_state("infibos")
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
                        self.set_state("has_position")
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
                                    self.full_data[self.new_arrow_index, 2], # 推進波の終点
                                    self.highlow_since_new_arrow[1]          # 調整波の終点
                                )
                                self.take_profit = detect_extension_reversal(prices_data_to_get_take_profit, self.tp_level, self.up_trend)
                                self.entry_line = neck_price + 0.006
                                self.entry_index = global_index
                                self.point_to_stoploss = abs(self.entry_line - self.highlow_stop_loss)
                                self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                                # リスクリワード比率が2以下なら候補を削除（np.array_equal を利用）
                                if (self.point_to_take_profit / self.point_to_stoploss) <= 0.1:
                                    self.determined_neck = [
                                        n for n in self.determined_neck
                                        if not np.array_equal(n, neckline)
                                    ]
                                else:
                                    self.set_state("has_position")
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
                                    self.full_data[self.new_arrow_index, 3], # 推進波の終点
                                    self.highlow_since_new_arrow[0]          # 調整波の終点
                                )
                                self.take_profit = detect_extension_reversal(prices_data_to_get_take_profit, self.tp_level, self.up_trend)
                                self.entry_line = neck_price - 0.006
                                self.entry_index = global_index
                                self.point_to_stoploss = abs(self.entry_line - self.highlow_stop_loss)
                                self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                                if (self.point_to_take_profit / self.point_to_stoploss) <= 0.1:
                                    self.determined_neck = [
                                        n for n in self.determined_neck
                                        if not np.array_equal(n, neckline)
                                    ]
                                else:
                                    self.set_state("has_position")
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
        self.set_state("closed")

    def handle_closed(self):
        if not self.destroy_reqest:
            self.destroy_reqest = True

    def on_enter_created_new_arrow(self):
        state_name = self.state
        self.state_times[state_name] = self.new_arrow_index
        if self.up_trend:
            prices = (self.full_data[self.start_index, 3], self.full_data[self.new_arrow_index, 2])
            self.base_fibo70, _, self.base_fibo37, _ = calculate_fibo_levels(prices, lower1_percent=0.3, higher1_percent=-0.37)
        else:
            prices = (self.full_data[self.start_index, 2], self.full_data[self.new_arrow_index, 3])
            self.base_fibo37, _, self.base_fibo70, _ = calculate_fibo_levels(prices, lower1_percent=0.37, higher1_percent=-0.3)
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
                self.set_state("infibos")
            else:
                return
        else:
            if self.watch_price_in_range(self.base_fibo37, self.base_fibo70, highest):
                self.index_of_fibo37 = self.get_touch37_index()
                self.start_of_simulation = self.new_arrow_detection_index
                self.set_state("infibos")
                return
            else:
                return

    def on_enter_infibos(self):
        self.get_potential_neck_wheninto_newarrow()

    def on_enter_has_position(self):
        self.entry_time = self.full_data[self.entry_index, 0]
        entry_time_dt = pd.to_datetime(self.entry_time, unit='ns', utc=True)
        # if entry_time_dt == pd.Timestamp("2025-01-28 01:53:00+00:00"):
        # print(f"DEBUG: Entry at {entry_time_dt}, Stoploss: {self.highlow_stop_loss}")
        self.global_entry_index = self.original_offset + self.entry_index
        self.set_state("closed")

    def on_enter_closed(self):
        if self.stop_strategy == "fibo":
            self.trade_log = self.calculate_trade_result_fibo()
        else: # デフォルトはsml
            self.trade_log = self.calculate_trade_result_sml()
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
                            fibo32_of_ptl_neck = calculate_fibo_levels(prices, -0.32, 0.32, None, None)
                            if self.watch_price_in_range(fibo32_of_ptl_neck[0], fibo32_of_ptl_neck[1], sml_pvts[i+1, 3]):
                                determined_neck = sml_pvts[i]
                        else:
                            potential_neck = sml_pvts[i]
                else:
                    if sml_pvts[i, 3] == 0 and sml_pvts[i, 2] > self.index_of_fibo37:
                        if i + 1 < sml_pvts.shape[0]:
                            prices = (sml_pvts[i-1, 2], sml_pvts[i, 2])
                            fibo32_of_ptl_neck = calculate_fibo_levels(prices, None, None, 0.32, -0.32)
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
                fibo32_of_ptl_neck = calculate_fibo_levels(pvts_to_get_32level, -0.32, 0.32, None, None)
                result = self.watch_price_in_range(fibo32_of_ptl_neck[0], fibo32_of_ptl_neck[1], judged_price)
                if result:
                    self.determined_neck.append(self.potential_neck[-1])
                    self.potential_neck.clear()
                    self.organize_determined_neck()
                else:
                    self.potential_neck.clear()
            else:
                judged_price = sml_pvts[-1][2]
                fibo32_of_ptl_neck = calculate_fibo_levels(pvts_to_get_32level, None, None, -0.32, 0.32)
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
                fibo32_of_ptl_neck = calculate_fibo_levels(sml_index_to_get32, -0.382, 0.382, None, None)
                last_highlow = self.get_high_and_low_in_term(self.potential_neck[-1][0], global_index)
                if self.watch_price_in_range(fibo32_of_ptl_neck[0], fibo32_of_ptl_neck[1], last_highlow[1]):
                    # エントリーラインおよびストップロス／テイクプロフィットの計算
                    self.highlow_since_new_arrow = self.get_high_and_low_in_term(self.index_of_fibo37, global_index + 1)
                    self.highlow_stop_loss = self.highlow_since_new_arrow[1] - 0.006
                    self.sml_stop_loss = last_highlow[1] - 0.006
                    self.final_neckline_index = self.potential_neck[-1][1]
                    prices_data_to_get_take_profit = (
                        self.full_data[self.new_arrow_index, 2], # 推進波の終点
                        self.highlow_since_new_arrow[1]          # 調整波の終点
                    )
                    self.take_profit = detect_extension_reversal(prices_data_to_get_take_profit, self.tp_level, self.up_trend)
                    self.entry_line = neck_price + 0.006
                    self.entry_index = global_index
                    self.point_to_stoploss = abs(self.entry_line - self.highlow_stop_loss)
                    self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                    # リスクリワード比のチェック：2未満なら候補を削除してFalseを返す
                    if (self.point_to_take_profit / self.point_to_stoploss) < 0.1:
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
                fibo32_of_ptl_neck = calculate_fibo_levels(sml_index_to_get32, None, None, -0.382, 0.382)
                last_highlow = self.get_high_and_low_in_term(self.potential_neck[-1][0], global_index)
                if self.watch_price_in_range(fibo32_of_ptl_neck[2], fibo32_of_ptl_neck[3], last_highlow[0]):
                    self.highlow_since_new_arrow = self.get_high_and_low_in_term(self.index_of_fibo37, global_index + 1)
                    self.highlow_stop_loss = self.highlow_since_new_arrow[0] + 0.006
                    self.sml_stop_loss = last_highlow[0] + 0.006
                    self.final_neckline_index = self.potential_neck[-1][1]
                    prices_data_to_get_take_profit = (
                        self.full_data[self.new_arrow_index, 3], # 推進波の終点
                        self.highlow_since_new_arrow[0]          # 調整波の終点
                    )
                    self.take_profit = detect_extension_reversal(prices_data_to_get_take_profit, self.tp_level, self.up_trend)
                    self.entry_line = neck_price - 0.006
                    self.entry_index = global_index
                    self.point_to_stoploss = abs(self.entry_line - self.highlow_stop_loss)
                    self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                    # リスクリワード比のチェック：2未満なら候補を削除してFalseを返す
                    if (self.point_to_take_profit / self.point_to_stoploss) < 0.1:
                        self.potential_neck.pop()
                        return False
                    else:
                        return True
            elif arr[3] < neck_price and not self.check_no_SMA(global_index, neck_price):
                return False
        return None



    def calculate_trade_result_fibo(self):
        """
        trade_log を計算するメソッド（セッション全体は self で管理）
        """
        data_slice = self.full_data[self.entry_index:]
        if data_slice.size == 0:
            # データが空の場合の特別処理
            if self.next_next_new_arrow_index and self.next_next_new_arrow_index > 0:
                exit_price = self.full_data[self.next_next_new_arrow_index - 1, 4]
                exit_time = self.full_data[self.next_next_new_arrow_index - 1, 0]
            else:
                exit_price = self.entry_line
                exit_time = self.entry_time
            exit_reason = "forced close"
            result = "forced"
            profit_loss = 0
        else:
            highs = data_slice[:, 2]
            lows = data_slice[:, 3]

            if self.up_trend:
                tp_hit_indices = np.where(highs >= self.take_profit)[0]
                sl_hit_indices = np.where(lows <= self.highlow_stop_loss)[0]
            else:
                tp_hit_indices = np.where(lows <= self.take_profit)[0]
                sl_hit_indices = np.where(highs >= self.highlow_stop_loss)[0]

            tp_index = tp_hit_indices[0] if tp_hit_indices.size > 0 else np.inf
            sl_index = sl_hit_indices[0] if sl_hit_indices.size > 0 else np.inf

            # 同時ヒットの判定
            if tp_index != np.inf and tp_index == sl_index:
                # 同じ足で両方にヒットした場合
                exit_index = tp_index
                exit_price = (self.take_profit + self.highlow_stop_loss) / 2
                exit_reason = "SL/TP simultaneous hit"
                result = "breakeven"
            elif tp_index < sl_index:
                # 利確が先にヒットした場合
                exit_index = tp_index
                exit_price = self.take_profit
                exit_reason = "T/P"
                result = "win"
            elif sl_index < tp_index:
                # 損切りが先にヒットした場合
                exit_index = sl_index
                exit_price = self.highlow_stop_loss
                exit_reason = "S/L (highlow)"
                result = "loss"
            else:
                # どちらもヒットしなかった場合
                exit_index = len(data_slice) - 1
                exit_price = data_slice[exit_index, 4]
                exit_reason = "forced close"
                result = "forced"

            exit_time = data_slice[exit_index, 0]
            profit_loss = (exit_price - self.entry_line if self.up_trend 
                        else self.entry_line - exit_price)

        return self.format_trade_log(exit_time, result, exit_price, exit_reason, profit_loss)
    
    # def calculate_trade_result_sml(self):
    #     """
    #     高速なトレード結果判定（SMLピボットベースの決済）
    #     ・エントリー後から最初のSMLピボットまでの区間をチェックし、SL条件があれば決済
    #     ・その後、各ピボット間の区間で、前のピボットの安値がブレイクされたかをチェック
    #     ・最終ピボット以降もチェックし、どれもヒットしなければ最終バーで強制決済
    #     """
    #     # SMLピボット（安値ピボット）のインデックスを取得
    #     pivot_indices = self.get_pivots_in_range(self.entry_index, len(self.full_data)-1, base_or_sml="sml")
    #     # typeが0.0（安値ピボット）のものだけを残す
    #     pivot_indices = [idx for idx in pivot_indices if self.full_data[idx, 14] == 0.0]

    #     current_sl = self.highlow_stop_loss
    #     entry_price = self.entry_line
    #     exit_price, exit_time, exit_reason, result = None, None, None, None

    #     # ①エントリー後から最初のピボットまでのスライスをチェック
    #     if pivot_indices:
    #         first_pivot = pivot_indices[0]
    #         slice_data = self.full_data[self.entry_index:first_pivot, 3]  # 3: low列
    #         if np.any(slice_data <= current_sl):
    #             exit_price = current_sl
    #             exit_time = self.full_data[self.entry_index + np.argmax(slice_data <= current_sl), 0]
    #             exit_reason, result = "S/L", "loss"
    #         else:
    #             # ②2番目以降のピボットについてチェックする
    #             for i in range(len(pivot_indices) - 1):
    #                 current_pivot_idx = pivot_indices[i]
    #                 next_pivot_idx = pivot_indices[i + 1]
    #                 current_pivot_price = self.full_data[current_pivot_idx, 13]  # 13: pivot price
    #                 # 現在のピボットと次のピボットの間のlow値を確認
    #                 slice_lows = self.full_data[current_pivot_idx:next_pivot_idx, 3]
    #                 if np.any(slice_lows <= current_pivot_price):
    #                     exit_price = current_pivot_price
    #                     exit_time = self.full_data[current_pivot_idx + np.argmax(slice_lows <= current_pivot_price), 0]
    #                     exit_reason = "SML pivot break"
    #                     # エントリー価格より高ければwin、そうでなければ微損
    #                     result = "win" if exit_price > entry_price else "微損"
    #                     break
    #             else:
    #                 # ③最後のピボット以降〜最終バーまでチェック
    #                 last_pivot_idx = pivot_indices[-1]
    #                 last_pivot_price = self.full_data[last_pivot_idx, 13]
    #                 slice_lows = self.full_data[last_pivot_idx:, 3]
    #                 if np.any(slice_lows <= last_pivot_price):
    #                     exit_price = last_pivot_price
    #                     exit_time = self.full_data[last_pivot_idx + np.argmax(slice_lows <= last_pivot_price), 0]
    #                     exit_reason = "SML pivot break"
    #                     result = "win" if exit_price > entry_price else "微損"
    #                 else:
    #                     # どの条件もヒットしなければ、最終バーで強制決済
    #                     exit_price = self.full_data[-1, 4]  # 4: close列
    #                     exit_time = self.full_data[-1, 0]
    #                     exit_reason, result = "forced close", "forced"
    #     else:
    #         # SMLピボットが全くなかった場合はエントリー後から最後までチェック
    #         slice_data = self.full_data[self.entry_index:, 3]
    #         if np.any(slice_data <= current_sl):
    #             exit_price = current_sl
    #             exit_time = self.full_data[self.entry_index + np.argmax(slice_data <= current_sl), 0]
    #             exit_reason, result = "S/L", "loss"
    #         else:
    #             exit_price = self.full_data[-1, 4]
    #             exit_time = self.full_data[-1, 0]
    #             exit_reason, result = "forced close", "forced"

    #     profit_loss = (exit_price - entry_price) if self.up_trend else (entry_price - exit_price)
    #     trade_log = {
    #         "entry_time": pd.to_datetime(self.entry_time, unit="ns", utc=True),
    #         "up_trend": self.up_trend,
    #         "exit_time": pd.to_datetime(exit_time, unit="ns", utc=True),
    #         "result": result,
    #         "entry_price": entry_price,
    #         "exit_price": exit_price,
    #         "exit_reason": exit_reason,
    #         "profit_loss": profit_loss,
    #         "risk_reward_ratio": self.point_to_take_profit / self.point_to_stoploss
    #     }
    #     return trade_log

    def calculate_trade_result_sml(self):
        """
        新しい決済ロジック（ネックライン割れ決済とトレーリング利確）
        ルール1：初期損切り
        ルール2：最初のネックライン割れでの決済（微損 or 微益）
        ルール3：2つ目以降のネックラインを使ったトレーリング利確
        """
        entry_price = self.entry_line
        initial_sl = self.highlow_stop_loss
        take_profit = self.take_profit # 利確ラインも考慮に入れる

        # トレンドに応じて高値/安値、ピボットタイプ、比較演算子を切り替える
        if self.up_trend:
            price_col, opposite_price_col = 3, 2 # low, high
            pvt_type = 0.0 # 安値ピボット
            is_sl_hit = lambda price, sl: price <= sl
            is_tp_hit = lambda price, tp: price >= tp
            is_neck_break = lambda price, neck: price <= neck
        else: # Down Trend
            price_col, opposite_price_col = 2, 3 # high, low
            pvt_type = 1.0 # 高値ピボット
            is_sl_hit = lambda price, sl: price >= sl
            is_tp_hit = lambda price, tp: price <= tp
            is_neck_break = lambda price, neck: price >= neck

        # エントリー後のSMLピボット（＝ネックライン候補）を取得
        all_pivots = self.get_pivots_in_range(self.entry_index, len(self.full_data) - 1, base_or_sml="sml")
        neckline_pivots_indices = [idx for idx in all_pivots if self.full_data[idx, 14] == pvt_type]

        # --- ルール1 & 2：最初のネックラインが形成されるまでの処理 ---
        first_neck_idx = neckline_pivots_indices[0] if neckline_pivots_indices else len(self.full_data) -1

        # エントリーから最初のネックラインまでの価格データを取得
        data_before_first_neck = self.full_data[self.entry_index : first_neck_idx + 1]

        # 初期損切りラインにヒットしたかチェック
        sl_hits = np.where(is_sl_hit(data_before_first_neck[:, price_col], initial_sl))[0]
        if sl_hits.size > 0:
            # ルール1：完全な負け
            exit_idx = self.entry_index + sl_hits[0]
            exit_price = initial_sl
            exit_time = self.full_data[exit_idx, 0]
            exit_reason, result = "S/L (Initial)", "loss"
            profit_loss = (exit_price - entry_price) if self.up_trend else (entry_price - exit_price)
            return self.format_trade_log(exit_time, result, exit_price, exit_reason, profit_loss)

        # 利確ラインにヒットしたかチェック
        tp_hits = np.where(is_tp_hit(data_before_first_neck[:, opposite_price_col], take_profit))[0]
        if tp_hits.size > 0:
            exit_idx = self.entry_index + tp_hits[0]
            exit_price = take_profit
            exit_time = self.full_data[exit_idx, 0]
            exit_reason, result = "T/P", "win"
            profit_loss = (exit_price - entry_price) if self.up_trend else (entry_price - exit_price)
            return self.format_trade_log(exit_time, result, exit_price, exit_reason, profit_loss)

        # ネックラインが一つも形成されずに終了した場合
        if not neckline_pivots_indices:
            exit_idx = len(self.full_data) - 1
            exit_price = self.full_data[exit_idx, 4] # close
            exit_time = self.full_data[exit_idx, 0]
            exit_reason, result = "Forced Close", "forced"
            profit_loss = (exit_price - entry_price) if self.up_trend else (entry_price - exit_price)
            return self.format_trade_log(exit_time, result, exit_price, exit_reason, profit_loss)

        # --- ルール3：2つ目以降のネックラインを使ったトレーリング利確 ---
        # ここに来た時点で、最初のネックラインは割れていないことが確定している
        # 負けはないので、ここからは利確モード
        
        # 最初のネックラインを初期のトレーリングストップラインとする
        trailing_sl_price = self.full_data[first_neck_idx, 13]

        # 2つ目以降のネックラインをループ処理
        for i in range(len(neckline_pivots_indices) - 1):
            current_neck_idx = neckline_pivots_indices[i]
            next_neck_idx = neckline_pivots_indices[i+1]
            
            # 現在のネックラインから次のネックラインまでの価格データを取得
            data_between_necks = self.full_data[current_neck_idx : next_neck_idx + 1]

            # トレーリングストップにヒットしたかチェック
            break_hits = np.where(is_neck_break(data_between_necks[:, price_col], trailing_sl_price))[0]
            if break_hits.size > 0:
                exit_idx = current_neck_idx + break_hits[0]
                exit_price = trailing_sl_price
                exit_time = self.full_data[exit_idx, 0]
                exit_reason = "Trailing SL"
                # この時点で決済価格がエントリー価格を上回っていれば勝ち、そうでなければ微損
                result = "win" if (self.up_trend and exit_price > entry_price) or \
                                  (not self.up_trend and exit_price < entry_price) else "微損"
                profit_loss = (exit_price - entry_price) if self.up_trend else (entry_price - exit_price)
                return self.format_trade_log(exit_time, result, exit_price, exit_reason, profit_loss)

            # 利確ラインにヒットしたかチェック
            tp_hits_trailing = np.where(is_tp_hit(data_between_necks[:, opposite_price_col], take_profit))[0]
            if tp_hits_trailing.size > 0:
                exit_idx = current_neck_idx + tp_hits_trailing[0]
                exit_price = take_profit
                exit_time = self.full_data[exit_idx, 0]
                exit_reason, result = "T/P", "win"
                profit_loss = (exit_price - entry_price) if self.up_trend else (entry_price - exit_price)
                return self.format_trade_log(exit_time, result, exit_price, exit_reason, profit_loss)

            # 次のループのためにトレーリングストップラインを更新
            trailing_sl_price = self.full_data[next_neck_idx, 13]

        # --- ループ終了後：最後のネックラインを基準に最後までチェック ---
        last_neck_idx = neckline_pivots_indices[-1]
        data_after_last_neck = self.full_data[last_neck_idx:]

        # 最後のトレーリングストップにヒットしたか
        final_break_hits = np.where(is_neck_break(data_after_last_neck[:, price_col], trailing_sl_price))[0]
        if final_break_hits.size > 0:
            exit_idx = last_neck_idx + final_break_hits[0]
            exit_price = trailing_sl_price
            exit_time = self.full_data[exit_idx, 0]
            exit_reason = "Trailing SL"
            result = "win" if (self.up_trend and exit_price > entry_price) or \
                              (not self.up_trend and exit_price < entry_price) else "微損"
            profit_loss = (exit_price - entry_price) if self.up_trend else (entry_price - exit_price)
            return self.format_trade_log(exit_time, result, exit_price, exit_reason, profit_loss)

        # 最後の利確ラインにヒットしたか
        final_tp_hits = np.where(is_tp_hit(data_after_last_neck[:, opposite_price_col], take_profit))[0]
        if final_tp_hits.size > 0:
            exit_idx = last_neck_idx + final_tp_hits[0]
            exit_price = take_profit
            exit_time = self.full_data[exit_idx, 0]
            exit_reason, result = "T/P", "win"
            profit_loss = (exit_price - entry_price) if self.up_trend else (entry_price - exit_price)
            return self.format_trade_log(exit_time, result, exit_price, exit_reason, profit_loss)

        # それでも決済されなければ、最終バーで強制決済
        exit_idx = len(self.full_data) - 1
        exit_price = self.full_data[exit_idx, 4] # close
        exit_time = self.full_data[exit_idx, 0]
        exit_reason, result = "Forced Close", "forced"
        profit_loss = (exit_price - entry_price) if self.up_trend else (entry_price - exit_price)
        return self.format_trade_log(exit_time, result, exit_price, exit_reason, profit_loss)

    def format_trade_log(self, exit_time, result, exit_price, exit_reason, profit_loss):
        """トレードログを整形して返すヘルパー関数"""
        return {
            "entry_time": pd.to_datetime(self.entry_time, unit="ns", utc=True),
            "up_trend": self.up_trend,
            "exit_time": pd.to_datetime(exit_time, unit="ns", utc=True),
            "result": result,
            "entry_price": self.entry_line,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "profit_loss": profit_loss,
            "risk_reward_ratio": self.point_to_take_profit / self.point_to_stoploss if self.point_to_stoploss > 0 else np.inf
        }





    def append_sml_pivot_data(self, required_data, new_sml_pivot_index):
        state_list = ["infibos"]
        
        # インデックス計算を明確に
        actual_index = (
            self.new_arrow_detection_index + new_sml_pivot_index
            if self.state == "created_new_arrow"
            else self.start_of_simulation + new_sml_pivot_index
        )

        row = self.full_data[actual_index]
        pivot_index = self.find_detection_index(row[12], 0)
        price = row[13]
        type_val = row[14]
        
        # 配列ではなくタプルを作成（hashableなため）
        new_row_tuple = (actual_index, pivot_index, price, type_val)
        
        # 初回のみセットを作成して高速検索
        if not hasattr(self, 'sml_pivot_set'):
            self.sml_pivot_set = set(tuple(existing) for existing in self.sml_pivots_after_goldencross)
        
        if new_row_tuple not in self.sml_pivot_set:
            self.sml_pivot_set.add(new_row_tuple)
            
            # numpy配列ではなくリストを使用
            if not hasattr(self, 'sml_pivot_list'):
                self.sml_pivot_list = list(self.sml_pivots_after_goldencross)
            
            self.sml_pivot_list.append(new_row_tuple)

            # sml_pivots_after_goldencrossを必要なときだけnumpy配列に変換する
            self.sml_pivots_after_goldencross = np.array(self.sml_pivot_list)

        # potential_neckの重複確認も同様のセットを使った方式で高速化
        if not hasattr(self, 'potential_neck_set'):
            self.potential_neck_set = set(tuple(existing) for existing in self.potential_neck)
        
        if new_row_tuple not in self.potential_neck_set:
            if self.state in state_list:
                if self.potential_neck:
                    self.check_potential_to_determine_neck()
                if not self.potential_neck:
                    if (self.up_trend and type_val == 1) or ((not self.up_trend) and type_val == 0):
                        self.potential_neck.append(new_row_tuple)
                        self.potential_neck_set.add(new_row_tuple)

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
    
def calculate_fibo_levels(prices, lower1_percent=None, lower2_percent=None, higher1_percent=None, higher2_percent=None):
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

def detect_extension_reversal(prices, extension_level, is_up_trend):
    """
    調整波の価格情報とextension_levelから、テイクプロフィット価格を計算する。
    :param prices: (推進波の終点価格, 調整波の終点価格) のタプル
    :param extension_level: 拡張レベル (例: 1.38)
    :param is_up_trend: トレンド方向
    :return: take_profit_price
    """
    wave_end_price, adjustment_end_price = prices
    adjustment_wave_length = abs(wave_end_price - adjustment_end_price)
    
    if is_up_trend:
        # 上昇トレンドの場合：調整波の安値 + (調整波の長さ * レベル)
        take_profit_price = adjustment_end_price + (adjustment_wave_length * extension_level)
    else:
        # 下降トレンドの場合：調整波の高値 - (調整波の長さ * レベル)
        take_profit_price = adjustment_end_price - (adjustment_wave_length * extension_level)
        
    return take_profit_price

def check_touch_line(center_price, tested_price):
    if center_price <= tested_price:
        return True
    elif center_price >= tested_price:
        return False