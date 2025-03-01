import time
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import threading
from transitions.extensions import HierarchicalMachine as Machine
import csv
import pandas as pd



# グローバル変数
current_price_global = []

symbol = "USDJPY"
last_pivot_data = 999
sml_last_pivot_data = 999

states = [
    "created_base_arrow",
    "touched_20",
    "created_new_arrow",
    {"name": "infibos", "children": "has_determined_neck", "initial": False},
    "has_position",
    "closed"
]

transitions = [
    {"trigger": "touch_20", "source": "created_base_arrow", "dest": "touched_20"},
    {"trigger": "create_new_arrow", "source": "touched_20", "dest": "created_new_arrow"},
    {"trigger": "touch_37", "source": ["created_new_arrow", "infibos_has_determined_neck"], "dest": "infibos"},
    {"trigger": "neck_determine", "source": "infibos", "dest": "infibos_has_determined_neck"},
    {"trigger": "build_position", "source": ["infibos", "infibos_has_determined_neck"], "dest": "has_position"},
    {"trigger": "close", "source": "has_position", "dest": "closed"}
]

#############################################
# セッションクラス（各セッションの状態管理）
#############################################
class MyModel(object):
    def __init__(self, name, pivot_data, up_trend):
        self.name = name
        self.pivot_data = pivot_data[-2:]  # セッション開始時のピボットデータ
        self.start_pivot = pivot_data[-1] if pivot_data else datetime.now()
        self.machine = Machine(model=self, states=states, transitions=transitions, initial="created_base_arrow")
        self.new_arrow_pivot = None  # 推進波終了時のピボット
        self.base_fibo37 = []  # 37%ライン用
        self.base_fibo70 = []
        self.time_of_goldencross = []
        self.highlow_since_new_arrow = []  # 調整波の高値/安値
        self.sml_pivot_data = []  # スモール足ピボット記録

        # ネックライン候補関連
        self.sml_pivots_after_goldencross = []
        self.potential_neck = []
        self.determined_neck = []

        self.destroy_reqest = False

        # up_trend の文字列引数から論理値を設定
        self.up_trend = True if up_trend == "True" else False

        self.state_times = {}  # 各状態遷移時の時刻記録

        # フィボナッチ用ライン計算（直前2ピボットから）
        if self.up_trend is True:
            _, _, self.fibo_minus_20, self.fibo_minus_200 = detect_extension_reversal(self.pivot_data[-2:], None, None, 0.2, 2)
        else:
            self.fibo_minus_20, self.fibo_minus_200, _, _ = detect_extension_reversal(self.pivot_data[-2:], -0.2, -2, None, None)

        # 各状態に対応する処理関数のディスパッチテーブル
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
        action = self.state_actions.get(self.state)
        if action:
            action(df.copy(), sml_df.copy())

    def handle_created_base_arrow(self, df, sml_df):
        if self.up_trend is True and check_touch_line(self.fibo_minus_20, df.iloc[-1]["high"]):
            self.touch_20()
        elif self.up_trend is False and not check_touch_line(self.fibo_minus_20, df.iloc[-1]["low"]):
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
        if self.up_trend is True and not check_touch_line(self.base_fibo37, self.highlow_since_new_arrow[1]):
            self.touch_37()
        elif self.up_trend is False and check_touch_line(self.base_fibo37, self.highlow_since_new_arrow[0]):
            self.touch_37()
        self.price_in_range_while_adjustment(df)

    def handle_infibos(self, df, sml_df):
        if self.potential_neck:
            entry_result = self.potential_entry(df, self.potential_neck)
            if entry_result is True:
                self.build_position()
                print(f" {self.name}: ビルドポジション1: {self.state}")
                print(f"まだインフィボ：名前：{self.name},テイクプロフィット：{self.take_profit}, ストップロス：{self.stop_loss}, エントリーライン：{self.entry_line}")
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
                print(f" {self.name}: ビルドポジション2: {self.state}")
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
                        pivots_data = (self.start_pivot, self.new_arrow_pivot)
                        highlow = detect_extension_reversal(pivots_data, higher1_percent=0.32)
                        self.take_profit = highlow[2]
                        self.entry_line = neckline[1] + 0.002
                        self.entry_pivot = df.iloc[-1]
                        self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                        self.point_to_stoploss = abs(self.entry_line - self.stop_loss)
                        self.build_position()
                        print(f" {self.name}: ビルドポジション3: {self.state}")
                    elif df.iloc[-1]["high"] > neckline[1] and not self.check_no_SMA(df.iloc[-1050:], neckline[1]):
                        self.determined_neck.remove(neckline)
                        if not self.determined_neck:
                            self.touch_37()
            if self.up_trend is False:
                for neckline in self.determined_neck[:]:
                    if self.state != 'infibos_has_determined_neck':
                        break
                    if df.iloc[-1]["low"] < neckline[1] and not self.check_no_SMA(df.iloc[-1050:], neckline[1]):
                        self.stop_loss = self.highlow_since_new_arrow[0] + 0.006
                        pivots_data = (self.start_pivot, self.new_arrow_pivot)
                        highlow = detect_extension_reversal(pivots_data, lower1_percent=-0.32)
                        self.take_profit = highlow[0]
                        self.entry_line = neckline[1] - 0.002
                        self.entry_pivot = df.iloc[-1]
                        self.point_to_stoploss = abs(self.entry_line - self.stop_loss)
                        self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                        self.build_position()
                        print(f" {self.name}: ビルドポジション4: {self.state}")
                    elif df.iloc[-1]["low"] < neckline[1] and not self.check_no_SMA(df.iloc[-1050:], neckline[1]):
                        self.determined_neck.remove(neckline)
                        if not self.determined_neck:
                            self.touch_37()
        self.price_in_range_while_adjustment(df)

    def handle_has_position(self, df, sml_df):
        print(f" {self.name}: ポジション監視中: {self.state}")
        should_close = False
        
        if self.up_trend:
            if df.iloc[-1]["low"] < self.stop_loss:
                self.win = False
                self.result = -1
                should_close = True
            elif df.iloc[-1]["high"] > self.take_profit:
                self.win = True
                self.result = self.point_to_take_profit / self.point_to_stoploss
                should_close = True
        else:
            if df.iloc[-1]["high"] > self.stop_loss:
                self.win = False
                self.result = -1
                should_close = True
            elif df.iloc[-1]["low"] < self.take_profit:
                self.win = True
                self.result = self.point_to_take_profit / self.point_to_stoploss
                should_close = True
        
        if should_close:
            self.close()  # 状態遷移をトリガー
            print(f" {self.name}: クローズ条件成立 {self.stop_loss}/{self.take_profit}")

    def handle_closed(self, df, sml_df):
        print(f"{self.name}: Entered 'くろーず' state.")
        self.destroy_reqest = True

    # on_enter 系の処理
    def record_state_time(self):
        state_name = self.state
        self.state_times[state_name] = current_df.iloc[-1]["time"]

    def on_enter_created_new_arrow(self):
        self.record_state_time()
        pvts = (self.start_pivot, self.new_arrow_pivot)
        if self.up_trend is True:
            self.base_fibo70, _, self.base_fibo37, _ = detect_extension_reversal(pvts, lower1_percent=0.3, higher1_percent=-0.37)
        else:
            self.base_fibo37, _, self.base_fibo70, _ = detect_extension_reversal(pvts, lower1_percent=0.37, higher1_percent=-0.3)

    def on_enter_infibos(self):
        self.record_state_time()

    def on_enter_has_position(self):
        self.record_state_time()
        print(f"エンターhas_position__名前：{self.name}, エントリーライン：{self.entry_line}, テイクプロフィット：{self.take_profit}, ストップロス：{self.stop_loss}, エントリーピボット：{self.entry_pivot}, PTS to SL：{self.point_to_stoploss}, PTS to TP：{self.point_to_take_profit}, ステート：{self.state},ステート時間：{self.state_times}")

    def on_enter_closed(self):
        self.record_state_time()
        print(f"{self.name}: Entered 'くろーず' state.")

    def __repr__(self):
        return f"MyModel(name={self.name}, state={self.state})"

    # ユーティリティ関数群
    def get_high_and_low_in_term(self, df, time_val):
        if df.index.name != "time":
            df = df.set_index("time", drop=False)
        required_df = df[time_val:].copy()
        highest_price = required_df[["open", "high", "low", "close"]].max().max()
        lowest_price = required_df[['open', 'high', 'low', 'close']].min().min()
        highest_close = required_df["close"].max()
        lowest_close = required_df["close"].min()
        return highest_price, lowest_price, highest_close, lowest_close

    def get_golden_cross_time(self, df, sml_df):
        df_indexed = df.copy().set_index("time", drop=False)
        sml_df_indexed = sml_df.copy().set_index("time", drop=False)
        base_sma_since_new_arrow = df_indexed.loc[self.new_arrow_pivot[0]:].copy()
        sml_sma_since_new_arrow = sml_df_indexed.loc[self.new_arrow_pivot[0]:].copy()
        if self.up_trend is True:
            for i in range(len(base_sma_since_new_arrow)):
                if base_sma_since_new_arrow.iloc[i]["BASE_SMA"] > sml_sma_since_new_arrow.iloc[i]["SML_SMA"]:
                    return base_sma_since_new_arrow.iloc[i]["time"] - timedelta(minutes=1)
            return df.iloc[-1]["time"]
        else:
            for i in range(len(base_sma_since_new_arrow)):
                if base_sma_since_new_arrow.iloc[i]["BASE_SMA"] < sml_sma_since_new_arrow.iloc[i]["SML_SMA"]:
                    return base_sma_since_new_arrow.iloc[i]["time"] - timedelta(minutes=1)
            return df.iloc[-1]["time"]

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
        if self.up_trend is True and neckline >= max(sma_values):
            return True
        elif not self.up_trend and neckline <= min(sma_values):
            return False
        else:
            return None

    def get_sml_pivots_after_goldencross(self, sml_pivots):
        for idx, pivot in enumerate(sml_pivots):
            if pivot[0] > self.time_of_goldencross - timedelta(minutes=1):
                self.sml_pivots_after_goldencross = sml_pivots[idx:]

    def get_potential_neck_wheninto_newarrow(self):
        sml_pvts = self.sml_pivots_after_goldencross
        if len(sml_pvts) >= 2 and self.up_trend is True:
            for i in range(1, len(sml_pvts)):
                if sml_pvts[i][2] == "high" and sml_pvts[i][0] > self.state_times["infibos"]:
                    if i + 1 < len(sml_pvts):
                        pvts = [sml_pvts[i], sml_pvts[i-1]]
                        fibo32 = detect_extension_reversal(pvts, -0.32, 0.32, None, None)
                        if watch_price_in_range(fibo32[0], fibo32[1], sml_pvts[i+1]):
                            self.determined_neck.append(sml_pvts[i])
                            self.organize_determined_neck()
                    else:
                        pvts = [sml_pvts[i], sml_pvts[i-1]]
                        fibo32 = detect_extension_reversal(pvts, -0.32, 0.32, None, None)
                        self.potential_neck.append(sml_pvts[i])
        if len(sml_pvts) >= 2 and not self.up_trend:
            for i in range(1, len(sml_pvts)):
                if sml_pvts[i][2] == "low" and sml_pvts[i][0] > self.state_times["infibos"]:
                    if i + 1 < len(sml_pvts):
                        pvts = [sml_pvts[i], sml_pvts[i-1]]
                        fibo32 = detect_extension_reversal(pvts, None, None, -0.32, 0.32)
                        if watch_price_in_range(fibo32[2], fibo32[3], sml_pvts[i+1]):
                            self.determined_neck.append(sml_pvts[i])
                            self.organize_determined_neck()
                    else:
                        pvts = [sml_pvts[i], sml_pvts[i-1]]
                        fibo32 = detect_extension_reversal(pvts, None, None, -0.32, 0.32)
                        self.potential_neck.append(sml_pvts[i])

    def check_potential_to_determine_neck(self):
        sml_pvts = self.sml_pivots_after_goldencross
        if self.potential_neck:
            pvts = [sml_pvts[-3], self.potential_neck[-1]]
            if self.up_trend:
                fibo32 = detect_extension_reversal(pvts, -0.32, 0.32, None, None)
                if watch_price_in_range(fibo32[0], fibo32[1], sml_pvts[-1][1]):
                    self.determined_neck.append(self.potential_neck[-1])
                    self.potential_neck.clear()
                    self.organize_determined_neck()
            else:
                fibo32 = detect_extension_reversal(pvts, None, None, -0.32, 0.32)
                if watch_price_in_range(fibo32[2], fibo32[3], sml_pvts[-1][1]):
                    self.determined_neck.append(self.potential_neck[-1])
                    self.potential_neck.clear()
                    self.organize_determined_neck()

    def potential_entry(self, df, neckline):
        if self.up_trend:
            if df.iloc[-1]["high"] > neckline[-1][1] and self.check_no_SMA(df.iloc[-1050:], neckline[-1][1]):
                self.stop_loss = self.highlow_since_new_arrow[1] - 0.006
                pivots_data = (self.start_pivot, self.new_arrow_pivot)
                highlow = detect_extension_reversal(pivots_data, higher1_percent=0.32)
                self.take_profit = highlow[2]
                self.entry_line = neckline[-1][1] + 0.002
                self.entry_pivot = df.iloc[-1]
                self.point_to_stoploss = abs(self.entry_line - self.stop_loss)
                self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                return True
            elif df.iloc[-1]["high"] > neckline[-1][1] and not self.check_no_SMA(df.iloc[-1050:], neckline[-1][1]):
                return False
        else:
            if df.iloc[-1]["low"] < neckline[-1][1] and not self.check_no_SMA(df.iloc[-1050:], neckline[-1][1]):
                self.stop_loss = self.highlow_since_new_arrow[0] + 0.006
                pivots_data = (self.start_pivot, self.new_arrow_pivot)
                highlow = detect_extension_reversal(pivots_data, lower1_percent=-0.32)
                self.take_profit = highlow[0]
                self.entry_line = neckline[-1][1] - 0.002
                self.entry_pivot = df.iloc[-1]
                self.point_to_stoploss = abs(self.entry_line - self.stop_loss)
                self.point_to_take_profit = abs(self.entry_line - self.take_profit)
                return True
            elif df.iloc[-1]["low"] < neckline[-1][1] and not self.check_no_SMA(df.iloc[-1050:], neckline[-1][1]):
                return False
        return None

    def organize_determined_neck(self):
        result = []
        for item in self.determined_neck:
            while result and item[1] > result[-1][1]:
                result.pop()
            result.append(item)
        self.determined_neck = result

    def price_in_range_while_adjustment(self, df):
        if self.up_trend:
            high = self.new_arrow_pivot[1]
            low = self.base_fibo70
            judged_price = self.get_high_and_low_in_term(df, self.new_arrow_pivot[0])
            if high < judged_price[0] or low > judged_price[1]:
                self.destroy_reqest = True
        else:
            high = self.base_fibo70
            low = self.new_arrow_pivot[1]
            judged_price = self.get_high_and_low_in_term(df, self.new_arrow_pivot[0])
            if high < judged_price[0] or low > judged_price[1]:
                self.destroy_reqest = True

#############################################
# セッションマネージャークラス
#############################################
class WaveManager(object):
    def __init__(self):
        self.sessions = {}  # セッションを session_id で管理
        self.next_session_id = 1
        self.trade_logs = []  # 各トレードログを記録するリスト

    def add_session(self, pivot_data, up_trend):
        session = MyModel(f"Session_{self.next_session_id}", pivot_data, up_trend)
        self.sessions[self.next_session_id] = session
        print(f"New session created: {session}, 時間:{pivot_data}, アプトレ:{session.up_trend}")
        self.next_session_id += 1
        return session

    def append_pivot_data(self, new_pivot_data, df, sml_df):
        sessions_to_delete = []
        for session_id, session in self.sessions.items():
            session.pivot_data.append(new_pivot_data)
            if session.state == "touched_20":
                session.new_arrow_pivot = new_pivot_data
                session.time_of_goldencross = session.get_golden_cross_time(df, sml_df)
                session.get_sml_pivots_after_goldencross(session.sml_pivot_data)
                session.get_potential_neck_wheninto_newarrow()
                session.create_new_arrow()
            elif session.state == "created_base_arrow":
                sessions_to_delete.append(session_id)
        for session_id in sessions_to_delete:
            del self.sessions[session_id]

    def append_sml_pivot_data(self, new_sml_pivot_data):
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
                if not session.potential_neck and session.up_trend and new_sml_pivot_data[2] == "high":
                    session.potential_neck.append(new_sml_pivot_data)
                elif not session.potential_neck and not session.up_trend and new_sml_pivot_data[2] == "low":
                    session.potential_neck.append(new_sml_pivot_data)

    def send_candle_data_tosession(self, df, sml_df):
        sessions_to_delete = []
        for session_id, session in self.sessions.items():
            # セッションが closed 状態ならトレードログに記録
            if session.state == "closed":
                # エントリー情報は session.entry_pivot, entry_line
                # 決済情報は、ここでは最新の current_df（df.tail(1)）から取得
                trade_log = {
                    "entry_time": session.entry_pivot["time"] if session.entry_pivot is not None else None,
                    "entry_price": session.entry_line if hasattr(session, "entry_line") else None,
                    "exit_time": df.iloc[-1]["time"],
                    "exit_price": df.iloc[-1]["close"],
                    "type": "high" if session.up_trend else "low",
                    "win": session.win if hasattr(session, "win") else None,
                    "result": session.result if hasattr(session, "result") else None
                }
                self.trade_logs.append(trade_log)
                # 内部確認用（f-string内はシングルクォートを使用）
                print(f"Trade log recorded for {session.name}: {trade_log['entry_time'].__class__}")
            session.execute_state_action(df, sml_df)
            print(f"Session {session_id} ステート：{session.state}、カレント:{current_df}") if session.name == "Session_40" else None
            if session.destroy_reqest:
                sessions_to_delete.append(session_id)
                print(f"Session {session_id} デストロイリクエスト{session.destroy_reqest}、ステート：{session.state}")
        for session_id in sessions_to_delete:
            print(f"Session {session_id} 削除、最終的にちゃんと実行.、ステート：{session.state}、デストロイリクエスト{session.destroy_reqest}")
            del self.sessions[session_id]

    def check_in_range(self):
        avoid_state = ("created_base_arrow", "build_position", "position_reached161", "position_reached200")
        sessions_to_delete = []
        for session_id, session in self.sessions.items():
            if session.state not in avoid_state:
                if session.up_trend:
                    result = watch_price_in_range(session.pivot_data[1], session.high150)
                    if result is False:
                        sessions_to_delete.append(session_id)
                else:
                    result = watch_price_in_range(session.pivot_data[0], session.low150)
                    if result is False:
                        sessions_to_delete.append(session_id)
        for session_id in sessions_to_delete:
            del self.sessions[session_id]

    def export_trade_logs_to_csv(self, filename="trade_logs.csv"):
        total_trades = len(self.trade_logs)
        wins = sum(1 for trade in self.trade_logs if trade.get("win") is True)
        win_rate = wins / total_trades if total_trades > 0 else 0
        total_profit = sum(trade.get("result", 0) for trade in self.trade_logs if trade.get("result", 0) > 0)
        total_loss = -sum(trade.get("result", 0) for trade in self.trade_logs if trade.get("result", 0) < 0)
        profit_factor = total_profit / total_loss if total_loss > 0 else None

        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["entry_time", "entry_price", "exit_time", "exit_price", "type", "win", "risk_reward"])
            for trade in self.trade_logs:
                entry_time_str = (trade["entry_time"].strftime("%Y-%m-%d %H:%M:%S")
                                  if hasattr(trade["entry_time"], "strftime") else trade["entry_time"])
                exit_time_str = (trade["exit_time"].strftime("%Y-%m-%d %H:%M:%S")
                                 if trade.get("exit_time") and hasattr(trade["exit_time"], "strftime") else trade.get("exit_time", ""))
                writer.writerow([
                    entry_time_str,
                    trade.get("entry_price", ""),
                    exit_time_str,
                    trade.get("exit_price", ""),
                    trade.get("type", ""),
                    trade.get("win", ""),
                    trade.get("result", "")
                ])
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
    if not mt5.initialize():
        print("MT5の初期化に失敗しました")
        return False
    return True

def shutdown_mt5():
    mt5.shutdown()

def fetch_data_range(symbol, from_date, to_date, timeframe=mt5.TIMEFRAME_M1):
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

def detect_extension_reversal(pivot_data, lower1_percent=None, lower2_percent=None, higher1_percent=None, higher2_percent=None):
    if len(pivot_data) < 2:
        return (None, None)
    price1 = pivot_data[-2][1]
    price2 = pivot_data[-1][1]
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

def watch_price_in_range(low, high, judged_price=current_price_global):
    low = min(low, high)
    high = max(low, high)
    if low <= judged_price <= high:
        return True
    else:
        return False

def save_fibonacci_to_csv(fib_data, filename="fibs.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_start", "time_end", "price_lower", "price_upper"])
        for f in fib_data:
            time_start = f[0].strftime('%Y-%m-%d %H:%M') if hasattr(f[0], 'strftime') else f[0]
            time_end = f[1].strftime('%Y-%m-%d %H:%M') if hasattr(f[1], 'strftime') else f[1]
            writer.writerow([time_start, time_end, f[2], f[3]])

def calculate_sma(df, window=20, name="SMA?"):
    df = df.copy()
    df[name] = df['close'].rolling(window=window).mean()
    df[name] = df[name].ffill()
    df = df.reset_index(drop=True)
    return df

def update_sma(df, window, name):
    df = df.copy()
    if len(df) < window:
        sma_value = df['close'].mean()
    else:
        sma_value = df['close'].iloc[-window:].mean()
    df.loc[df.index[-1], name] = sma_value
    return df

def determine_trend(df, name):
    up_down_list = [False]
    for i in range(1, len(df)):
        up_down_list.append(df[name][i] > df[name][i-1])
    df["UP_DOWN"] = up_down_list
    return df

def update_determine_trend(df, name):
    if df.loc[df.index[-2], name] < df.loc[df.index[-1], name]:
        df.loc[df.index[-1], "UP_DOWN"] = True
    elif df.loc[df.index[-2], name] > df.loc[df.index[-1], name]:
        df.loc[df.index[-1], "UP_DOWN"] = False
    else:
        df.loc[df.index[-1], "UP_DOWN"] = None
    return df

def detect_pivots(df, name, POINT_THRESHOLD=0.01, LOOKBACK_BARS=15, consecutive_bars=3, arrow_spacing=10):
    last_pivot_index = -999
    up_trend = False
    prev_h_or_l_index = None
    pivot_data = []
    pivot_index = None
    minimum_gap = 2
    up_down_list = df["UP_DOWN"]
    for i in range(3, len(df)):
        three_up = up_down_list[i]
        three_down = (not up_down_list[i])
        if three_down and up_trend == True:
            if last_pivot_index is not None and (i - last_pivot_index) < minimum_gap:
                continue
            pivot_index = i
            last_pivot_index = i
            sma_slice = df[name][pivot_index-LOOKBACK_BARS:pivot_index+1]
            sma_highest = sma_slice.max()
            current_sma = df[name][pivot_index]
            if (sma_highest - current_sma) >= POINT_THRESHOLD:
                sma_highest_index = sma_slice.idxmax()
                hs = (df['high'][prev_h_or_l_index:pivot_index+1] 
                      if prev_h_or_l_index is not None 
                      else df['high'][pivot_index-LOOKBACK_BARS:pivot_index+1])
                highest_index = hs.idxmax()
                highest = hs.max()
                highest_datetime = df["time"][highest_index]
                pivot_data.append((highest_datetime, highest, "high"))
                last_pivot_index = pivot_index
                up_trend = False
                prev_h_or_l_index = sma_highest_index
        if three_up and up_trend == False:
            if last_pivot_index is not None and (i - last_pivot_index) < minimum_gap:
                continue
            pivot_index = i
            last_pivot_index = i
            if pivot_index - LOOKBACK_BARS < 0:
                continue
            sma_slice = df[name][pivot_index-LOOKBACK_BARS:pivot_index+1]
            sma_lowest = sma_slice.min()
            current_sma = df[name][pivot_index]
            if current_sma - sma_lowest >= POINT_THRESHOLD:
                sma_lowest_index = sma_slice.idxmin()
                ls = (df['low'][prev_h_or_l_index:pivot_index+1] 
                      if prev_h_or_l_index is not None 
                      else df['low'][pivot_index-LOOKBACK_BARS:pivot_index+1])
                lowest_index = ls.idxmin()
                lowest = ls.min()
                lowest_datetime = df["time"][lowest_index]
                pivot_data.append((lowest_datetime, lowest, "low"))
                last_pivot_index = pivot_index
                up_trend = True
                prev_h_or_l_index = sma_lowest_index
    return pivot_data

def update_detect_pivot(df, name, point_threshold, lookback_bars, consecutive_bars, arrow_spacing, window=1000):
    subset_df = df.iloc[-window:].copy().reset_index(drop=True)
    up_down_list = subset_df["UP_DOWN"].tolist()
    pivots = detect_pivots(subset_df, name, POINT_THRESHOLD=point_threshold,
                            LOOKBACK_BARS=lookback_bars, consecutive_bars=consecutive_bars, arrow_spacing=arrow_spacing)
    if pivots:
        last_pivot = pivots[-1]
        pivot_time, pivot_price, pivot_type = last_pivot
        idx = df.index[df["time"] == pivot_time]
        if len(idx) > 0:
            df.loc[idx[0], "Pivot"] = True if pivot_type == "high" else False
        return last_pivot
    return None

def save_pivots_to_csv(pivot_data, filename="pivots.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "price", "type"])
        for row in pivot_data:
            time_str = row[0].strftime('%Y-%m-%d %H:%M') if hasattr(row[0], 'strftime') else row[0]
            writer.writerow([time_str, row[1], row[2]])

import pytz

def process_data(symbol="USDJPY"):
    global last_pivot_data, sml_last_pivot_data, current_price_global, current_df

    pivot_data = []
    sml_pivot_data = []

    if not initialize_mt5():
        return

    print("実行中")
    timezone = pytz.timezone("Etc/UTC")
    fromdate = datetime(2025, 2, 4, 15, 0, tzinfo=timezone)
    todate   = datetime(2025, 2, 7, 6, 50, tzinfo=timezone)

    original_df = fetch_data_range(symbol, fromdate, todate)
    if original_df is None:
        shutdown_mt5()
        return

    wm = WaveManager()

    df = original_df.iloc[:1600].copy()
    sml_df = original_df.iloc[:1600].copy()

    df = calculate_sma(df.copy(), window=20, name="BASE_SMA")
    sml_df = calculate_sma(sml_df.copy(), window=4, name="SML_SMA")

    determine_trend(df, "BASE_SMA")
    determine_trend(sml_df, "SML_SMA")

    pivot_data = detect_pivots(df.copy(), POINT_THRESHOLD=0.008, LOOKBACK_BARS=15, name="BASE_SMA", arrow_spacing=8)
    sml_pivot_data = detect_pivots(sml_df.copy(), POINT_THRESHOLD=0.001, LOOKBACK_BARS=3, consecutive_bars=1, name="SML_SMA", arrow_spacing=1)

    last_pivot_data = pivot_data[-1]
    sml_last_pivot_data = sml_pivot_data[-1]

    print(f"開始時間：{df.iloc[-1]['time']}")
    
    for idx in range(1600, len(original_df)):
        new_row = original_df.iloc[idx:idx+1]
        df = pd.concat([df, new_row], ignore_index=True)
        sml_df = pd.concat([sml_df, new_row], ignore_index=True)
        df = calculate_sma(df, window=20, name="BASE_SMA")
        sml_df = calculate_sma(sml_df, window=4, name="SML_SMA")
        update_determine_trend(df, "BASE_SMA")
        update_determine_trend(sml_df, "SML_SMA")
        new_pivot = update_detect_pivot(df, point_threshold=0.009, lookback_bars=15, consecutive_bars=3, arrow_spacing=8, name="BASE_SMA")
        if new_pivot is not None and new_pivot != last_pivot_data:
            last_pivot_data = new_pivot
            pivot_data.append(new_pivot)
            wm.append_pivot_data(last_pivot_data, df, sml_df)
            if new_pivot[2] == "high":
                wm.add_session(pivot_data[-2:], up_trend="False")
            else:
                wm.add_session(pivot_data[-2:], up_trend="True")
        sml_new_pivot = update_detect_pivot(sml_df, point_threshold=0.003, lookback_bars=3, consecutive_bars=1, name="SML_SMA", arrow_spacing=1)
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
    wm.export_trade_logs_to_csv(filename="trade_logs.csv")
        
if __name__ == "__main__":
    process_data()
