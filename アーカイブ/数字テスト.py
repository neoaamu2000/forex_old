# test_mt5_simulation.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# --- 必要な関数のダミー実装 ---
def calculate_sma(df, window, name="SMA"):
    df[name] = df['close'].rolling(window=window).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def determine_trend(sma_series):
    up_down_list = [False]
    for i in range(1, len(sma_series)):
        up_down_list.append(sma_series.iloc[i] > sma_series.iloc[i-1])
    return up_down_list

def detect_pivots(df, up_down_list, name, point_threshold=0.01, lookback_bars=15, consecutive_bars=3, arrow_spacing=10):
    # ダミー実装：直近の行をピボットとする
    pivot = (df['time'].iloc[-1], df['close'].iloc[-1], "high" if up_down_list[-1] else "low")
    return [pivot]

def save_pivots_to_csv(pivot_data, filename="pivots.csv"):
    print(f"Simulated saving {filename}: {pivot_data}")

def detect_extension_reversal(pivot_data, lower1_percent=None, lower2_percent=None, higher1_percent=None, higher2_percent=None):
    # ダミー実装：直近のピボット価格 ±0.2〜±0.4
    price = pivot_data[-1][1]
    return (price - 0.2, price - 0.4, price + 0.2, price + 0.4)

def check_touch_line(center_price, tested_price):
    return tested_price >= center_price

def watch_price_in_range(low, high, judged_price):
    low, high = min(low, high), max(low, high)
    return low <= judged_price <= high

def check_no_SMA(current_price):
    # 常に True と仮定（テスト用）
    return True

# ダミーのグローバル変数
current_price_global = []

# --- セッション管理用クラス群 ---
from transitions.extensions import HierarchicalMachine as Machine

states = [
    "created_base_arrow",
    "touched_20",
    "created_new_arrow",
    "infibos",
    "has_position"
]

transitions = [
    {"trigger": "touch_20", "source": "created_base_arrow", "dest": "touched_20"},
    {"trigger": "create_new_arrow", "source": "touched_20", "dest": "created_new_arrow"},
    {"trigger": "touch_37", "source": "created_new_arrow", "dest": "infibos"},
    {"trigger": "build_position", "source": "infibos", "dest": "has_position"},
]

class MyModel(object):
    def __init__(self, name, pivot_data, up_trend):
        self.name = name
        self.pivot_data = pivot_data[-2:]
        self.start_pivot = pivot_data[-1][0] if pivot_data else datetime.now()
        self.machine = Machine(model=self, states=states, transitions=transitions, initial="created_base_arrow")
        self.new_arrow_pivot = []  # 後で更新される
        self.base_fibo37 = None
        self.base_fibo70 = None
        self.time_of_goldencross = None
        self.highlow_since_new_arrow = None
        self.sml_pivot_data = []
        self.destroy_request = False
        self.up_trend = True if up_trend == "True" else False
        self.state_times = {}

        if self.up_trend:
            _, _, self.fibo_minus_20, self.fibo_minus_150 = detect_extension_reversal(self.pivot_data, None, None, 0.2, 1.5)
        else:
            self.fibo_minus_20, self.fibo_minus_150, _, _ = detect_extension_reversal(self.pivot_data, -0.2, -1.5, None, None)

        # 簡単な状態アクションは execute_state_action 内でログ出力とする
        self.state_actions = {}

    def execute_state_action(self, df, sml_df):
        print(f"{self.name}: Executing state action for state {self.state}")

    def touch_20(self):
        self.machine.touch_20()

    def create_new_arrow(self):
        self.machine.create_new_arrow()

    def touch_37(self):
        self.machine.touch_37()

    def build_position(self):
        self.machine.build_position()

    def destroy(self):
        print(f"{self.name}: Destroying session resources.")

    def get_high_and_low_in_term(self, df, time_value):
        if df.index.name != "time":
            df = df.set_index("time", drop=False)
        required_df = df[time_value:]
        highest_price = required_df[["open", "high", "low", "close"]].max().max()
        lowest_price = required_df[["open", "high", "low", "close"]].min().min()
        highest_close = required_df["close"].max()
        lowest_close = required_df["close"].min()
        return highest_price, lowest_price, highest_close, lowest_close

    def __repr__(self):
        return f"MyModel(name={self.name}, state={self.state})"

class WaveManager(object):
    def __init__(self):
        self.sessions = {}
        self.next_session_id = 1

    def add_session(self, pivot_data, up_trend):
        session = MyModel(f"Session_{self.next_session_id}", pivot_data, up_trend)
        self.sessions[self.next_session_id] = session
        print(f"New session created: {session}")
        self.next_session_id += 1
        return session

    def append_pivot_data(self, new_pivot_data, df):
        sessions_to_delete = []
        for session_id, session in self.sessions.items():
            session.pivot_data.append(new_pivot_data)
            if session.state == "touched_20":
                session.time_of_goldencross = session.start_pivot  # ダミー処理
                session.new_arrow_pivot = new_pivot_data
                session.create_new_arrow()
            elif session.state == "created_base_arrow":
                sessions_to_delete.append(session_id)
        for session_id in sessions_to_delete:
            self.delete_session(session_id)

    def delete_session(self, session_id):
        if session_id in self.sessions:
            self.sessions[session_id].destroy()
            del self.sessions[session_id]
            print(f"Session {session_id} deleted.")

    def append_sml_pivot_data(self, new_sml_pivot_data, sml_df):
        for session in self.sessions.values():
            session.sml_pivot_data.append(new_sml_pivot_data)

    def send_candle_data_tosession(self, df, sml_df):
        sessions_to_delete = []
        for session_id, session in self.sessions.items():
            session.execute_state_action(df, sml_df)
            if session.destroy_request:
                sessions_to_delete.append(session_id)
        for session_id in sessions_to_delete:
            self.delete_session(session_id)

    def check_in_range(self):
        # ダミー処理
        pass

    def __repr__(self):
        return f"WaveManager(sessions={list(self.sessions.values())})"

# --- MT5 データ取得 (指定期間のデータを取得) ---
def fetch_data_mt5(symbol, start_dt, end_dt, timeframe=mt5.TIMEFRAME_M1):
    # MT5 の初期化
    if not mt5.initialize():
        print("MT5 initialization failed")
        return None
    # 指定期間のデータを取得
    rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)
    if rates is None:
        print("No data retrieved")
        mt5.shutdown()
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    mt5.shutdown()
    return df

# --- シミュレーション処理 ---
def simulate_trading():
    # 期間設定：2025/02/18 02:00:00 ～ 2025/02/18 08:00:00
    start_dt = datetime(2025, 2, 18, 2, 0, 0)
    end_dt = datetime(2025, 2, 18, 8, 0, 0)
    
    df = fetch_data_mt5("USDJPY", start_dt, end_dt, timeframe=mt5.TIMEFRAME_M1)
    if df is None or df.empty:
        print("Failed to fetch historical data from MT5.")
        return
    
    # SMA計算
    df = calculate_sma(df, window=20, name="BASE_SMA")
    sml_df = calculate_sma(df.copy(), window=4, name="SML_SMA")
    
    # トレンド判定
    up_down_list = determine_trend(df["BASE_SMA"])
    sml_up_down_list = determine_trend(sml_df["SML_SMA"])
    
    # ピボット検出（ダミー実装）
    pivot_data = detect_pivots(df, up_down_list, name="BASE_SMA", point_threshold=0.01, lookback_bars=15, consecutive_bars=3, arrow_spacing=10)
    sml_pivot_data = detect_pivots(sml_df, sml_up_down_list, name="SML_SMA", point_threshold=0.005, lookback_bars=5, consecutive_bars=1, arrow_spacing=10)
    
    # WaveManager の生成
    wm = WaveManager()
    
    # セッション生成：最新のピボットデータに基づき、up_trend に応じてセッションを生成
    if pivot_data:
        last_pivot = pivot_data[-1]
        if last_pivot[2] == "high":
            wm.add_session(pivot_data[-2:], up_trend="False")
        else:
            wm.add_session(pivot_data[-2:], up_trend="True")
    
    # シミュレーション：指定期間のデータを順次処理（高速に過去チャートを再生）
    # ここでは、全データを10本ずつのスライスとして処理する例です
    for idx in range(100, len(df), 10):
        current_df = df.iloc[idx-100:idx+1]
        current_sml_df = sml_df.iloc[idx-100:idx+1]
        
        # 最新の close 価格を current_price_global に設定（テスト用）
        global current_price_global
        current_price_global.clear()
        current_price_global.append(current_df.iloc[-1]["close"])
        
        wm.send_candle_data_tosession(current_df, current_sml_df)
        time.sleep(0.05)  # テスト用に短いスリープ
        
    # シミュレーション終了後、各セッションの結果を出力
    print("\n--- Final Session States ---")
    for session_id, session in wm.sessions.items():
        print(f"Session {session_id}:")
        print(f"  Start pivot: {session.start_pivot}")
        print(f"  New arrow pivot: {session.new_arrow_pivot}")
        print(f"  Take profit: {getattr(session, 'take_profit', 'N/A')}")
        print(f"  Stop loss: {getattr(session, 'stop_loss', 'N/A')}")
        print(f"  State transitions: {session.state_times}")

if __name__ == "__main__":
    simulate_trading()
