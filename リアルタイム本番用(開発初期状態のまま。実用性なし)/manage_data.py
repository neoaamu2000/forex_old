#manage_data
 
import time
import threading
import pandas as pd
import MetaTrader5 as mt5
from login import initialize_mt5, shutdown_mt5


global current_price_global
current_price_global = []


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


# def fetch_data(symbol, timeframe=mt5.TIMEFRAME_M1, n_bars=200):
#     """
#     MT5から過去データを取得し、DataFrameとして返す。
    
#     Args:
#         symbol (str): 通貨ペア（デフォルト "USDJPY"）。
#         timeframe: MT5のタイムフレーム（例: mt5.TIMEFRAME_M1）。
#         n_bars (int): 取得するバー数。
        
#     Returns:
#         DataFrame または None
#     """
#     initialize_mt5()
#     rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
#     if rates is None:
#         print("データが取得できませんでした")
#         return None
#     df = pd.DataFrame(rates)
#     df['time'] = pd.to_datetime(df['time'], unit='s')
#     shutdown_mt5()
#     return df
    

def update_current_price(symbol, interval=0.1):
    """スレッド関数内で MT5 を初期化し、一定間隔でティック価格を取得して global に格納。"""
    if not mt5.initialize():
        print("MT5の初期化に失敗しました")
        return
    print("MT5 initialized in thread.")

    while True:
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            global current_price_global
            current_price_global.clear()
            current_price_global.append(tick)
            print(f"Updated price: {current_price_global}")
        else:
            print("ティックデータ取得失敗（update_current_price）")
        time.sleep(interval)

# def check_no_SMA(symbol = "USDJPY", n_bars=200):
#     """
#     指定シンボルについて、1分足と5分足の過去 n_bars 本の確定済みデータから、
#     SMA (close) の各期間（25,75,100,150,200）の最新値を計算し、
#     その中で最も低い値と最も高い値を返す。
    
#     Args:
#         symbol (str): 例 "USDJPY"
#         n_bars (int): 取得するバー数。デフォルトは200
        
#     Returns:
#         tuple: (lowest_SMA, highest_SMA)
#                もしデータが取得できない場合は None を返す。
#     """
#     periods = [25, 75, 100, 150, 200]
#     sma_values = []
    
        
#         # それぞれの SMA を計算する。ローリング平均は直近の確定足（in-progressの足は含まれない）
#         for period in periods:
#             if len(df) < period:
#                 continue  # 十分なデータがない場合はスキップ
#             sma_col = f"SMA_{period}"
#             df[sma_col] = df['close'].rolling(window=period).mean()
#             # dropna() しても良いが、最後の行は十分なデータがあればNaNにならないので、そのまま取得
#             sma_value = df[sma_col].iloc[-1]
#             sma_values.append(sma_value)
    
#     if current_price_global >= max(sma_values):
#         return True
    
#     if current_price_global <= min(sma_values):
#         return False
    
#     if not sma_values or min(sma_values) < current_price_global < max(sma_values):
#         return None


if __name__ == "__main__":
    # スレッドを起動
    price_thread = threading.Thread(
        target=update_current_price,
        kwargs={"symbol": "USDJPY", "interval": 0.1},
        daemon=True
    )
    price_thread.start()

    # メインスレッド側で別の処理を行う
    for _ in range(5):
        print(f"Main loop sees: {current_price_global}")
        time.sleep(1)
