import time
import threading
import pandas as pd
import MetaTrader5 as mt5
from login import initialize_mt5, shutdown_mt5


global current_price_global
current_price_global = []



def fetch_data(symbol, timeframe=mt5.TIMEFRAME_M1, n_bars=200):
    """
    MT5から過去データを取得し、DataFrameとして返す。
    
    Args:
        symbol (str): 通貨ペア（デフォルト "USDJPY"）。
        timeframe: MT5のタイムフレーム（例: mt5.TIMEFRAME_M1）。
        n_bars (int): 取得するバー数。
        
    Returns:
        DataFrame または None
    """
    initialize_mt5()
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        print("データが取得できませんでした")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    shutdown_mt5()
    print(df)


if __name__ == "__main__":
    fetch_data(symbol = "USDJPY")