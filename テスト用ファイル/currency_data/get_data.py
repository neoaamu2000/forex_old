# get_data.py

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
import pytz

# 保存先ディレクトリ
DATA_DIR = r"C:\Users\torti\Documents\currency_data"

# シンボルとタイムフレームの定義
symbols = ["XAUUSD", "EURUSD", "USDJPY", "GBPUSD", "USDCAD"]

# タイムフレームのマッピング：キーは任意の識別子、値はMT5の定数
timeframes = {
    "1M": mt5.TIMEFRAME_M1,
    "5M": mt5.TIMEFRAME_M5,
    "15M": mt5.TIMEFRAME_M15,
    "30M": mt5.TIMEFRAME_M30,
    "1H": mt5.TIMEFRAME_H1,
    "4H": mt5.TIMEFRAME_H4,
    "1D": mt5.TIMEFRAME_D1,
    "1Mth": mt5.TIMEFRAME_MN1  # 月足
}

# 取得開始日（2008年以降のデータを狙う）
start_date = datetime(2008, 1, 1, tzinfo=pytz.UTC)
# 取得終了日は現在の日付
end_date = datetime.now(pytz.UTC)

def get_data(symbol, timeframe, start_date, end_date):
    """
    MT5から指定のシンボル、タイムフレーム、期間のローソク足データを取得し、DataFrameで返す
    """
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"データが取得できませんでした: {symbol} {timeframe}")
        return None
    df = pd.DataFrame(rates)
    # MT5のtimeはUNIXタイムスタンプなので変換
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df

def save_data(df, symbol, timeframe_str):
    """
    DataFrameをCSVファイルとして保存する。
    ファイル名はシンボル_タイムフレーム.csv とする。
    """
    filename = os.path.join(DATA_DIR, f"{symbol}_{timeframe_str}.csv")
    df.to_csv(filename, index=False)
    print(f"保存完了: {filename}")

def main():
    # MT5の初期化
    if not mt5.initialize():
        print("MT5の初期化に失敗しました")
        return
    print("MT5初期化成功")

    # 各シンボル、各タイムフレームごとにデータ取得と保存を実施
    for symbol in symbols:
        for tf_str, tf_const in timeframes.items():
            print(f"取得中: {symbol} {tf_str}")
            df = get_data(symbol, tf_const, start_date, end_date)
            if df is not None:
                save_data(df, symbol, tf_str)
            else:
                print(f"{symbol} {tf_str}: データなし")
    mt5.shutdown()

if __name__ == "__main__":
    main()
