# import os
# import pandas as pd
# import numpy as np
# from concurrent.futures import ProcessPoolExecutor
# from datetime import datetime
# import pytz

# # ------------------------------
# # もともとの calculate_sma, determine_trend
# # ------------------------------
# def calculate_sma(np_arr, window=20):
#     """
#     NumPy配列の4列目（close値）からSMA（単純移動平均）を計算します。
#     入力行数がwindow未満の場合、全てNaNの配列を返します。
#     """
#     np_arr = np_arr.copy()
#     # ※データの列順が [index, time, open, high, low, close, ...] の場合、
#     # close 列は通常インデックス5ですが、ここは元コードの前提（closeが4番目）で計算しています。
#     close = np_arr[:, 4]
#     if len(close) < window:
#         return np.full_like(close, np.nan)
#     kernel = np.ones(window, dtype=np.float64) / window
#     sma_valid = np.convolve(close, kernel, mode="valid")
#     sma_arr = np.empty_like(close)
#     sma_arr[:window-1] = np.nan
#     sma_arr[window-1:] = sma_valid
#     print("しゅうりょう")
#     return sma_arr

# def determine_trend(sma):
#     """
#     SMA の時系列データから、前の値と比較して上昇しているかを判定するリストを作成する。
#     最初の値は 0.0（False）とし、後続は sma[:-1] < sma[1:] の結果（True:1.0, False:0.0）
#     """
#     trend_array = np.empty(sma.shape, dtype=np.float64)
#     trend_array[0] = 0.0
#     trend_array[1:] = sma[:-1] < sma[1:]
#     return trend_array

# # ------------------------------
# # 各タイムフレームのSMA計算結果を求め、そのDataFrameを返す関数（merge_asof用）
# # ------------------------------
# import os
# import pandas as pd
# import numpy as np
# from concurrent.futures import ProcessPoolExecutor
# from datetime import datetime
# import pytz

# # ------------------------------
# # SMA計算 & トレンド判定関数（改良版）
# # ------------------------------
# def calculate_sma(close_prices, window=20):
#     """
#     close値の配列からSMAを計算（NaN考慮版）
#     """
#     if len(close_prices) < window:
#         return np.full_like(close_prices, np.nan)
    
#     sma = np.full_like(close_prices, np.nan)
#     for i in range(window-1, len(close_prices)):
#         sma[i] = np.nanmean(close_prices[i-window+1:i+1])
#     return sma

import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import pytz

def determine_trend(sma):
    """
    NaNを考慮したトレンド判定（前方フィル付き）
    """
    trend = np.zeros_like(sma)
    prev_valid = 0.0
    for i in range(1, len(sma)):
        if np.isnan(sma[i-1]) or np.isnan(sma[i]):
            trend[i] = prev_valid
        else:
            trend[i] = 1.0 if sma[i-1] < sma[i] else 0.0
            prev_valid = trend[i]
    return trend



def calculate_sma(close_prices, window=20):
    weights = np.ones(window) / window
    sma_valid = np.convolve(close_prices, weights, mode='valid')
    sma = np.full_like(close_prices, np.nan, dtype=float)
    sma[window-1:] = sma_valid
    return sma

def adjust_timestamp(df, timeframe):
    """時間足の開始時刻に合わせるタイムスタンプ調整"""
    freq_map = {
        '1M': '1T',
        '5M': '5T',
        '15M': '15T',
        '1H': 'H',
        '4H': '4H',
        '1D': 'D'
    }
    df['time'] = df['time'].dt.floor(freq_map[timeframe])
    return df

def process_timeframe_file(args):
    tf_name, tf_info = args
    file_path = os.path.join(os.path.dirname(__file__), "..", "pickle_data", "USDJPY", tf_info["file"])
    df = pd.read_pickle(file_path)
    print("読み込みはできたよ")
    # タイムスタンプ調整（5分足なら5分間隔の開始時刻に）
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = adjust_timestamp(df, tf_name)
    
    # 重複排除（同じ時間足のデータが複数ある場合）
    df = df.groupby('time').last().reset_index()
    
    # SMA計算
    close_prices = df["close"].to_numpy()
    for window in tf_info["sma"]:
        col_name = f"SMA_{window}"
        df[col_name] = calculate_sma(close_prices, window)
        df[f"{col_name}_trend"] = determine_trend(df[col_name])
    print(f"SMA_{window}計算終わったよ")
    return (tf_name, df)

def merge_timeframe_to_base(base_df, tf_name, tf_df, tolerance):
    """時間足の開始時刻に合わせて前方参照しないマージ"""
    merged_cols = {}
    base_sorted = base_df.sort_values("time")
    
    for col in tf_df.columns:
        if "SMA_" in col and "_trend" not in col:
            tf_sub = tf_df[["time", col, f"{col}_trend"]].copy()
            tf_sub = tf_sub.sort_values("time")
            
            # 次の時間足の開始時刻まで有効になるようにマージ
            merged = pd.merge_asof(
                base_sorted,
                tf_sub,
                on="time",
                direction="forward",
                tolerance=tolerance
            )
            
            merged_cols[f"{tf_name}_{col}"] = merged[col]
            merged_cols[f"{tf_name}_{col}_trend"] = merged[f"{col}_trend"]
    
    return merged_cols

def merge_all_timeframes():
    # 1分足データの読み込み（ベースデータ）
    base_path = os.path.join(os.path.dirname(__file__), "..", "pickle_data", "USDJPY", "USDJPY_1M.pkl")
    base_df = pd.read_pickle(base_path)
    print("merge_all_dataで一番最初に1M読み込んだよ")
    base_df["time"] = pd.to_datetime(base_df["time"], utc=True)
    base_df = base_df.sort_values("time")
    
    # タイムフレーム設定（1分足を含む）
    timeframes = {
        '1M': {'file': 'USDJPY_1M.pkl', 'sma': [50, 100, 150, 200], 'tolerance': pd.Timedelta(minutes=2)},
        '5M': {'file': 'USDJPY_5M.pkl', 'sma': [75, 150, 200], 'tolerance': pd.Timedelta(minutes=4)},
        '15M': {'file': 'USDJPY_15M.pkl', 'sma': [100, 200], 'tolerance': pd.Timedelta(minutes=14)},
        '1H': {'file': 'USDJPY_1H.pkl', 'sma': [200], 'tolerance': pd.Timedelta(minutes = 59)},
        '4H': {'file': 'USDJPY_4H.pkl', 'sma': [100, 200], 'tolerance': pd.Timedelta(hours=3)},
        '1D': {'file': 'USDJPY_1D.pkl', 'sma': [75, 130, 200], 'tolerance': pd.Timedelta(hours=23)},
    }

    # マルチプロセス処理
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_timeframe_file, (name, info)) for name, info in timeframes.items()]
        tf_results = {name: future.result()[1] for name, future in zip(timeframes.keys(), futures)}
    print("計算は終わったよ")
    # ベースデータ（1分足）にマージ
    for tf_name in timeframes:
        # if tf_name == '1M':  # 1分足は別処理
        #     for col in tf_results['1M'].columns:
        #         if 'SMA_' in col:
        #             base_df[col] = tf_results['1M'][col]
        #     continue
            
        merged_cols = merge_timeframe_to_base(
            base_df,
            tf_name,
            tf_results[tf_name],
            timeframes[tf_name]["tolerance"]
        )
        base_df = base_df.assign(**merged_cols)
    
    # 5分足の例：17:00の5分足データは17:00～17:04の1分足に適用
    return base_df

if __name__ == "__main__":
    df = merge_all_timeframes()
    print(df[["time", "close", "5M_SMA_75", "1D_SMA_200"]].tail())