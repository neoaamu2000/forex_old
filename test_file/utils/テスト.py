import os 
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def calculate_sma(close_prices, window=20):
    sma = np.full_like(close_prices, np.nan)
    for i in range(window - 1, len(close_prices)):
        sma[i] = close_prices[i - window + 1:i + 1].mean()
    print("計算終わったよ")
    return sma

def determine_trend(sma):
    trend = np.zeros_like(sma)
    prev_valid = 0.0
    for i in range(1, len(sma)):
        if np.isnan(sma[i - 1]) or np.isnan(sma[i]):
            trend[i] = prev_valid
        else:
            trend[i] = 1.0 if sma[i - 1] < sma[i] else 0.0
            prev_valid = trend[i]
    return trend

def adjust_timestamp(df, timeframe):
    freq_map = {
        '1M': '1min', '5M': '5min', '15M': '15min',
        '1H': 'h', '4H': '4h', '1D': 'D'
    }
    df['time'] = df['time'].dt.floor(freq_map[timeframe])
    return df

def process_timeframe_file(args):
    tf_name, tf_info = args
    file_path = os.path.join(os.path.dirname(__file__), "..", "pickle_data", "USDJPY", tf_info["file"])
    df = pd.read_pickle(file_path)
    print("読み込みはできたよ")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = adjust_timestamp(df, tf_name)
    df = df.groupby('time').last().reset_index()

    close_prices = df["close"].to_numpy()

    for window in tf_info["sma"]:
        col_name = f"SMA_{window}"
        trend_col_name = f"SMA_{window}_trend"

        df[col_name] = np.nan
        df[trend_col_name] = np.nan

        if len(df) < window:
            print(f"警告: {tf_name} {window}SMA データ不足 ({len(df)}/{window})")
            continue

        df[col_name] = calculate_sma(close_prices, window)
        df[trend_col_name] = determine_trend(df[col_name])

    return tf_name, df

def merge_timeframe_to_base(base_df, tf_name, tf_df, tolerance):
    if tf_df.empty:
        print(f"警告: {tf_name}のDataFrameは空のためマージをスキップします。")
        return {}

    merged_cols = {}
    base_sorted = base_df.sort_values("time")

    sma_columns = [col for col in tf_df.columns if col.startswith('SMA_') and not col.endswith('_trend')]

    for col in sma_columns:
        trend_col = f"{col}_trend"
        tf_sub = tf_df[["time", col, trend_col]].copy().sort_values("time")

        tf_sub.rename(columns={
            col: f"{tf_name}_{col}",
            trend_col: f"{tf_name}_{trend_col}"
        }, inplace=True)

        merged = pd.merge_asof(
            base_sorted,
            tf_sub,
            on="time",
            direction="backward",
            tolerance=tolerance
        )

        merged_cols[f"{tf_name}_{col}"] = merged[f"{tf_name}_{col}"]
        merged_cols[f"{tf_name}_{col}_trend"] = merged[f"{tf_name}_{trend_col}"]

    return merged_cols

def merge_all_timeframes():
    base_path = os.path.join(os.path.dirname(__file__), "..", "pickle_data", "USDJPY", "USDJPY_1M会.pkl")
    base_df = pd.read_pickle(base_path)
    base_df["time"] = pd.to_datetime(base_df["time"], utc=True).dt.floor('1min')
    print("merge_all_dataで一番最初に1M読み込んだよ")

    timeframes = {
        '1M': {'file': 'USDJPY_1M.pkl', 'sma': [50, 100, 150, 200], 'tolerance': pd.Timedelta(minutes=90)},
        '5M': {'file': 'USDJPY_5M.pkl', 'sma': [75, 150, 200], 'tolerance': pd.Timedelta(minutes=90)},
        '15M': {'file': 'USDJPY_15M.pkl', 'sma': [100, 200], 'tolerance': pd.Timedelta(minutes=90)},
        '1H': {'file': 'USDJPY_1H.pkl', 'sma': [200], 'tolerance': pd.Timedelta(minutes=90)},
        '4H': {'file': 'USDJPY_4H.pkl', 'sma': [100, 200], 'tolerance': pd.Timedelta(hours=6)},
        '1D': {'file': 'USDJPY_1D.pkl', 'sma': [75, 130, 200], 'tolerance': pd.Timedelta(hours=26)},
    }

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {name: executor.submit(process_timeframe_file, (name, info)) for name, info in timeframes.items()}
        tf_results = {name: futures[name].result() for name in timeframes}
        for name, df_tf in tf_results.values():
            print(f"{name} 処理完了: {len(df_tf)}行")

    # マージ前に1Mデータから重複する列を除去する
    df_1M = tf_results['1M'][1].copy()
    duplicate_cols = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    df_1M_reduced = df_1M.drop(columns=duplicate_cols, errors='ignore')

    # 1Mデータの不要な部分を除いた状態でマージ
    base_df = pd.merge(base_df, df_1M_reduced, on='time', how='left')

    for tf_name in [n for n in timeframes if n != '1M']:
        merged_cols = merge_timeframe_to_base(
            base_df,
            tf_name,
            tf_results[tf_name][1],
            timeframes[tf_name]["tolerance"]
        )
        base_df = base_df.assign(**merged_cols)
    
    # 念のため不要な列をドロップ
    cols_to_drop = ["real_volume", "open_y", "high_y", "low_y", "close_y", "tick_volume_y", "spread_y", "real_volume_y"]
    base_df = base_df.drop(columns=cols_to_drop, errors='ignore')
    
    # dropnaしてインデックスリセットした後、直近100,000行だけを抽出
    base_df = base_df.dropna(subset=['1D_SMA_200']).reset_index(drop=True)
    return base_df


if __name__ == "__main__":
    df = merge_all_timeframes()
    print(df.tail())
    print(df.columns)
    # df.to_pickle("merged_data.pkl")
