import time
import timeit
import pandas as pd
import numpy as np

# Pickle読み込みの計測
start_read = time.time()
df = pd.read_pickle("currency_data/USDJPY_1M.pkl").reset_index()
sub_df = pd.read_pickle("currency_data/USDJPY_1M.pkl").reset_index()
sub_np = pd.read_pickle("currency_data/USDJPY_1M.pkl").reset_index().to_numpy()
end_read = time.time()

print(df)

def df_calculate_sma(df, window=20, name = "SMA?"):
    """
    DataFrameに対してSMA（単純移動平均）を計算し、'sma25'列として追加する。
    
    Args:
        df (DataFrame): 入力データ（少なくとも'close'列が必要）。
        window (int): 移動平均のウィンドウサイズ（デフォルト20）。
        
    Returns:
        DataFrame: SMA計算後のデータフレーム。
    """
    df = df.copy()  # 明示的なコピー作成
    df[name] = df['close'].rolling(window=window).mean()
    df[name] = df[name].ffill()
    df = df.reset_index(drop=True)
    return df

def calculate_sma(df, window=20, name = "SMA?"):
    """
    DataFrameに対してSMA（単純移動平均）を計算し、'sma25'列として追加する。
    
    Args:
        df (DataFrame): 入力データ（少なくとも'close'列が必要）。
        window (int): 移動平均のウィンドウサイズ（デフォルト20）。
        
    Returns:
        DataFrame: SMA計算後のデータフレーム。
    """
    df = df.copy()  # 明示的なコピー作成
    close = df["close"].to_numpy(dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    sma_valid = np.convolve(close, kernel, mode="valid")
    # 最初の window-1 行は計算できないので NaN で埋める
    sma = np.empty_like(close)
    sma[:window-1] = np.nan
    sma[window-1:] = sma_valid
    return sma
    # 移動平均を計算

def merge_sma_to_df(df, window=20, name = "BASE_SMA"):
    sma_numpy = calculate_sma(df, window, name)
    # 元の DataFrame のコピーに、新たな 'SMA_numpy' カラムとして追加
    df = df.copy()
    df[name] = sma_numpy
    return df

time_numpy = timeit.timeit('df_calculate_sma(sub_df, window=20, name="BASE_SMA")', globals=globals(), number=10)
print("dfのまま (10回):", time_numpy, "秒")

time_numpy = timeit.timeit('calculate_sma(df, window=20, name="BASE_SMA")', globals=globals(), number=10)
print("npのまま (10回):", time_numpy, "秒")

# n = 7_000_000

# # --- Approach 1: pandas の rolling() を利用 ---
# def calculate_sma_pandas(df, window=20):
#     # DataFrameのコピーを作ってから処理する（元データを保護）
#     df_local = df.copy()
#     # rolling() で移動平均を計算して ffill() で前方埋め
#     df_local["SMA"] = df_local["close"].rolling(window=window, min_periods=1).mean().ffill()
#     return df_local

# # --- Approach 2: NumPy を利用して SMA を計算し、後で DataFrame にマージ ---
# def calculate_sma_numpy(df, window=20):
#     # close列を NumPy 配列に変換（float64 で計算）
#     close = df["close"].to_numpy(dtype=np.float64)
#     print(df.to_numpy)
#     # 単純移動平均のカーネルを作成
#     kernel = np.ones(window, dtype=np.float64) / window
#     # np.convolve を使って移動平均を計算（mode='valid' で、ウィンドウ分のサイズが縮小）
#     sma_valid = np.convolve(close, kernel, mode="valid")
#     # 最初の window-1 行は計算できないので NaN で埋める
#     sma = np.empty_like(close)
#     sma[:window-1] = np.nan
#     sma[window-1:] = sma_valid
#     return sma

# # --- タイム計測 ---
# # それぞれ10回繰り返して計測（大規模データの場合は計測時間が長くなるので number を適宜調整）

# time_pandas = timeit.timeit("calculate_sma_pandas(df, window=20)", globals=globals(), number=10)
# print("Pandas rolling SMA (10 iterations):", time_pandas, "秒")

# time_numpy = timeit.timeit("calculate_sma_numpy(df, window=20)", globals=globals(), number=10)
# print("NumPy convolve SMA (10 iterations):", time_numpy, "秒")

# # --- さらに、NumPyで計算した結果を DataFrame にマージする例 ---
# def merge_sma_to_df(df, window=20):
#     sma_numpy = calculate_sma_numpy(df, window=window)
#     # 元の DataFrame のコピーに、新たな 'SMA_numpy' カラムとして追加
#     df_local = df.copy()
#     df_local["SMA_numpy"] = sma_numpy
#     return df_local

# time_merge = timeit.timeit("merge_sma_to_df(df, window=20)", globals=globals(), number=10)
# print("NumPy SMA と DataFrame マージ (10 iterations):", time_merge, "秒")
# print("Pickle読み込みにかかった時間:", end_read - start_read, "秒")

# import timeit
# time_loc = timeit.timeit("df['close'].rolling(window=20).mean()",globals=globals(), number=1)
# print(f"全期間のSMA処理時間 (1回): {time_loc} 秒")

# # DataFrameのインデックスに時間があるか確認
# print("Columns:", df.columns)
# print("Index:", df.index)

# # インデックスに時間がある場合、インデックスをカラムに戻してから処理
# df_reset = df.reset_index()  # これで 'index' というカラムに時間情報が入るので、必要なら rename する
# df_reset.rename(columns={"index": "time"}, inplace=True)

# # その後、.iloc の速度計測

# # .loc の速度計測：time カラムを使う
# # ここでは、すでに 'time' カラムが存在しているので、set_index は必要なくそのまま利用できる
# df_time_indexed = df_reset.set_index("time")
# time_loc = timeit.timeit("df_time_indexed.loc['2008-01-02 09:05:00+00:00':'2009-01-02 09:06:00+00:00']", 
#                          globals=globals(), number=1)
# print(f".loc の処理時間 (1回): {time_loc} 秒")

# time_iloc = timeit.timeit("df.iloc[1000:1080]", globals=globals(), number=1000000)
# print(f".iloc の処理時間 (1000000回): {time_iloc} 秒")

# time_iloc = timeit.timeit("np[1000:1080,:]", globals=globals(), number=1000000)
# print(f".iloc の処理時間 (1000000回): {time_iloc} 秒")


# sub_df = df.copy().to_numpy()
# time_iloc = timeit.timeit("df[1000:1100, :]", globals=globals(), number=1000000)
# print(f"Numpyの処理時間 (1000000回): {time_iloc} 秒")



