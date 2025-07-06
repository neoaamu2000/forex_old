# SMA_highlow_functions.py
import pandas as pd
import csv


global pivot_data_global
pivot_data_global = []

def calculate_sma(df, window=20, name = "SMA?"):
    """
    DataFrameに対してSMA（単純移動平均）を計算し、'sma25'列として追加する。
    
    Args:
        df (DataFrame): 入力データ（少なくとも'close'列が必要）。
        window (int): 移動平均のウィンドウサイズ（デフォルト20）。
        
    Returns:
        DataFrame: SMA計算後のデータフレーム。
    """
    df[name] = df['close'].rolling(window=window).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def determine_trend(df,name):
    """
    SMAの時系列データから、前の値と比較して上昇しているかを判定するリストを作成する。
    
    Args:
        sma_series (Series): SMAの時系列データ。
    
    Returns:
        list: 各インデックスでの上昇(True) or 非上昇(False)のリスト。
    """
    sma_series = df[name]
    up_down_list = [False]  # 先頭は比較対象がないのでFalse
    for i in range(1, len(sma_series)):
        up_down_list.append(sma_series[i] > sma_series[i-1])
    return up_down_list

def detect_pivots(df, up_down_list, name, point_threshold=0.01, lookback_bars=15, consecutive_bars=3,arrow_spacing = 10):
    """
    25SMAとその上昇/下降情報から、トレンド転換による高値・安値（ピボット）を検出する。
    連続して上昇／下降したバーの数（consecutive_bars）をパラメーターで設定可能。
    
    Args:
        df (DataFrame): 時系列データ。'time', 'close', 'high', 'low', 'sma25'列を含む。
        up_down_list (list): SMAの上昇/下降を表すブールリスト。
        point_threshold (float): トレンド転換とみなすためのSMA差の閾値。
        lookback_bars (int): 過去何本分のデータでチェックするか。
        consecutive_bars (int): 上昇または下降とみなす連続バーの数（デフォルトは3）。
        
    Returns:
        list: (datetime, price, type) のタプルのリスト。typeは "high" もしくは "low"。
    """
    last_pivot_index = -999
    up_trend = False  # 初期は下降トレンドと仮定
    prev_h_or_l_index = None
    pivot_data = []

    # i は連続バー数 - 1 から開始する
    for i in range(consecutive_bars - 1, len(df)):
        # 連続した上昇・下降の判定（直近 consecutive_bars 本分）
        consecutive_up = all(up_down_list[j] for j in range(i - consecutive_bars + 1, i + 1))
        consecutive_down = all(not up_down_list[j] for j in range(i - consecutive_bars + 1, i + 1))

        # ----- 上昇→下降（高値形成）の検出 -----
        if consecutive_down and up_trend:
            pivot_index = i
            if pivot_index - last_pivot_index < arrow_spacing or pivot_index - lookback_bars < 0:
                continue

            sma_slice = df[name][pivot_index - lookback_bars : pivot_index + 1]
            sma_highest = sma_slice.max()
            current_sma = df[name][pivot_index]
            if (sma_highest - current_sma) >= point_threshold:
                hs = (df['high'][prev_h_or_l_index : pivot_index + 1]
                      if prev_h_or_l_index is not None
                      else df['high'][pivot_index - lookback_bars : pivot_index + 1])
                highest_index = hs.idxmax()
                highest = df['high'][highest_index]
                highest_datetime = df["time"][highest_index]
                pivot_data.append((highest_datetime, highest, "high"))#下向きの矢印（down trendになりうる可能性のある波の始まり）
                last_pivot_index = pivot_index
                up_trend = False
                prev_h_or_l_index = sma_slice.idxmax()

        # ----- 下降→上昇（安値形成）の検出 -----
        if consecutive_up and not up_trend:
            pivot_index = i
            if pivot_index - last_pivot_index < 10 or pivot_index - lookback_bars < 0:
                continue

            sma_slice = df[name][pivot_index - lookback_bars : pivot_index + 1]
            sma_lowest = sma_slice.min()
            current_sma = df[name][pivot_index]
            if (current_sma - sma_lowest) >= point_threshold:
                ls = (df['low'][prev_h_or_l_index : pivot_index + 1]
                      if prev_h_or_l_index is not None
                      else df['low'][pivot_index - lookback_bars : pivot_index + 1])
                lowest_index = ls.idxmin()
                lowest = df['low'][lowest_index]
                lowest_datetime = df["time"][lowest_index]
                pivot_data.append((lowest_datetime, lowest, "low"))#上向きの矢印（up trendになりうる可能性のある波の始まり）
                last_pivot_index = pivot_index
                up_trend = True
                prev_h_or_l_index = sma_slice.idxmin()

    return pivot_data


def save_pivots_to_csv(pivot_data, filename="pivots.csv"):
    """
    ピボット情報をCSV形式で保存する（デバッグ用）。
    
    Args:
        pivot_data (list): (datetime, price, type)のタプルのリスト。
        filename (str): 出力ファイル名。
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "price", "type"])
        for row in pivot_data:
            # datetimeはstrftimeできる場合のみ
            time_str = row[0].strftime('%Y-%m-%d %H:%M') if hasattr(row[0], 'strftime') else row[0]
            writer.writerow([time_str, row[1], row[2]])
