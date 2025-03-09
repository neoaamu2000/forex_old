# main.py

import time
import MetaTrader5 as mt5
from datetime import datetime
import threading
from transitions.extensions import HierarchicalMachine as Machine
import MetaTrader5 as mt5
import csv
import threading
import pandas as pd

from transitions.extensions import HierarchicalMachine as Machine


# セッション管理用のグローバル WaveManager インスタンス

current_price_global = []



symbol="USDJPY"

last_pivot_data = 999
sml_last_pivot_data = 999
last_price_minute = 999


    

def initialize_mt5():
    """
    MT5への接続を初期化する。
    接続に成功すればTrue、失敗すればFalseを返す。
    """
    if not mt5.initialize():
        print("MT5の初期化に失敗しました")
        return False
    return True

def shutdown_mt5():
    """
    MT5の接続をシャットダウンする。
    """
    mt5.shutdown()


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
    

# def update_current_price(symbol, interval=0.1):
#     """スレッド関数内で MT5 を初期化し、一定間隔でティック価格を取得して global に格納。"""
#     if not mt5.initialize():
#         print("MT5の初期化に失敗しました")
#         return
#     print("MT5 initialized in thread.")

#     while True:
#         tick = mt5.symbol_info_tick(symbol)
#         if tick is not None:
#             global current_price_global
#             current_price_global.clear()
#             current_price_global.append(tick)
#             print(f"Updated price: {current_price_global}")
#         else:
#             print("ティックデータ取得失敗（update_current_price）")
#         time.sleep(interval)






def detect_extension_reversal(pivot_data, lower1_percent=None, lower2_percent=None, higher1_percent=None, higher2_percent=None):
    """
    low1はフィボナッチあてる2点のうち低い方の価格を0として考える。
    high1はフィボナッチあてる2点のうち高い方の価格を0として考える。
    例えば150と160のフィボの場合、low1に-0.2を入れると152
    low2に0.4を入れると154、high1に-0.2を入れると158、high2に0.2を入れると162
    """    
    
    if len(pivot_data) < 2:
        return (None, None)
    
    # 前回と直近のピボットの価格を取り出す
    price1 = pivot_data[-2][1]
    price2 = pivot_data[-1][1]
    
    # 波の低い方と高い方を求める
    low_val = min(price1, price2)
    high_val = max(price1, price2)
    
    # 波幅の計算
    wave_range = high_val - low_val

    if lower1_percent is not None:
        low1 = low_val - (-wave_range * lower1_percent)
    else:
        low1 = None

    if higher1_percent is not None:
        high1 = high_val - (-wave_range * higher1_percent)
    else:
        high1 =None

    if lower2_percent is not None:
        low2 = low_val - (-wave_range * lower2_percent)
    else:
        low2 = None

    if higher2_percent is not None:
        high2 = high_val - (-wave_range * higher2_percent)
    else:
        high2 = None
    
    return (low1, low2, high1, high2)


def detect_small_reversal(base_p,end_adjustment_p):
    low_val = min(base_p, end_adjustment_p)
    high_val = max(base_p, end_adjustment_p)
    
def get_out_of_range(low, high):
    """
    セッションのトレンド開始を判断する20%のラインを突破したか確認する関数
    True → 上に設定した価格を上抜けたという合図
    True → 下に設定した価格を下抜けたという合図
    None → どちらの価格も突破せず、lowとhighの間にいるという合図
    """
    if current_price_global >= high:
        return True
    elif current_price_global <= low:
        return False
    else:
        return None
    
def check_touch_line(center_price, tested_price):
    if center_price <= tested_price:
        return True
    elif center_price >= tested_price:
        return False
    
        
def check_price_reached(price_to_judge):
    """
    ネックライン超えたか判断するための関数
    指定した価格より現在の価格が上にあるか下にあるかを返す関数
    True → 現在価格がprice_to_judgeよりも上にある
    False → 現在価格がprice_to_judgeよりも下にある
    """
    if current_price_global >= price_to_judge:
        return True
    elif current_price_global <= price_to_judge:
        return False



    
def watch_price_in_range(low,high,judged_price = current_price_global):
    low = min(low, high)
    high = max(low, high)
    if low <= judged_price <= high:
        return True
    else:
        return False
    

    



#------------------------------------------------------------------------------------
#直接はトレードに関係ない系

def save_fibonacci_to_csv(fib_data, filename="fibs.csv"):
    """
    フィボナッチ矩形情報をCSV形式で保存する（デバッグ用）。
    
    Args:
        fib_data (list): (time_start, time_end, price_lower, price_upper)のタプルのリスト。
        filename (str): 出力先ファイル名。
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_start", "time_end", "price_lower", "price_upper"])
        for f in fib_data:
            time_start = f[0].strftime('%Y-%m-%d %H:%M') if hasattr(f[0], 'strftime') else f[0]
            time_end   = f[1].strftime('%Y-%m-%d %H:%M') if hasattr(f[1], 'strftime') else f[1]
            writer.writerow([time_start, time_end, f[2], f[3]])

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
    df[name] = df[name].ffill()
    df.reset_index(drop=True, inplace=True)
    return df

def update_sma(df, window, name):
    """
    df の最後の行のみ、直近 window 本の 'close' 値から SMA を計算し、name カラムを更新する。
    """
    # もしデータ数が window 未満なら、全体平均を計算
    if len(df) < window:
        sma_value = df['close'].mean()
    else:
        sma_value = df['close'].iloc[-window:].mean()
    df.loc[df.index[-1], name] = sma_value
    return df

def determine_trend(df,name):
    """
    SMAの時系列データから、前の値と比較して上昇しているかを判定するリストを作成する。
    
    Args:
        sma_series (Series): SMAの時系列データ。
    
    Returns:
        list: 各インデックスでの上昇(True) or 非上昇(False)のリスト。
    """

    up_down_list = [False]  # 先頭は比較対象がないのでFalse
    for i in range(1, len(df)):
        up_down_list.append(df[name][i] > df[name][i-1])
    df["UP_DOWN"] = up_down_list #ここおおお
    return df

def update_determine_trend(df,name):
    if df.loc[df.index[-2], name] < df.loc[df.index[-1], name]:
        df.loc[df.index[-1], "UP_DOWN"] = True
    elif df.loc[df.index[-2], name] > df.loc[df.index[-1], name]:
        df.loc[df.index[-1], "UP_DOWN"] = False
    else:
        df.loc[df.index[-1], "UP_DOWN"] = None
    return df

def detect_pivots(df,name, POINT_THRESHOLD= 0.01, LOOKBACK_BARS=15, consecutive_bars=3,arrow_spacing = 10):
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
    POINT_THRESHOLD = 0.01  # 1ポイント以上のSMA差
    LOOKBACK_BARS = 5      # 過去5本内をチェック

    last_pivot_index = -999
    up_trend = False  # 初期は下降トレンドと仮定
    sma_h_and_l = []
    prev_h_or_l_index = None
    pivot_data = []
    pivot_index = None
    minimum_gap = 2

    up_down_list = df["UP_DOWN"]
    print(up_down_list)

    for i in range(3, len(df)):
        three_up = (up_down_list[i])
        three_down = ((not up_down_list[i]))

        # ----- 上昇→下降（高値形成）の検出 -----
        if three_down and up_trend == True:
            if last_pivot_index is not None and (i - last_pivot_index) < minimum_gap:
                continue 

            pivot_index = i
            last_pivot_index = i

            sma_slice = df[name][pivot_index-LOOKBACK_BARS : pivot_index+1]
            sma_highest = sma_slice.max()
            current_sma = df[name][pivot_index]

            if (sma_highest - current_sma) >= POINT_THRESHOLD:
                sma_highest_index = sma_slice.idxmax()#最高値のindexを入れる
                # sma_h_and_l.append(sma_highest_index)#sma_h_and_lに最高値のindexを追加

                hs = (df['high'][prev_h_or_l_index : pivot_index+1] 
                      if prev_h_or_l_index is not None 
                      else df['high'][pivot_index-LOOKBACK_BARS : pivot_index+1])
                highest_index = hs.idxmax()
                highest = hs.max()
                highest_datetime = df["time"][highest_index]
                pivot_data.append((highest_datetime, highest, "high"))
                last_pivot_index = pivot_index
                up_trend = False
                prev_h_or_l_index = sma_highest_index

        # ----- 下降→上昇（安値形成）の検出 -----
        if three_up and up_trend == False:
            if last_pivot_index is not None and (i - last_pivot_index) < minimum_gap:
                continue 

            pivot_index = i
            last_pivot_index = i

            if pivot_index - LOOKBACK_BARS < 0:
                continue

            sma_slice = df[name][pivot_index-LOOKBACK_BARS : pivot_index+1]
            sma_lowest = sma_slice.min()
            current_sma = df[name][pivot_index]

            if current_sma-sma_lowest >= POINT_THRESHOLD:
                sma_lowest_index = sma_slice.idxmin()
                # sma_h_and_l.append(sma_lowest_index)

                ls = df['low'][prev_h_or_l_index : pivot_index+1] if prev_h_or_l_index is not None else df['low'][pivot_index-LOOKBACK_BARS : pivot_index+1]
                lowest_index = ls.idxmin()
                lowest = ls.min()
                lowest_datetime = df["time"][lowest_index]

                pivot_data.append((lowest_datetime, lowest, "low"))

                last_pivot_index = pivot_index
                up_trend = True
                prev_h_or_l_index = sma_lowest_index
    
    return pivot_data



def update_detect_pivot(df, name, point_threshold, lookback_bars, consecutive_bars, arrow_spacing, window=50):
    """
    df の最後 window 行のみを対象にピボット検出を行い、
    最新のピボットイベントがあれば、元の df の該当行の "Pivot" カラムを更新する。
    戻り値は検出された最新のピボットイベント（タプル）または None。
    """
    # 最新の window 行をコピーして subset_df を作成
    subset_df = df.iloc[-window:].copy().reset_index(drop=True)
    # subset_df の UP_DOWN リストを取得
    up_down_list = subset_df["UP_DOWN"].tolist()
    # detect_pivots() は (pivot_data, updated_subset_df) を返すので、pivot_data を抽出
    pivots = detect_pivots(subset_df, name, point_threshold=point_threshold, 
                              lookback_bars=lookback_bars, consecutive_bars=consecutive_bars, 
                              arrow_spacing=arrow_spacing)
    if pivots:
        # 最新のピボットイベント
        last_pivot = pivots[-1]
        
        pivot_time, pivot_price, pivot_type = last_pivot
        # 元の df で pivot_time と一致する行のインデックスを取得
        idx = df.index[df["time"] == pivot_time]
        if len(idx) > 0:
            # "high" なら True、"low" なら False として記録（必要に応じて値を変更してください）
            df.loc[idx[0], "Pivot"] = True if pivot_type == "high" else False
        return last_pivot
    return None


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

import pytz



def process_data(symbol="USDJPY"):

    global last_pivot_data, last_price_minute, sml_last_pivot_data, current_price_global

    pivot_data =[]
    sml_pivot_data = []

    if not initialize_mt5():
        return
    
    print("実行中")
    timezone = pytz.timezone("Etc/UTC")
    fromdate = datetime(2024, 2, 17, 0, 0,tzinfo=timezone)
    todate   = datetime(2025, 2, 18, 7, 0,tzinfo=timezone)

    original_df = fetch_data_range(symbol,fromdate, todate)
    if original_df is None:
        shutdown_mt5()
        return
    

    # 1. SMAを計算しSMAの行をdfに追加する
    
    df = original_df.iloc[:2000].copy()
    
# 初期のSMA計算（この時点では200本分）







    df['SML_SMA'] = df['close'].rolling(window=4).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    determine_trend(df,name = "SML_SMA")
    

    pivot_data = detect_pivots(df,POINT_THRESHOLD= 0.005, LOOKBACK_BARS=5,consecutive_bars=1,name="SML_SMA", arrow_spacing=1)
        







    # print("=== Pivot Data ===")
    # for pivot in pivot_data:
    #     dt, price, ptype = pivot
    #     print(f"Time: {dt}, Price: {price}, Type: {ptype}")
        
if __name__ == "__main__":
    process_data()