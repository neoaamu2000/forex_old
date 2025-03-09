# -*- coding: utf-8 -*-
import csv
import MetaTrader5 as mt5
import pandas as pd
import datetime as dt
import time
import threading
import io
from flask import Flask, Response

# Flaskサーバーのセットアップ
app = Flask(__name__)

# グローバル変数：最新のピボット情報を保持（(datetime, price, type) のリスト）
global pivot_data_global
pivot_data_global = []

# フィボナッチ矩形情報を保持する（(dt_start, dt_end, fib_lower, fib_upper) のリスト）
global fib_data_global
fib_data_global = []

global last_200_rates_global
last_200_rates_global = []



def main():
    """
    MT5からUSDJPYの過去データを取得し、
    25SMAを用いてトレンド転換（高値形成／安値形成）を検出する。
    その後、連続するpivot同士を使ってフィボナッチ(31%, 62%)の価格帯も計算。
    """
    global last_200_rates_global
    global pivot_data_global
    global fib_data_global

    # ===== 1. MT5接続 =====
    if not mt5.initialize():
        print("MT5の初期化に失敗しました")
        return

    symbol = "USDJPY"
    timeframe = mt5.TIMEFRAME_M1
    n_bars = 99999

    import pytz
    from datetime import datetime
    timezone = pytz.timezone("Etc/UTC")
    fromdate = datetime(2025, 2, 17, 0, 0,tzinfo=timezone)
    todate   = datetime(2025, 2, 18, 7, 0,tzinfo=timezone)
    

    # ===== 2. データ取得 =====
    rates = mt5.copy_rates_range(symbol, timeframe, fromdate, todate)
    if rates is None:
        print("データが取得できませんでした")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    last_200_rates_global = df.iloc[-200:]
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # ===== 3. 25SMAの計算 (元コードは20になっていましたがお好みで調整) =====
    df['sma25'] = df['close'].rolling(window=4).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ===== 4. SMAの上昇・下降判定 =====
    up_down_list = [False]  # 先頭は比較対象がないのでFalse
    for i in range(1, len(df)):
        if df['sma25'][i] > df['sma25'][i-1]:
            up_down_list.append(True)
        else:
            up_down_list.append(False)

    # ===== 5. トレンド転換の検出 =====
    POINT_THRESHOLD = 0.001  # 1ポイント以上のSMA差
    LOOKBACK_BARS = 5      # 過去5本内をチェック

    last_pivot_index = -999
    up_trend = False  # 初期は下降トレンドと仮定
    sma_h_and_l = []
    prev_h_or_l_index = None
    pivot_data = []
    pivot_index = None
    minimum_gap = 2

    for i in range(3, len(df)):
        three_up = (up_down_list[i])
        three_down = ((not up_down_list[i]))

        # ----- 上昇→下降（高値形成）の検出 -----
        if three_down and up_trend == True:
            if last_pivot_index is not None and (i - last_pivot_index) < minimum_gap:
                continue 

            pivot_index = i
            last_pivot_index = i

            sma_slice = df['sma25'][pivot_index-LOOKBACK_BARS : pivot_index+1]
            sma_highest = sma_slice.max()
            current_sma = df['sma25'][pivot_index]

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

            sma_slice = df['sma25'][pivot_index-LOOKBACK_BARS : pivot_index+1]
            sma_lowest = sma_slice.min()
            current_sma = df['sma25'][pivot_index]

            if (current_sma - sma_lowest) >= POINT_THRESHOLD:
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

                # print("=== 安値形成(下降→上昇) ===")
                # print(f"日付: {date_str}")
                # print(f"終値: {close_price}")
                # print(f"25SMA: {current_sma}")
                # print(f"SMA安値: {sma_lowest}")
                # print(f"安値: {lowest}")
                # print(f"時間: {lowest_datetime}")
                # print("-------------------------")

    # ===== ピボットをCSV出力（デバッグ用。EA側はWeb経由で取得） =====
    with open("pivots.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "price", "type"])
        for row in pivot_data:
            writer.writerow(row)

    # グローバル更新
    
    pivot_data_global = pivot_data[:]

    # ===== 追加: フィボナッチ矩形の計算 =====
    fib_data = []
    for i in range(1, len(pivot_data)):
        prev_p = pivot_data[i-1]  # (dt, price, type)
        curr_p = pivot_data[i]
        price_diff = abs(curr_p[1] - prev_p[1])  # 絶対値をとる
        if prev_p[1] > curr_p[1]:  # 下降トレンド（high → low）
            fib31 = curr_p[1] + price_diff * 0.37
            fib62 = curr_p[1] + price_diff * 0.64
        else:  # 上昇トレンド（low → high）
            fib31 = prev_p[1] + price_diff * 0.37
            fib62 = prev_p[1] + price_diff * 0.64
        # 上下関係
        # fib_lower = min(fib31, fib62)
        # fib_upper = max(fib31, fib62)
        # 時間の前後
        dt_start = prev_p[0]
        dt_end   = curr_p[0]
        fib_data.append((dt_start, dt_end, fib31, fib62))
        # print(prev_p,curr_p,fib31,fib62)

    # dt_start(Timestamp('2025-02-18 03:51:00'), np.float64(151.683), 'low') 
    # dt_end(Timestamp('2025-02-18 04:39:00'), np.float64(151.854), 'high') 
    # 151.73601 
    # 151.78902

    # グローバル更新

    fib_data_global = fib_data[:]

    # 一応CSV出力（デバッグ用）
    with open("fibs.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_start", "time_end", "price_lower", "price_upper"])
        for f in fib_data:
            writer.writerow([f[0], f[1], f[2], f[3]])

    mt5.shutdown()

    print("=== Pivot Data ===")
    for pivot in pivot_data:
        # pivot は (datetime, price, type) のタプル
        dt, price, ptype = pivot
        print(f"Time: {dt}, Price: {price}, Type: {ptype}")

if __name__ == "__main__":

    # 定期実行ループ：60秒ごとにメイン処理を再実行
    while True:
        main()
        time.sleep(60)