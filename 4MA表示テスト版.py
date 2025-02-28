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

@app.route('/pivots', methods=['GET'])
def get_pivots():
    """
    ピボットポイントをCSV形式で返すエンドポイント
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["time", "price", "type"])  # ヘッダー
    for p in pivot_data_global:
        writer.writerow([p[0].strftime('%Y-%m-%d %H:%M'), p[1], p[2]])
    return Response(output.getvalue(), mimetype="text/csv")

@app.route('/fibs', methods=['GET'])
def get_fibs():
    """
    フィボナッチ用の矩形情報をCSV形式で返すエンドポイント
    time_start, time_end, price_lower, price_upper
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["time_start", "time_end", "price_lower", "price_upper"])  # ヘッダー
    for f in fib_data_global:
        writer.writerow([
            f[0].strftime('%Y-%m-%d %H:%M'),
            f[1].strftime('%Y-%m-%d %H:%M'),
            f[2],
            f[3]
        ])
    return Response(output.getvalue(), mimetype="text/csv")


def detect_pivots(df, up_down_list, name, point_threshold=0.01, lookback_bars=15, consecutive_bars=3, arrow_spacing=10):
    """
    25SMAとその上昇/下降情報から、トレンド転換による高値・安値（ピボット）を検出する。
    連続して上昇／下降したバーの数（consecutive_bars）をパラメーターで設定可能。
    
    Args:
        df (DataFrame): 時系列データ。'time', 'close', 'high', 'low', 'sma25'列を含む。
        up_down_list (list): SMAの上昇/下降を表すブールリスト。
        name (str): SMAの列名。ここでは"sma25"など。
        point_threshold (float): トレンド転換とみなすためのSMA差の閾値。
        lookback_bars (int): 過去何本分のデータでチェックするか。
        consecutive_bars (int): 上昇または下降とみなす連続バーの数。
        arrow_spacing (int): ピボット間の最小間隔。
        
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
                pivot_data.append((highest_datetime, highest, "high"))
                last_pivot_index = pivot_index
                up_trend = False
                prev_h_or_l_index = sma_slice.idxmax()

        # ----- 下降→上昇（安値形成）の検出 -----
        if consecutive_up and not up_trend:
            pivot_index = i
            if pivot_index - last_pivot_index < arrow_spacing or pivot_index - lookback_bars < 0:
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
                pivot_data.append((lowest_datetime, lowest, "low"))
                last_pivot_index = pivot_index
                up_trend = True
                prev_h_or_l_index = sma_slice.idxmin()

    return pivot_data


def main():
    """
    MT5からUSDJPYの過去データを取得し、
    25SMA（ここでは4MAとして計算）を用いてトレンド転換（高値形成／安値形成）を検出する。
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

    # ===== 2. データ取得 =====
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        print("データが取得できませんでした")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    last_200_rates_global = df.iloc[-200:]
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # ===== 3. 4MAの計算 =====
    # ※ここでは「sma25」として計算（window=4）
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

    # ===== 5. ピボット（高値／安値）の検出 =====
    # 以下のパラメーターは4MA用に調整（必要に応じて変更してください）
    # ここでは、consecutive_bars=1 として各バー単体での判定を行い、
    # arrow_spacing を2 に設定
    pivot_data = detect_pivots(
        df, up_down_list, name="sma25",
        point_threshold=0.001, 
        lookback_bars=5,
        consecutive_bars=1,
        arrow_spacing=2
    )

    # ===== 検出結果をCSV出力（デバッグ用） =====
    with open("pivots.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "price", "type"])
        for row in pivot_data:
            writer.writerow([row[0].strftime('%Y-%m-%d %H:%M'), row[1], row[2]])

    # グローバル更新
    pivot_data_global = pivot_data[:]

    # ===== 6. フィボナッチ矩形の計算 =====
    fib_data = []
    for i in range(1, len(pivot_data)):
        prev_p = pivot_data[i-1]  # (dt, price, type)
        curr_p = pivot_data[i]
        price_diff = abs(curr_p[1] - prev_p[1])
        if prev_p[1] > curr_p[1]:  # 下降トレンド（high → low）
            fib31 = curr_p[1] + price_diff * 0.37
            fib62 = curr_p[1] + price_diff * 0.64
        else:  # 上昇トレンド（low → high）
            fib31 = prev_p[1] + price_diff * 0.37
            fib62 = prev_p[1] + price_diff * 0.64
        dt_start = prev_p[0]
        dt_end   = curr_p[0]
        fib_data.append((dt_start, dt_end, fib31, fib62))

    # グローバル更新
    fib_data_global = fib_data[:]

    # ===== フィボナッチ矩形のCSV出力（デバッグ用） =====
    with open("fibs.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_start", "time_end", "price_lower", "price_upper"])
        for f in fib_data:
            writer.writerow([f[0], f[1], f[2], f[3]])

    mt5.shutdown()


if __name__ == "__main__":
    # Flaskサーバーを別スレッドで起動
    flask_thread = threading.Thread(target=app.run, kwargs={'host': "127.0.0.1", 'port': 5000})
    flask_thread.daemon = True
    flask_thread.start()

    # 定期実行ループ：10秒ごとにメイン処理を再実行
    while True:
        main()
        time.sleep(10)
