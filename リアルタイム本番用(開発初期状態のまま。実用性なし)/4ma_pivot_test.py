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
# グローバル変数：最新のピボット情報を保持（形式は (datetime, price, type) のリスト）
pivot_data_global = []

@app.route('/pivots', methods=['GET'])
def get_pivots():
    # CSV形式でピボット情報を返す
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["time", "price", "type"])  # ヘッダー
    for p in pivot_data_global:
        writer.writerow([p[0].strftime('%Y-%m-%d %H:%M'), p[1], p[2]])
    return Response(output.getvalue(), mimetype="text/csv")

def main():
    """
    MT5からUSDJPYの過去1300本のデータを取得し、
    25SMAを計算、SMAの連続上昇・下降をもとにトレンド転換（高値形成／安値形成）を検出します。
    
    トレンド転換の判定条件：
      - 直前までのトレンドが上昇（up_trend == True）の状態で、
        現在から3本連続が下降（three_down）→ 高値形成（上昇→下降）
      - 直前までのトレンドが下降（up_trend == False）の状態で、
        現在から3本連続が上昇（three_up）→ 安値形成（下降→上昇）
      ※ さらに、pivotから過去15本内でSMAの変動が10ポイント以上の場合としています。
    """
    
    # ===== 1. MT5接続 =====
    if not mt5.initialize():
        print("MT5の初期化に失敗しました")
        return

    symbol = "USDJPY"
    timeframe = mt5.TIMEFRAME_M1
    n_bars = 9999

    # ===== 2. データ取得 =====
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        print("データが取得できませんでした")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # ===== 3. 25SMAの計算 =====
    df['sma25'] = df['close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df[["open","close"]].head(30))  # 最初の30行を表示

    # ===== 4. SMAの上昇・下降判定 =====
    # up_down_list[i] = True なら、sma25が前のバーより上昇していると判定
    # 同じ値の場合は False（下降扱い）
    up_down_list = [False]  # 先頭は比較対象がないのでFalse
    for i in range(1, len(df)):
        if df['sma25'][i] > df['sma25'][i-1]:
            up_down_list.append(True)
        else:
            up_down_list.append(False)

    # ===== 5. トレンド転換の検出 =====
    POINT_THRESHOLD = 0.001  # 10ポイント以上のSMA差
    LOOKBACK_BARS = 4      # 過去15本内でチェック

    last_pivot_index = -999  
    up_trend = False  # 初期は下降トレンド（False）と仮定
    sma_h_and_l = []
    prev_h_or_l_index = None
    pivot_data = []
    
    for i in range(3, len(df)):
        three_up = (up_down_list[i] and up_down_list[i-1])
        three_down = ((not up_down_list[i]) and (not up_down_list[i-1]))
        
        # ----- 上昇→下降（高値形成）の検出 -----
        if three_down and up_trend == True:
            pivot_index = i
            if pivot_index - last_pivot_index < 10:
                continue
            if pivot_index - LOOKBACK_BARS < 0:
                continue
            
            sma_slice = df['sma25'][pivot_index-LOOKBACK_BARS : pivot_index+1]
            sma_highest = sma_slice.max()
            current_sma = df['sma25'][pivot_index]
            
            if (sma_highest - current_sma) >= POINT_THRESHOLD:
                date_str = df['time'][pivot_index].strftime('%Y-%m-%d %H:%M')
                close_price = df['close'][pivot_index]
                sma_highest_index = sma_slice.idxmax()
                sma_h_and_l.append(sma_highest_index)
                # prev_h_or_l_indexがNoneの場合はLOOKBACK_BARS分を使う
                hs = df['high'][prev_h_or_l_index : pivot_index+1] if prev_h_or_l_index is not None else df['high'][pivot_index-LOOKBACK_BARS : pivot_index+1]
                highest_index = hs.idxmax()
                highest = hs.max()
                highest_datetime = df["time"][highest_index]
                
                print("=== 高値形成(上昇→下降) ===")
                print(f"日付: {date_str}")
                print(f"終値: {close_price}")
                print(f"25SMA: {current_sma}")
                print(f"SMA高値: {sma_highest}")
                print(f"高値: {highest}")
                print(f"時間: {highest_datetime}")
                print("-------------------------")
                pivot_data.append((highest_datetime, highest, "high"))
                
                last_pivot_index = pivot_index
                up_trend = False  
                prev_h_or_l_index = sma_highest_index

        # ----- 下降→上昇（安値形成）の検出 -----
        if three_up and up_trend == False:
            pivot_index = i
            if pivot_index - last_pivot_index < 10:
                continue
            if pivot_index - LOOKBACK_BARS < 0:
                continue
            
            sma_slice = df['sma25'][pivot_index-LOOKBACK_BARS : pivot_index+1]
            sma_lowest = sma_slice.min()
            current_sma = df['sma25'][pivot_index]
            
            if (current_sma - sma_lowest) >= POINT_THRESHOLD:
                date_str = df['time'][pivot_index].strftime('%Y-%m-%d %H:%M')
                close_price = df['close'][pivot_index]
                sma_lowest_index = sma_slice.idxmin()
                sma_h_and_l.append(sma_lowest_index)
                ls = df['low'][prev_h_or_l_index : pivot_index+1] if prev_h_or_l_index is not None else df['low'][pivot_index-LOOKBACK_BARS : pivot_index+1]
                lowest_index = ls.idxmin()
                lowest = ls.min()
                lowest_datetime = df["time"][lowest_index]
                
                print("=== 安値形成(下降→上昇) ===")
                print(f"日付: {date_str}")
                print(f"終値: {close_price}")
                print(f"25SMA: {current_sma}")
                print(f"SMA安値: {sma_lowest}")
                print(f"安値: {lowest}")
                print(f"時間: {lowest_datetime}")
                print("-------------------------")
                pivot_data.append((lowest_datetime, lowest, "low"))
                
                last_pivot_index = pivot_index
                up_trend = True
                prev_h_or_l_index = sma_lowest_index

    with open("pivots.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "price", "type"])  # ヘッダー
        for row in pivot_data:
            writer.writerow(row)

    global pivot_data_global
    pivot_data_global = pivot_data[:]

    mt5.shutdown()

if __name__ == "__main__":
    # Flaskサーバーを別スレッドで起動（MT5 EA側は http://127.0.0.1:5000/pivots にアクセス）
    flask_thread = threading.Thread(target=app.run, kwargs={'host': "127.0.0.1", 'port': 5000})
    flask_thread.daemon = True
    flask_thread.start()
    
    # 定期実行ループ：30秒ごとにメイン処理を実行（リアルタイム更新＆CSV出力）
    while True:
        main()
        time.sleep(60)
