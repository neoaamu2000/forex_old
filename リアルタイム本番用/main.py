# main.py

import time
import MetaTrader5 as mt5
from datetime import datetime
import threading

from login import initialize_mt5, shutdown_mt5
from SMA_highlow_functions import calculate_sma, determine_trend, detect_pivots, save_pivots_to_csv, pivot_data_global
from server import app
from manage_data import fetch_data
from session import WaveManager as Wavemanager


# セッション管理用のグローバル WaveManager インスタンス


wave_manager_global = Wavemanager()

symbol="USDJPY"

last_pivot_data = 999
sml_last_pivot_data = 999
last_price_minute = 999



def process_data(symbol="USDJPY"):

    global last_pivot_data, last_price_minute, wave_manager_global, sml_last_pivot_data

    if not initialize_mt5():
        return
    
    print("実行中")

    df = fetch_data(symbol,n_bars=250)
    minute = datetime.now().minute
    if df is None:
        shutdown_mt5()
        return

    # 1. SMAを計算しSMAの行をdfに追加する
    df = calculate_sma(df, window=20, name = "BASE_SMA")
    sml_df = calculate_sma(df, window=4, name = "SML_SMA")

    # 2. 新しいSMAがその直前のSMAに対して上昇/下降しているかを判定
    up_down_list = determine_trend(df['BASE_SMA'])
    sml_up_down_list = determine_trend(sml_df['SML_SMA'])

    # 3. 20MAのピボット検出
    pivot_data = detect_pivots(df, up_down_list, point_threshold=0.01, lookback_bars=15, name = "BASE_SMA",arrow_spacing=10)
    if pivot_data[-1] != last_pivot_data:   
        last_pivot_data = pivot_data[-1]
        wave_manager_global.append_pivot_data(last_pivot_data,df.iloc[-1])
        if pivot_data[-1][2] == "high":#下向きの矢印（down trendになりうる可能性のある波の始まり）
            wave_manager_global.add_session(pivot_data[-2:],up_trend = "False")
        else: #上向きの矢印（up trendになりうる可能性のある波の始まり）
            wave_manager_global.add_session(pivot_data[-2:],up_trend = "True")

    # 4. 4MAのピボット検出
    sml_pivot_data = detect_pivots(
                                sml_df, sml_up_down_list, 
                                point_threshold=0.005, 
                                lookback_bars=5,
                                consecutive_bars=1,
                                name="SML_SMA",arrow_spacing=10)
    if sml_pivot_data[-1] != sml_last_pivot_data:
        sml_last_pivot_data = sml_pivot_data[-1]
        wave_manager_global.append_sml_pivot_data(sml_last_pivot_data,sml_df.iloc[-1])



    # 1分足が確定下タイミングで最新のdfの情報をセッションマネージャーに送りsend_candle_data_tosessionで各セッションの更新を確認
    if last_price_minute != minute:
        wave_manager_global.check_in_range()
        wave_manager_global.send_candle_data_tosession(df.iloc[-100:], sml_df.iloc[-100:])
        last_price_minute = minute

    # 3.1 ピボットデータをCSVに保存（デバッグ用）
    save_pivots_to_csv(pivot_data, filename="pivots.csv")
    save_pivots_to_csv(sml_pivot_data, filename="sml_pivots.csv")
    

    shutdown_mt5()

def should_check():
    """現在の秒数がチェック対象の範囲内か判定する。"""
    # 現在の秒を取得
    sec = datetime.now().second
    # 例えば、58秒〜59秒、もしくは0秒〜3秒の間をチェック対象にする
    return sec >= 54 or sec <= 2

def run_server():
    """
    Flaskサーバーを起動する。
    """
    app.run(host="127.0.0.1", port=5000)

if __name__ == "__main__":
    # Flaskサーバーを別スレッドで起動
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    # #起動時１回限り多めにデータ取得しておいてpivot_data等ある程度保有
    # process_data()

    # 定期実行ループ：10秒ごとに process_data() を実行
    while True:
        # if should_check():
            # チェック対象の期間内なら1秒ごとに実行する
            # ここに新しい足の確認処理や process_data() の呼び出しを記述
            process_data()
            time.sleep(10)  # 1秒ごとにチェック
        # else:
        #     # チェック対象外の期間は負荷を下げるため、少し長めにスリープ
        #     time.sleep(5)


# # main.py
# import time
# import threading
# import pandas as pd
# import MetaTrader5 as mt5

# import datetime as dt

# from login import initialize_mt5, shutdown_mt5
# from pivots import calculate_sma, determine_trend, detect_pivots, save_pivots_to_csv
# from fibonacci import calculate_fibonacci_rectangles, save_fibonacci_to_csv
# from server import app, pivot_data_global, fib_data_global

# def fetch_data(symbol="USDJPY", timeframe=mt5.TIMEFRAME_M1, n_bars=999):
#     """
#     MT5から過去データを取得し、DataFrameとして返す。
    
#     Args:
#         symbol (str): 通貨ペア（デフォルト "USDJPY"）。
#         timeframe: MT5のタイムフレーム（例: mt5.TIMEFRAME_M1）。
#         n_bars (int): 取得するバー数。
        
#     Returns:
#         DataFrame または None
#     """
#     rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
#     if rates is None:
#         print("データが取得できませんでした")
#         return None
#     df = pd.DataFrame(rates)
#     df['time'] = pd.to_datetime(df['time'], unit='s')
#     return df

# # main.py の process_data() 内

# def process_data():
#     """
#     MT5からデータを取得し、SMA計算、ピボット検出、フィボナッチ矩形計算に加えて、
#     前回の調整波に対するフィボナッチエクステンション（120%）をチェックし、
#     そのレベルに触れた場合にトレンド転換と認識する処理を実行する。
#     結果はCSV出力（デバッグ用）とグローバル変数の更新を行う。
#     """
#     if not initialize_mt5():
#         return

#     df = fetch_data()
#     if df is None:
#         shutdown_mt5()
#         return

#     # 1. SMAの計算
#     df = calculate_sma(df, window=20)
    
#     # 2. SMAの上昇/下降判定
#     up_down_list = determine_trend(df['sma25'])
    
#     # 3. ピボット（高値・安値）の検出
#     pivot_data = detect_pivots(df, up_down_list, point_threshold=0.01, lookback_bars=15)
    
#     # 3.1 ピボットをCSVに保存（デバッグ用）
#     save_pivots_to_csv(pivot_data, filename="pivots.csv")
    
#     # 4. 現在の価格を最新の終値から取得
#     current_price = df['close'].iloc[-1]
    
#     # 5. 前回の調整波に対するエクステンションレベルをチェック
#     from fibonacci import detect_extension_reversal  # 必要に応じてインポート
#     reversal_signal, reversal_level = detect_extension_reversal(pivot_data)
#     if reversal_signal:
#         # トレンド転換と認識するため、現在のバーの時刻と計算レベルをピボットリストに追加する
#         reversal_time = df['time'].iloc[-1]
#         print(f"トレンド転換シグナル検出！ 時刻: {reversal_time}, 反転レベル: {reversal_level}")
#         pivot_data.append((reversal_time, reversal_level, "reversal"))
    
#     # グローバル変数の更新（Flaskサーバー用）
#     global pivot_data_global
#     pivot_data_global.clear()
#     pivot_data_global.extend(pivot_data)
    
#     # 6. フィボナッチ矩形の計算
#     fib_data = calculate_fibonacci_rectangles(pivot_data)
#     save_fibonacci_to_csv(fib_data, filename="fibs.csv")
    
#     global fib_data_global
#     fib_data_global.clear()
#     fib_data_global.extend(fib_data)
    
#     shutdown_mt5()




# def run_server():
#     """
#     Flaskサーバーを起動する。
#     """
#     app.run(host="127.0.0.1", port=5000)

# if __name__ == "__main__":
#     # Flaskサーバーを別スレッドで起動
#     server_thread = threading.Thread(target=run_server)
#     server_thread.daemon = True
#     server_thread.start()

#     # 定期実行ループ：10秒ごとにprocess_data()を実行
#     while True:
#         process_data()
#         time.sleep(10)
        




# # -*- coding: utf-8 -*-
# import csv
# import MetaTrader5 as mt5
# import pandas as pd
# import datetime as dt
# import time
# import threading
# import io
# from flask import Flask, Response

# # Flaskサーバーのセットアップ
# app = Flask(__name__)

# # グローバル変数：最新のピボット情報を保持（(datetime, price, type) のリスト）
# pivot_data_global = []

# # フィボナッチ矩形情報を保持する（(dt_start, dt_end, fib_lower, fib_upper) のリスト）
# fib_data_global = []

# @app.route('/pivots', methods=['GET'])
# def get_pivots():
#     """
#     ピボットポイントをCSV形式で返すエンドポイント
#     """
#     output = io.StringIO()
#     writer = csv.writer(output)
#     writer.writerow(["time", "price", "type"])  # ヘッダー
#     for p in pivot_data_global:
#         writer.writerow([p[0].strftime('%Y-%m-%d %H:%M'), p[1], p[2]])
#     return Response(output.getvalue(), mimetype="text/csv")

# @app.route('/fibs', methods=['GET'])
# def get_fibs():
#     """
#     フィボナッチ用の矩形情報をCSV形式で返すエンドポイント
#     time_start, time_end, price_lower, price_upper
#     """
#     output = io.StringIO()
#     writer = csv.writer(output)
#     writer.writerow(["time_start", "time_end", "price_lower", "price_upper"])  # ヘッダー
#     for f in fib_data_global:
#         writer.writerow([
#             f[0].strftime('%Y-%m-%d %H:%M'),
#             f[1].strftime('%Y-%m-%d %H:%M'),
#             f[2],
#             f[3]
#         ])
#     return Response(output.getvalue(), mimetype="text/csv")

# def main():
#     """
#     MT5からUSDJPYの過去データを取得し、
#     25SMAを用いてトレンド転換（高値形成／安値形成）を検出する。
#     その後、連続するpivot同士を使ってフィボナッチ(31%, 62%)の価格帯も計算。
#     """

#     # ===== 1. MT5接続 =====
#     if not mt5.initialize():
#         print("MT5の初期化に失敗しました")
#         return

#     symbol = "USDJPY"
#     timeframe = mt5.TIMEFRAME_M1
#     n_bars = 99999

#     # ===== 2. データ取得 =====
#     rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
#     if rates is None:
#         print("データが取得できませんでした")
#         mt5.shutdown()
#         return

#     df = pd.DataFrame(rates)
#     df['time'] = pd.to_datetime(df['time'], unit='s')

#     # ===== 3. 25SMAの計算 (元コードは20になっていましたがお好みで調整) =====
#     df['sma25'] = df['close'].rolling(window=20).mean()
#     df.dropna(inplace=True)
#     df.reset_index(drop=True, inplace=True)

#     # ===== 4. SMAの上昇・下降判定 =====
#     up_down_list = [False]  # 先頭は比較対象がないのでFalse
#     for i in range(1, len(df)):
#         if df['sma25'][i] > df['sma25'][i-1]:
#             up_down_list.append(True)
#         else:
#             up_down_list.append(False)

#     # ===== 5. トレンド転換の検出 =====
#     POINT_THRESHOLD = 0.01  # 10ポイント以上のSMA差
#     LOOKBACK_BARS = 15      # 過去15本内をチェック

#     last_pivot_index = -999
#     up_trend = False  # 初期は下降トレンドと仮定
#     sma_h_and_l = []
#     prev_h_or_l_index = None
#     pivot_data = []

#     for i in range(3, len(df)):
#         three_up = (up_down_list[i] and up_down_list[i-1] and up_down_list[i-2])
#         three_down = ((not up_down_list[i]) and (not up_down_list[i-1]) and (not up_down_list[i-2]))

#         # ----- 上昇→下降（高値形成）の検出 -----
#         if three_down and up_trend == True:
#             pivot_index = i
#             if pivot_index - last_pivot_index < 10:
#                 continue
#             if pivot_index - LOOKBACK_BARS < 0:
#                 continue

#             sma_slice = df['sma25'][pivot_index-LOOKBACK_BARS : pivot_index+1]
#             sma_highest = sma_slice.max()
#             current_sma = df['sma25'][pivot_index]

#             if (sma_highest - current_sma) >= POINT_THRESHOLD:
#                 date_str = df['time'][pivot_index].strftime('%Y-%m-%d %H:%M')
#                 close_price = df['close'][pivot_index]
#                 sma_highest_index = sma_slice.idxmax()
#                 sma_h_and_l.append(sma_highest_index)

#                 hs = df['high'][prev_h_or_l_index : pivot_index+1] if prev_h_or_l_index is not None else df['high'][pivot_index-LOOKBACK_BARS : pivot_index+1]
#                 highest_index = hs.idxmax()
#                 highest = hs.max()
#                 highest_datetime = df["time"][highest_index]

#                 print("=== 高値形成(上昇→下降) ===")
#                 print(f"日付: {date_str}")
#                 print(f"終値: {close_price}")
#                 print(f"25SMA: {current_sma}")
#                 print(f"SMA高値: {sma_highest}")
#                 print(f"高値: {highest}")
#                 print(f"時間: {highest_datetime}")
#                 print("-------------------------")
#                 pivot_data.append((highest_datetime, highest, "high"))

#                 last_pivot_index = pivot_index
#                 up_trend = False
#                 prev_h_or_l_index = sma_highest_index

#         # ----- 下降→上昇（安値形成）の検出 -----
#         if three_up and up_trend == False:
#             pivot_index = i
#             if pivot_index - last_pivot_index < 10:
#                 continue
#             if pivot_index - LOOKBACK_BARS < 0:
#                 continue

#             sma_slice = df['sma25'][pivot_index-LOOKBACK_BARS : pivot_index+1]
#             sma_lowest = sma_slice.min()
#             current_sma = df['sma25'][pivot_index]

#             if (current_sma - sma_lowest) >= POINT_THRESHOLD:
#                 date_str = df['time'][pivot_index].strftime('%Y-%m-%d %H:%M')
#                 close_price = df['close'][pivot_index]
#                 sma_lowest_index = sma_slice.idxmin()
#                 sma_h_and_l.append(sma_lowest_index)

#                 ls = df['low'][prev_h_or_l_index : pivot_index+1] if prev_h_or_l_index is not None else df['low'][pivot_index-LOOKBACK_BARS : pivot_index+1]
#                 lowest_index = ls.idxmin()
#                 lowest = ls.min()
#                 lowest_datetime = df["time"][lowest_index]

#                 print("=== 安値形成(下降→上昇) ===")
#                 print(f"日付: {date_str}")
#                 print(f"終値: {close_price}")
#                 print(f"25SMA: {current_sma}")
#                 print(f"SMA安値: {sma_lowest}")
#                 print(f"安値: {lowest}")
#                 print(f"時間: {lowest_datetime}")
#                 print("-------------------------")
#                 pivot_data.append((lowest_datetime, lowest, "low"))

#                 last_pivot_index = pivot_index
#                 up_trend = True
#                 prev_h_or_l_index = sma_lowest_index

#     # ===== ピボットをCSV出力（デバッグ用。EA側はWeb経由で取得） =====
#     with open("pivots.csv", mode="w", newline="", encoding="utf-8") as file:
#         writer = csv.writer(file)
#         writer.writerow(["time", "price", "type"])
#         for row in pivot_data:
#             writer.writerow(row)

#     # グローバル更新
#     global pivot_data_global
#     pivot_data_global = pivot_data[:]

#     # ===== 追加: フィボナッチ矩形の計算 =====
#     fib_data = []
#     for i in range(1, len(pivot_data)):
#         prev_p = pivot_data[i-1]  # (dt, price, type)
#         curr_p = pivot_data[i]
#         price_diff = abs(curr_p[1] - prev_p[1])  # 絶対値をとる
#         if prev_p[1] > curr_p[1]:  # 下降トレンド（high → low）
#             fib31 = curr_p[1] + price_diff * 0.37
#             fib62 = curr_p[1] + price_diff * 0.64
#         else:  # 上昇トレンド（low → high）
#             fib31 = prev_p[1] + price_diff * 0.37
#             fib62 = prev_p[1] + price_diff * 0.64
#         # 上下関係
#         # fib_lower = min(fib31, fib62)
#         # fib_upper = max(fib31, fib62)
#         # 時間の前後
#         dt_start = prev_p[0]
#         dt_end   = curr_p[0]
#         fib_data.append((dt_start, dt_end, fib31, fib62))
#         print(prev_p,curr_p,fib31,fib62)

#     # dt_start(Timestamp('2025-02-18 03:51:00'), np.float64(151.683), 'low') 
#     # dt_end(Timestamp('2025-02-18 04:39:00'), np.float64(151.854), 'high') 
#     # 151.73601 
#     # 151.78902

#     # グローバル更新
#     global fib_data_global
#     fib_data_global = fib_data[:]

#     # 一応CSV出力（デバッグ用）
#     with open("fibs.csv", mode="w", newline="", encoding="utf-8") as file:
#         writer = csv.writer(file)
#         writer.writerow(["time_start", "time_end", "price_lower", "price_upper"])
#         for f in fib_data:
#             writer.writerow([f[0], f[1], f[2], f[3]])

#     mt5.shutdown()

# if __name__ == "__main__":
#     # Flaskサーバーを別スレッドで起動
#     flask_thread = threading.Thread(target=app.run, kwargs={'host': "127.0.0.1", 'port': 5000})
#     flask_thread.daemon = True
#     flask_thread.start()

#     # 定期実行ループ：60秒ごとにメイン処理を再実行
#     while True:
#         main()
#         time.sleep(10)
