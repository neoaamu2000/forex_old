# fibonacci_functions.py
from login import initialize_mt5, shutdown_mt5
import MetaTrader5 as mt5
import csv

from リアルタイム本番用.manage_data import current_price_global


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
        high1 = low_val - (-wave_range * higher1_percent)
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


# def calculate_fibonacci_rectangles(pivot_data,low_percentage=0.382, high_percentage=0.69):
#     """
#     連続するピボット間の価格差からフィボナッチ矩形（38.2%, 64%）の価格帯を計算する。
    
#     Args:
#         pivot_data (list): (datetime, price, type) のタプルのリスト。
        
#     Returns:
#         list: (dt_start, dt_end, fib31, fib62) のタプルのリスト。
#     """
#     fib_data = []
#     for i in range(1, len(pivot_data)):
#         prev_p = pivot_data[i-1]
#         curr_p = pivot_data[i]
#         price_diff = abs(curr_p[1] - prev_p[1])
#         if prev_p[1] > curr_p[1]:  # 下降トレンド（high → low）
#             fib_low = curr_p[1] + price_diff * low_percentage
#             fib_high = curr_p[1] + price_diff * high_percentage
#             way_of_wave  = "high to low"
#         else:  # 上昇トレンド（low → high）
#             fib_low = prev_p[1] + price_diff * low_percentage
#             fib_high = prev_p[1] + price_diff * high_percentage
#             way_of_wave  = "low to high"
#         dt_start = prev_p[0]
#         dt_end   = curr_p[0]
#         fib_data.append((dt_start, dt_end, fib_low, fib_high, way_of_wave))
#     return fib_data
