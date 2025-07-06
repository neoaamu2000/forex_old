# main.py

import time
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import threading
# from transitions.extensions import HierarchicalMachine as Machine
import csv
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
import pytz

from utils.login import initialize_mt5, shutdown_mt5
from utils.data_utils import calculate_sma, determine_trend
from session_model import MyModel
from wave_manager import WaveManager

np.set_printoptions(threshold=np.inf)

# グローバル変数
current_price_global = {"high": None, "low": None}
symbol = "USDJPY"  # デフォルト値
last_pivot_data = 999
sml_last_pivot_data = 999

# # 状態定義
# states = [
#     "created_base_arrow",
#     "created_new_arrow",
#     "infibos",
#     "has_position",
#     "closed"
# ]

# # 遷移定義
# transitions = [
#     {"trigger": "create_new_arrow", "source": "created_base_arrow", "dest": "created_new_arrow"},
#     {"trigger": "touch_37", "source": "created_new_arrow", "dest": "infibos"},
#     {"trigger": "build_position", "source": "infibos", "dest": "has_position"},
#     {"trigger": "close", "source": "has_position", "dest": "closed"}
# ]







def detect_pivots(np_arr, time_df, name, conditions, POINT_THRESHOLD, LOOKBACK_BARS, consecutive_bars, arrow_spacing):
    if name == "BASE_SMA":
        wm = WaveManager()
    sma = np_arr[:, -2]
    trend_arr = np_arr[:, -1]
    pivot_data = []
    last_pivot_index = 0
    last_detect_index = 0
    run_counter = 1
    n = np_arr.shape[0]
    up_trend = None
    stop_strategy = conditions.get("stop", "sml")
    tp_level = conditions.get("tp_level", 138)
    for i in range(1, n):
        if trend_arr[i] == trend_arr[i-1]:
            run_counter += 1
        else:
            run_counter = 1
        if run_counter < consecutive_bars:
            continue
        if i - last_pivot_index < arrow_spacing:
            continue
        if up_trend is None:
            if trend_arr[i] == 0.0:
                up_trend = True
            elif trend_arr[i] == 1.0:
                up_trend = False
        if up_trend and (trend_arr[i] == 0.0):
            start = max(0, i - LOOKBACK_BARS)
            window_sma = sma[start: i+1]
            sma_max = np.nanmax(window_sma)
            if (sma_max - sma[i]) >= POINT_THRESHOLD:
                window_high = np_arr[last_pivot_index: i+1, 2]
                local_high_idx = np.argmax(window_high) + last_pivot_index if last_pivot_index >= 0 else start
                detection_time = np_arr[i, 0]
                pivot_time = np_arr[local_high_idx, 0]
                pivot_value = np_arr[local_high_idx, 2]
                pivot_type = 1.0
                pivot_data.append((detection_time, pivot_time, pivot_value, pivot_type))
                up_trend = False
                if name == "BASE_SMA" and last_pivot_index > 150 and n - i > 1500:
                    wm.add_session(start_index=local_high_idx, start_time_index=i, prev_index=last_pivot_index, prev_time_index=last_detect_index, up_trend="False", stop_strategy=stop_strategy, tp_level=tp_level)
                last_pivot_index = local_high_idx
                last_detect_index = i
        elif (not up_trend) and (trend_arr[i] == 1.0):
            start = max(0, i - LOOKBACK_BARS)
            window = sma[start: i+1]
            window_min = np.nanmin(window)
            if (sma[i] - window_min) >= POINT_THRESHOLD:
                window_min = np_arr[last_pivot_index: i+1, 3]
                local_min_idx = np.argmin(window_min) + last_pivot_index if last_pivot_index >= 0 else start
                detection_time = np_arr[i, 0]
                pivot_time = np_arr[local_min_idx, 0]
                pivot_value = np_arr[local_min_idx, 3]
                pivot_type = 0.0
                pivot_data.append((detection_time, pivot_time, pivot_value, pivot_type))
                up_trend = True
                if name == "BASE_SMA" and last_pivot_index > 100 and n - i > 1500:
                    wm.add_session(start_index=local_min_idx, start_time_index=i, prev_index=last_pivot_index, prev_time_index=last_detect_index, up_trend="True", stop_strategy=stop_strategy, tp_level=tp_level)
                last_pivot_index = local_min_idx
                last_detect_index = i
    if pivot_data:
        if name == "BASE_SMA":
            return wm, np.array(pivot_data, dtype=np.float64)
        else:
            return np.array(pivot_data, dtype=np.float64)
    else:
        return np.empty((0, 4), dtype=np.float64)

def merge_arr(base_arr, sml_arr, additional_sma_df):
    base_np = base_arr["sma_arr"]
    sml_sma_arr = sml_arr["sma_arr"]
    base_pivot_arr = base_arr["pivot_arr"]
    sml_pivot_arr = sml_arr["pivot_arr"]
    np_arr_with_base_sml_sma = np.column_stack((base_np, sml_sma_arr.reshape(-1, 1)))
    columns = ["time", "open", "high", "low", "close", "tick_volume", "spread", "BASE_SMA", "SML_SMA"]
    df = pd.DataFrame(np_arr_with_base_sml_sma, columns=columns)
    df["time"] = pd.to_datetime(df["time"], unit="ns", utc=True)
    pivot_columns = ["detection_time", "pivot_time", "pivot_value", "pivot_type"]
    df_pivot = pd.DataFrame(base_pivot_arr, columns=pivot_columns)
    df_pivot["detection_time"] = pd.to_datetime(df_pivot["detection_time"], unit="ns", utc=True)
    df_pivot["pivot_time"] = pd.to_datetime(df_pivot["pivot_time"], unit="ns", utc=True)
    sml_pivot_columns = ["sml_detection_time", "sml_pivot_time", "sml_pivot_value", "sml_pivot_type"]
    sml_df_pivot = pd.DataFrame(sml_pivot_arr, columns=sml_pivot_columns)
    sml_df_pivot["sml_detection_time"] = pd.to_datetime(sml_df_pivot["sml_detection_time"], unit="ns", utc=True)
    sml_df_pivot["sml_pivot_time"] = pd.to_datetime(sml_df_pivot["sml_pivot_time"], unit="ns", utc=True)
    df_sorted = df.sort_values("time")
    df_pivot_sorted = df_pivot.sort_values("detection_time")
    sml_df_pivot_sorted = sml_df_pivot.sort_values("sml_detection_time")
    merged_temp = pd.merge_asof(df_sorted, df_pivot_sorted,
                                left_on="time", right_on="detection_time",
                                direction="nearest", tolerance=pd.Timedelta("2sec"))
    merged_temp2 = pd.merge_asof(merged_temp, sml_df_pivot_sorted,
                                 left_on="time", right_on="sml_detection_time",
                                 direction="nearest", tolerance=pd.Timedelta("2sec"))
    merged_temp2.reset_index(drop=True, inplace=True)
    additional_sma_df.reset_index(drop=True, inplace=True)
    final_merged = pd.concat([merged_temp2, additional_sma_df], axis=1)
    final_merged = final_merged.drop(columns=["detection_time", "sml_detection_time"])
    print("最終出力データのマージが完了しました。")
    print(final_merged.head())
    return final_merged

def merge_all_results(final_df, merged_tf_df):
    base_columns = ["time", "open", "high", "low", "close", "tick_volume", "spread", "BASE_SMA", "SML_SMA"]
    extra_cols = merged_tf_df.drop(columns=base_columns)
    combined_df = pd.merge(final_df, extra_cols, on="time", how="left")
    return combined_df

def pre_data_process(np_arr, conditions, name, time_df):
    if name == "BASE_SMA":
        window = conditions.get("BASE_SMA", 20)
        point_threshold = conditions.get("BASE_threshold", 0)
        lookback_bars = conditions.get("BASE_lookback", 15)
        consecutive_bars = conditions.get("BASE_consecutive", 3)
        arrow_spacing = conditions.get("BASE_arrow_spacing", 8)
    elif name == "SML_SMA":
        window = conditions.get("SML_SMA", 4)
        point_threshold = conditions.get("SML_threshold", 0.002)
        lookback_bars = conditions.get("SML_lookback", 3)
        consecutive_bars = conditions.get("SML_consecutive", 1)
        arrow_spacing = conditions.get("SML_arrow_spacing", 1)
    sma_arr = calculate_sma(np_arr[:, 4], window=window)
    trend_array = determine_trend(sma_arr)
    np_arr = np.column_stack((np_arr, sma_arr.reshape(-1, 1), trend_array.reshape(-1, 1)))
    if name == "BASE_SMA":
        wm, pivot_arr = detect_pivots(np_arr, time_df, name, conditions,
                                       POINT_THRESHOLD=point_threshold,
                                       LOOKBACK_BARS=lookback_bars,
                                       consecutive_bars=consecutive_bars,
                                       arrow_spacing=arrow_spacing)
        np_arr = np_arr[:, :-1]
        return (name, np_arr, pivot_arr, wm)
    elif name == "SML_SMA":
        pivot_arr = detect_pivots(np_arr, time_df, name, conditions,
                                  POINT_THRESHOLD=point_threshold,
                                  LOOKBACK_BARS=lookback_bars,
                                  consecutive_bars=consecutive_bars,
                                  arrow_spacing=arrow_spacing)
        np_arr = np_arr[:, :-1]
        return (name, sma_arr, pivot_arr)

def process_data(conditions):
    global tp_level_global, check_no_SMA_global, range_param_global, stop_loss_global, time_df
    print("Current working directory:", os.getcwd())
    print(f"テスト開始時間 {datetime.now()}")
    symbol = conditions.get("symbol", "USDJPY")
    fromdate = conditions.get("fromdate", datetime(2023, 12, 1, 20, 0, tzinfo=pytz.UTC))
    todate = conditions.get("todate", datetime(2025, 2, 23, 6, 50, tzinfo=pytz.UTC))
    BASE_SMA = conditions.get("BASE_SMA", 20)
    BASE_threshold = conditions.get("BASE_threshold", 0.005)
    BASE_lookback = conditions.get("BASE_lookback", 15)
    BASE_consecutive = conditions.get("BASE_consecutive", 3)
    BASE_arrow_spacing = conditions.get("BASE_arrow_spacing", 8)
    SML_SMA = conditions.get("SML_SMA", 4)
    SML_threshold = conditions.get("SML_threshold", 0.005)
    SML_lookback = conditions.get("SML_lookback", 3)
    SML_consecutive = conditions.get("SML_consecutive", 1)
    SML_arrow_spacing = conditions.get("SML_arrow_spacing", 1)
    tp_level_global = conditions.get("tp_level", 138)
    check_no_SMA_global = conditions.get("check_no_sma", True)
    output_file = conditions.get("output_file", "trade_logs.csv")
    range_param_global = conditions.get("range", 80)
    stop_loss_global = conditions.get("stop", "sml")
    
    base_path = os.path.join("test_file", "pickle_data", symbol, "USDJPY_1M.pkl")
    origin_df = pd.read_pickle(base_path)
    origin_df = origin_df.loc[:, ~origin_df.columns.str.contains('^Unnamed')]
    origin_df["time"] = pd.to_datetime(origin_df["time"], utc=True)
    origin_df = origin_df.set_index("time")
    origin_df = origin_df.loc[fromdate:todate].reset_index()
    base_df = origin_df.iloc[:, :7]
    sma_df = origin_df.iloc[:, 7:]

    time_df = base_df["time"]
    np_arr = base_df.to_numpy(dtype=np.float64)
    base_result = pre_data_process(np_arr, conditions, "BASE_SMA", time_df)
    sml_result = pre_data_process(np_arr, conditions, "SML_SMA", time_df)
    results = [base_result, sml_result]
    result_dict = {}
    for result in results:
        if len(result) == 4:
            name, arr, pivot_arr, wm = result
            result_dict[name] = {"sma_arr": arr, "pivot_arr": pivot_arr, "wm": wm}
        else:
            name, arr, pivot_arr = result
            result_dict[name] = {"sma_arr": arr, "pivot_arr": pivot_arr}
    base_arr = result_dict.get("BASE_SMA")
    sml_arr = result_dict.get("SML_SMA")
    wm = base_arr.get("wm")
    final_df = merge_arr(base_arr, sml_arr, sma_df)
    wm.full_data = final_df.to_numpy(dtype=np.float64)
    
    print("データ整理完了しanalyze_sessions")
    wm.analyze_sessions()
    # risk_percentage を conditions から取得（例：3.0%）
    # risk_percentage = conditions.get("risk_percentage", 3.0)
    
    print(f"終了時間 {datetime.now()}")
    
    return wm.trade_logs
    
    

if __name__ == "__main__":
    conditions = {
        "symbol": "USDJPY",
        "fromdate": datetime(2014, 1, 14, 0, 0, tzinfo=pytz.UTC),
        "todate": datetime(2025, 2, 1, 0, 0, tzinfo=pytz.UTC),
        "BASE_SMA": 20,
        "BASE_threshold": 0.009,
        "BASE_lookback": 15,
        "BASE_consecutive": 3,
        "BASE_arrow_spacing": 8,
        "SML_SMA": 4,
        "SML_threshold": 0.003,
        "SML_lookback": 3,
        "SML_consecutive": 1,
        "SML_arrow_spacing": 2,
        "range": 80,
        "stop": "fibo",
        "tp_level": 138,
        "check_no_sma": True,
        "output_file": "test_result/USDJPY_138_trade_logs調整.csv",
        "risk_percentage": 3.0  # ここで1回あたりの損失許容額（%）を指定（例：3%）
    }
    start = time.time()
    process_data(conditions)
    end = time.time()
    time_diff = end - start
    print(time_diff)
