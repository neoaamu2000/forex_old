from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import pytz
import time
from main import process_data, WaveManager

def split_periods(start_date, end_date, overlap=2000, splits=8):
    total_days = (end_date - start_date).days
    days_per_split = total_days // splits

    periods = []
    for i in range(splits):
        split_start = start_date + timedelta(days=i * days_per_split)
        split_end = split_start + timedelta(days=days_per_split)
        print(f"スタート{split_start}")
        print(f"エンド{split_end}")
        if i != 0:
            split_start -= timedelta(minutes=overlap)
        if i != splits - 1:
            split_end += timedelta(minutes=overlap)

        periods.append((split_start, split_end))
    return periods

def main_parallel():
    symbol = "USDJPY"
    start_date = datetime(2025, 1, 14, tzinfo=pytz.UTC)
    end_date = datetime(2025, 1, 20, tzinfo=pytz.UTC)

    periods = split_periods(start_date, end_date, overlap=2000, splits=1)
    conditions_list = [{
        "symbol": symbol,
        "fromdate": period[0],
        "todate": period[1],
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
        "stop": "sml",
        "tp_level": 138,
        "check_no_sma": True,
        "risk_percentage": 5.0,
    } for period in periods]

    all_trade_logs = []

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_data, cond) for cond in conditions_list]

        for future in futures:
            trade_logs_df = future.result()
            all_trade_logs.append(trade_logs_df)

    # 全ログを統合して重複削除（entry_timeを基準に重複を判定）
    combined_trade_logs = pd.concat(all_trade_logs, ignore_index=True)
    combined_trade_logs.drop_duplicates(subset=["entry_time"], inplace=True)
    combined_trade_logs.sort_values(by="entry_time", inplace=True)

    # 統計情報を計算
    wm_final = WaveManager()
    wm_final.trade_logs = combined_trade_logs
    wm_final.summarize_and_export_results(
        filename="final_combined_trade_logs.csv",
        initial_capital=10000,
        risk_percentage=conditions_list[0]["risk_percentage"]
    )

    end_time = time.time()
    print(f"全体の処理時間: {end_time - start_time:.2f}秒")

if __name__ == "__main__":
    main_parallel()
