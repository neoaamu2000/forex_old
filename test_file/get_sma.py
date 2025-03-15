from utils.テスト import calculate_sma, determine_trend,merge_all_timeframes


if __name__ == "__main__":
    # conditions = {
    #     "symbol" = ["USDJPY"]
    # }
    final_df = merge_all_timeframes()
    print("pkl書き出し中")
    final_df.reset_index(drop=True).to_pickle("USDJPY_data会.pkl")
    print("pkl書き出し完了")