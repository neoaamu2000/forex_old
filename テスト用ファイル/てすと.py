import pandas as pd
import os




# # CSV から読み込む場合（例として time 列を日付型に変換しインデックスに設定）
# df = pd.read_csv("currency_data/USDJPY/USDJPY_1M.csv", parse_dates=["time"], index_col="time")

# # ピクル形式で保存
# df.to_pickle("USDJPY_1M.pkl")

df_loaded = pd.read_pickle("USDJPY_1M.pkl")
print(df_loaded)
print(os.getcwd())
