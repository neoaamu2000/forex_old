import pandas as pd

df = pd.read_csv("USDJPY_data.csv")
df.to_pickle("USDJPY_1M会.pkl")