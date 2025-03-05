import pandas as pd
import numpy as np
import timeit

# テスト用のダミーデータフレームを作成
n = 1000
df = pd.DataFrame({
    'close': np.random.rand(n) * 100
})
neckline = 50.0  # ダミーのneckline

def check_no_SMA(df, neckline):
    df = df.copy()
    periods = [25, 75, 100, 150, 200, 375, 625, 1000]
    sma_values = []
    for period in periods:
        if len(df) < period:
            continue
        sma_col = f"SMA_{period}"
        df.loc[:, sma_col] = df['close'].rolling(window=period).mean()
        sma_value = df[sma_col].iloc[-1]
        sma_values.append(sma_value)
    return (min(sma_values), max(sma_values)) if sma_values else None

# timeitで1000回実行した場合の時間を測定
t = timeit.timeit('check_no_SMA(df, neckline)', number=1000, globals=globals())
print(f"1000回の実行時間: {t}秒")

# 1000回あたりの平均実行時間（秒）
avg_time = t / 1000
print(f"1回あたりの実行時間: {avg_time}秒")

# 1億回の場合の推定時間
estimated_total = avg_time * 100_000_000
print(f"1億回実行した場合の推定時間: {estimated_total / 3600:.2f} 時間")
