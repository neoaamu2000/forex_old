import datetime

# 与えられた値（ナノ秒）
timestamp_ns = 1.73985492e+18

# ナノ秒を秒に変換
timestamp_s = timestamp_ns / 1e9

# Unixエポックからの日時を取得 (UTC)
dt = datetime.datetime.utcfromtimestamp(timestamp_s)

print("変換結果:", dt)


d = [1,2,3,4,5,34,2,6,1]
print(len(d))

print(range(len(d)))