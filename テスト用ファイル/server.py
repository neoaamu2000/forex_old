# server.py
import csv
import io
from flask import Flask, Response

app = Flask(__name__)


# グローバル変数：各モジュールから更新される最新のデータを保持
pivot_data_global = []      # (datetime, price, type) のリスト
fib_data_global = []        # (time_start, time_end, price_lower, price_upper) のリスト

@app.route('/pivots', methods=['GET'])
def get_pivots():
    """
    ピボット情報をCSV形式で返すエンドポイント。
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["time", "price", "type"])
    for p in pivot_data_global:
        time_str = p[0].strftime('%Y-%m-%d %H:%M') if hasattr(p[0], 'strftime') else p[0]
        writer.writerow([time_str, p[1], p[2]])
    return Response(output.getvalue(), mimetype="text/csv")

@app.route('/fibs', methods=['GET'])
def get_fibs():
    """
    フィボナッチ矩形情報をCSV形式で返すエンドポイント。
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["time_start", "time_end", "price_lower", "price_upper"])
    for f in fib_data_global:
        time_start = f[0].strftime('%Y-%m-%d %H:%M') if hasattr(f[0], 'strftime') else f[0]
        time_end   = f[1].strftime('%Y-%m-%d %H:%M') if hasattr(f[1], 'strftime') else f[1]
        writer.writerow([time_start, time_end, f[2], f[3]])
    return Response(output.getvalue(), mimetype="text/csv")

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)
