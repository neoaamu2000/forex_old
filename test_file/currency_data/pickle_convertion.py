import os
import pandas as pd

# もとのCSVファイルがあるフォルダ
input_base_path = r'C:\Users\torti\Documents\EA\テスト用ファイル\currency_data'

# Pickleファイルを保存したいフォルダ
output_base_path = r'C:\Users\torti\Documents\EA\テスト用ファイル\pickle_data'

# フォルダが存在しなければ作成
os.makedirs(output_base_path, exist_ok=True)

# 通貨ペアフォルダを巡回
for currency_pair in os.listdir(input_base_path):
    currency_folder = os.path.join(input_base_path, currency_pair)

    # フォルダでなければスキップ
    if not os.path.isdir(currency_folder):
        continue

    # Pickle保存先に同じ名前の通貨ペアフォルダを作成
    output_currency_folder = os.path.join(output_base_path, currency_pair)
    os.makedirs(output_currency_folder, exist_ok=True)

    # 各CSVファイルを処理
    for csv_file in os.listdir(currency_folder):
        if csv_file.endswith('.csv'):
            input_csv_path = os.path.join(currency_folder, csv_file)

            # CSVをDataFrameとして読み込む
            df = pd.read_csv(input_csv_path)

            # Pickleファイル名（拡張子をpklに変更）
            pickle_file_name = csv_file.replace('.csv', '.pkl')
            output_pickle_path = os.path.join(output_currency_folder, pickle_file_name)

            # DataFrameをPickleとして保存
            df.to_pickle(output_pickle_path)

            print(f'変換完了: {input_csv_path} -> {output_pickle_path}')