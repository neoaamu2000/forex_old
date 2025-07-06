import pandas as pd
import numpy as np
import csv
from session_model import MyModel

class WaveManager(object):
    def __init__(self):
        self.sessions = {}  # セッションは session_id をキーに管理
        self.next_session_id = 1
        self.trade_logs = []
        self.full_data = []
        self.risk_percentage = 5.0

    def analyze_sessions(self):
        for session_id in list(self.sessions.keys()):
            session = self.sessions[session_id]
            session.start_origin = session.start_index
            session.full_data = self.full_data[session.prev_index-100:session.start_index+1500, :]
            difference = session.start_time_index - session.start_index
            session.start_index = session.start_index - session.prev_index + 100
            session.start_pivot_time = session.full_data[session.start_index, 0]
            session.start_time_index = session.start_index + difference
            session.prev_index = 100
            session.execute_state_action()
            if session.trade_log is not None and len(session.trade_log) > 0:
                self.trade_logs.append(session.trade_log)
            del self.sessions[session_id]
        if self.trade_logs:
            trade_logs_df = pd.DataFrame(self.trade_logs)
            time_columns = ["entry_time", "start_pivot_time"]
            for col in time_columns:
                if col in trade_logs_df.columns:
                    trade_logs_df[col] = pd.to_datetime(trade_logs_df[col], unit="ns", utc=True, errors='coerce')
            trade_logs_df.sort_values(by="entry_time", inplace=True)
            # trade_logs_df.to_csv("test_result/usdjpy_check_no_sma搭載1.csv", index=False)
            # print("ログ出力完了", trade_logs_df.head())
            # print(f"ログ数：{len(trade_logs_df)}")
            self.trade_logs = trade_logs_df
            #↓並列処理導入前に必要だった処理
            # self.summarize_and_export_results(filename=conditions.get("output_file", "final_trade_logs.csv"),
            #                         initial_capital=10000,
            #                         risk_percentage=self.risk_percentage)
            
    def organize_trade_logs(self):
        columns = ["entry_time", "up_trend", "start_pivot_time", "global_entry_index", 
                   "entry_line", "take_profit", "highlow_stop_loss",# "sml_stop_loss", 
                   "point_to_stoploss", "point_to_take_profit", "name"]
        trade_logs = pd.DataFrame(self.trade_logs, columns=columns)
        time_columns = ["entry_time", "start_pivot_time", "exit_time", "order_time"]
        for col in time_columns:
            if col in trade_logs.columns:
                trade_logs[col] = pd.to_datetime(trade_logs[col], unit="ns", utc=True, errors='coerce')
        self.trade_logs = trade_logs

    # @profile
    def add_session(self, start_index, start_time_index, prev_index, prev_time_index, up_trend, stop_strategy="sml", tp_level=138):
        session = MyModel(f"Session_{self.next_session_id}", start_index, start_time_index, prev_index, prev_time_index, up_trend, stop_strategy, tp_level)
        session.original_offset = prev_index - 100
        self.sessions[self.next_session_id] = session
        self.next_session_id += 1
        return session

    def summarize_and_export_results(self, filename="final_trade_logs.csv", initial_capital=10000, risk_percentage=10.0):
        """
        trade_logs の統計情報を計算し、CSVの最後に以下の情報を追加する：
          ・プロフィットファクター：勝ちトレードの総利益 / 負けトレードの総損失
          ・平均リスクリワード比率：各トレードでの risk_reward_ratio の平均値
          ・最大連敗数：連続した負けトレードの最大回数
          ・最大ドローダウン：資金推移における最大の落ち込み
          ・初期資金からの資金推移：1万円スタート時のシミュレーション結果
          ・設定リスク（%）および1回あたりのリスク金額（初期資金×risk_percentage/100）
        
        ※動的リスク管理シミュレーション：各トレードごとに、エントリー時の資金に対して risk_percentage(例：3%)をリスク金額とし、
          勝ちの場合はそのリスク金額×reward_ratio、負けの場合はそのリスク金額分資金から差し引くシミュレーションを行う��
        """
        df = self.trade_logs.copy()
        if df.empty or "profit_loss" not in df.columns:
            print("トレードログが空か、必要な列がありません。")
            return

        # 既存の累積損益（固定値）ではなく、動的リスク管理シミュレーションを実施
        df = df.sort_values("entry_time").reset_index(drop=True)
        capital_list = []
        simulated_pl_list = []
        current_capital = initial_capital
        capital_list.append(current_capital)
        for idx, row in df.iterrows():
            # 各トレードのリスク金額は、エントリー時の資金の risk_percentage%
            risk = current_capital * (risk_percentage / 100.0)
            if row["result"] == "win":
                # 勝ちトレードの場合、利益 = risk × リスクリワード比率
                pl = risk * row["risk_reward_ratio"]
            else:
                # 負けトレードの場合、損失 = risk
                pl = -risk
            simulated_pl_list.append(pl)
            current_capital += pl
            capital_list.append(current_capital)
        df["simulated_pl"] = simulated_pl_list

        # シミュレーションによる資金推移（各トレード後の資金）
        simulated_capital = pd.Series(capital_list[1:], name="capital")
        df["capital"] = simulated_capital

        # シミュレーションによる最大ドローダウンの計算
        cummax = simulated_capital.cummax()
        drawdown = cummax - simulated_capital
        sim_max_drawdown = drawdown.max()

        # 最大連敗数の計算（シミュレーションではなくログ上の profit_loss を使用）
        df["is_loss"] = df["profit_loss"] < 0
        df["loss_group"] = (~df["is_loss"]).cumsum()
        losing_streak = df.groupby("loss_group").cumcount() + 1
        max_consecutive_losses = losing_streak.max()

        # プロフィットファクターの計算（ログの profit_loss を使用）
        total_win = df.loc[df["profit_loss"] > 0, "profit_loss"].sum()
        total_loss = -df.loc[df["profit_loss"] < 0, "profit_loss"].sum()
        profit_factor = total_win / total_loss if total_loss > 0 else np.inf

        # 平均リスクリワード比率（ログの risk_reward_ratio の平均）
        average_rr = df["risk_reward_ratio"].mean()

        risk_amount = initial_capital * (risk_percentage / 100.0)

        # 勝率の計算
        win_rate = (df["result"] == "win").mean() * 100

        # CSV出力
        df.to_csv(filename, index=False)
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["統計情報"])
            writer.writerow(["プロフィットファクター", profit_factor])
            writer.writerow(["平均リスクリワード比率", average_rr])
            writer.writerow(["最大連敗数", max_consecutive_losses])
            writer.writerow(["最大ドローダウン（シミュレーション）", sim_max_drawdown])
            writer.writerow(["初期資金", initial_capital])
            writer.writerow(["最終資金（シミュレーション）", simulated_capital.iloc[-1]])
            writer.writerow(["設定リスク（%）", risk_percentage])
            writer.writerow(["1���あたりのリスク金額（初期時）", risk_amount])
            writer.writerow(["勝率（%）", win_rate])
        print(f"トレードログと統計情報を {filename} に書き出しました。")

    def __repr__(self):
        return f"WaveManager(sessions={list(self.sessions.values())})"
