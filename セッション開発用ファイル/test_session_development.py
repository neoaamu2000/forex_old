import pytest
import pandas as pd
from datetime import datetime
import pytz
import numpy as np

from test_main import MyModel

# conditionsをフィクスチャとして定義
@pytest.fixture
def conditions():
    return {
        "symbol": "USDJPY",
        "fromdate": datetime(2025, 2, 18, 1, 55, tzinfo=pytz.UTC),
        "todate": datetime(2025, 2, 28, 7, 0, tzinfo=pytz.UTC),
    }

def test_execute_actions(conditions):
    fromdate = conditions.get("fromdate")
    todate = conditions.get("todate")
    df = pd.read_csv("main_data.csv")
    df = df.drop(df.columns[0], axis=1)
    df["time"] = pd.to_datetime(df["time"])
    df = df.loc[df["time"] < todate]
    df = df.reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"]).astype("int64")
    df["pivot_time"] = pd.to_datetime(df["pivot_time"]).astype("int64")
    df["sml_pivot_time"] = pd.to_datetime(df["sml_pivot_time"]).astype("int64")
    print(df["pivot_time"].dtype)
    full_data = df.to_numpy(dtype=np.float64)
    test_model = MyModel(1,start_index=231, start_time_index=251, prev_index=216, prev_time_index=230,up_trend="True")
    test_model.full_data = full_data
    test_model.execute_state_action()

if __name__ == "__main__":

    pytest.main()
