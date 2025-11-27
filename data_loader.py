
import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("time_sec").reset_index(drop=True)
    df["time_of_day"] = pd.to_datetime(df["time_of_day"], format="%H:%M:%S")
    return df
