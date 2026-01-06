import pandas as pd

def load_series(path, value_col="Adj Close"):
    df = pd.read_csv(path)

    # Tarihi parse et
    df["Date"] = pd.to_datetime(df["Date"])

    # Eskiden → yeniye sırala
    df = df.sort_values("Date")

    if value_col not in df.columns:
        raise ValueError(f"{value_col} column not found in CSV")

    series = df[value_col].astype(float).reset_index(drop=True)
    return series
