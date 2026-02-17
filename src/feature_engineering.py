# src/feature_engineering.py

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data" / "raw" / "factoryguard_production_data_20min.parquet"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "factoryguard_features.parquet"



def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machine_id", "timestamp"])

    sensor_cols = ["temperature", "vibration", "pressure"]
    windows = {
        "1h": 3,
        "6h": 18,
        "12h": 36,
    }

    for col in sensor_cols:
        # Lag features
        df[f"{col}_lag1"] = df.groupby("machine_id")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("machine_id")[col].shift(2)

        # Rolling statistics
        for w_name, w_size in windows.items():
            df[f"{col}_mean_{w_name}"] = (
                df.groupby("machine_id")[col]
                .shift(1)
                .rolling(w_size)
                .mean()
            )

            df[f"{col}_std_{w_name}"] = (
                df.groupby("machine_id")[col]
                .shift(1)
                .rolling(w_size)
                .std()
            )

    return df


def main():
    print("Loading raw data...")
    df = pd.read_parquet(RAW_PATH)

    print("Creating features...")
    df_fe = create_features(df)

    # Drop rows with NaNs caused by rolling windows
    df_fe = df_fe.dropna().reset_index(drop=True)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_fe.to_parquet(PROCESSED_PATH, index=False)

    print(f"Feature engineering complete.")
    print(f"Saved to: {PROCESSED_PATH}")
    print(f"Final shape: {df_fe.shape}")


if __name__ == "__main__":
    main()
