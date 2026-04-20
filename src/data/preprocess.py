"""Preprocess raw customer data."""

import argparse
import tomllib
from pathlib import Path

import pandas as pd


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Drop nulls and return only needed columns."""
    all_columns = feature_columns + ["churn"]
    return df[all_columns].dropna()


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save processed DataFrame to CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved processed data → {filepath}  ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # This lets you pass arguments from the terminal to your script, instead
    # of hardcoding them inside.
    # without argparse you'd hardcode the path inside the script
    # with argparse you can pass it from outside:
    # python src/data/preprocess.py --config configs/config.toml
    # python src/data/preprocess.py --config configs/config_v2.toml  # easy to
    # swap!
    parser.add_argument("--config", default="configs/config.toml")
    args = parser.parse_args()

    print("args", args.config)

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    df = load_raw_data(config["data"]["raw_data_path"])
    df = clean_data(df, config["data"]["feature_columns"])
    save_data(df, config["data"]["processed_data_path"])
