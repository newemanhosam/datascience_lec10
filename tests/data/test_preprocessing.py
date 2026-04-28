"""Tests for preprocessing functions."""

import pandas as pd
from src.data.preprocess import clean_data, load_raw_data


def test_load_raw_data():
    """Test that loading returns a DataFrame."""
    df = load_raw_data("data/raw/customers.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_clean_data_removes_nulls():
    """Test that clean_data drops rows with nulls."""
    df = pd.DataFrame(
        {
            "age": [25, None, 30],
            "balance": [1000, 2000, None],
            "tenure": [12, 24, 36],
            "transaction_count": [10, 20, 30],
            "churn": [0, 1, 0],
        }
    )
    feature_columns = ["age", "balance", "tenure", "transaction_count"]
    cleaned = clean_data(df, feature_columns)
    assert len(cleaned) == 1  # only first row has no nulls


def test_clean_data_correct_columns():
    """Test that clean_data returns only expected columns."""
    df = pd.DataFrame(
        {
            "age": [25, 30],
            "balance": [1000, 2000],
            "tenure": [12, 24],
            "transaction_count": [10, 20],
            "churn": [0, 1],
        }
    )
    feature_columns = ["age", "balance", "tenure", "transaction_count"]
    cleaned = clean_data(df, feature_columns)
    assert list(cleaned.columns) == feature_columns + ["churn"]
