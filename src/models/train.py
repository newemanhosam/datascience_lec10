"""Train a Random Forest model for churn prediction."""

import argparse
import json
import pickle
import tomllib
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def load_data(filepath: str, feature_columns: list[str], target_column: str):
    """Load processed data and split into features and target."""
    df = pd.read_csv(filepath)
    X = df[feature_columns]
    y = df[target_column]
    return X, y


def train_model(X_train, y_train, config: dict) -> RandomForestClassifier:
    """Train a random forest classifier."""
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["data"]["random_state"],
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Return accuracy and f1 metrics."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }


def save_model(model, filepath: str) -> None:
    """Pickle the trained model."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {filepath}")


def save_metrics(metrics: dict, filepath: str) -> None:
    """Save metrics dict to JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {filepath}")
    print(f"  accuracy : {metrics['accuracy']}")
    print(f"  f1_score : {metrics['f1_score']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.toml")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    X, y = load_data(
        config["data"]["processed_data_path"],
        config["data"]["feature_columns"],
        config["data"]["target_column"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    model = train_model(X_train, y_train, config)
    metrics = evaluate_model(model, X_test, y_test)

    save_model(model, config["model"]["model_output_path"])
    save_metrics(metrics, config["reports"]["metrics_path"])
