"""Tests for training functions."""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.train import evaluate_model, train_model


# Tiny fake dataset used across all tests
@pytest.fixture
def fake_data():
    """Create a minimal fake dataset for testing."""
    X_train = pd.DataFrame({
        "age": [25, 45, 30, 55, 22],
        "balance": [1000, 3000, 500, 8000, 100],
        "tenure": [12, 60, 6, 120, 1],
        "transaction_count": [30, 80, 10, 200, 2],
    })
    y_train = pd.Series([0, 0, 1, 0, 1])
    return X_train, y_train


# Fake config matching your config.toml structure
@pytest.fixture
def fake_config():
    """Minimal config for testing."""
    return {
        "model": {"n_estimators": 10, "max_depth": 3},
        "data": {"random_state": 42},
    }


def test_train_model_returns_classifier(fake_data, fake_config):
    """Test that train_model returns a fitted RandomForest."""
    X_train, y_train = fake_data
    model = train_model(X_train, y_train, fake_config)
    assert isinstance(model, RandomForestClassifier)


def test_train_model_can_predict(fake_data, fake_config):
    """Test that the trained model can make predictions."""
    X_train, y_train = fake_data
    model = train_model(X_train, y_train, fake_config)
    predictions = model.predict(X_train)
    assert len(predictions) == len(y_train)


def test_evaluate_model_returns_metrics(fake_data, fake_config):
    """Test that evaluate_model returns accuracy and f1."""
    X_train, y_train = fake_data
    model = train_model(X_train, y_train, fake_config)
    metrics = evaluate_model(model, X_train, y_train)
    assert "accuracy" in metrics
    assert "f1_score" in metrics


def test_evaluate_model_scores_between_0_and_1(fake_data, fake_config):
    """Test that metric values are valid scores."""
    X_train, y_train = fake_data
    model = train_model(X_train, y_train, fake_config)
    metrics = evaluate_model(model, X_train, y_train)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
