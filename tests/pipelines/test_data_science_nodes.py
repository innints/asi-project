import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from asi_project.config import get_contract, load_project_config
from asi_project.pipelines.data_science.pipeline import evaluate_model
from asi_project.pipelines.data_science.training import (
    split_prepared_data,
    train_baseline_model,
)


@pytest.fixture
def sample_prepared_data() -> pd.DataFrame:
    config = load_project_config()
    contract = get_contract(config)

    data = {feature: [1.0, 2.0] * 5 for feature in config.numeric_features}
    for cat_feat in config.categorical_features:
        data[cat_feat] = ["A", "B"] * 5

    target_name = (
        getattr(contract, "target", None)
        or getattr(contract, "target_column", None)
        or getattr(config, "target_column", None)
        or getattr(config, "target", "target")
    )
    data[target_name] = [0, 1] * 5

    return pd.DataFrame(data)


def test_split_prepared_data(sample_prepared_data):
    test_size = 0.2
    random_state = 42

    X_train, X_test, y_train, y_test = split_prepared_data(
        sample_prepared_data,
        test_size=test_size,
        random_state=random_state,
    )

    assert len(X_train) + len(X_test) == len(sample_prepared_data)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_train_model(sample_prepared_data):
    X_train, _, y_train, _ = split_prepared_data(
        sample_prepared_data,
        test_size=0.2,
        random_state=42,
    )

    params = {"max_iter": 100, "random_state": 42}
    model = train_baseline_model(X_train, y_train, model_parameters=params)

    assert isinstance(model, Pipeline)
    assert hasattr(model, "predict")


def test_evaluate_model(sample_prepared_data):
    X_train, X_test, y_train, y_test = split_prepared_data(
        sample_prepared_data,
        test_size=0.2,
        random_state=42,
    )

    params = {"max_iter": 100, "random_state": 42}
    model = train_baseline_model(X_train, y_train, model_parameters=params)

    split_params = {"test_size": 0.2, "random_state": 42}
    metrics = evaluate_model(model, X_test, y_test, split_params)

    assert "split" in metrics
    assert "metrics" in metrics
    assert "accuracy" in metrics["metrics"]
    assert "f1" in metrics["metrics"]
    assert "confusion_matrix" in metrics["metrics"]
    assert 0.0 <= metrics["metrics"]["accuracy"] <= 1.0
