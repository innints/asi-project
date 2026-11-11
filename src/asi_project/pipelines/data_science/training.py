"""Węzły podziału i treningu z parametrami przekazywanymi przez Kedro."""

import pandas as pd
from kedro.pipeline import Node, node
from sklearn.pipeline import Pipeline

from asi_project.config import get_contract, load_project_config
from asi_project.data import make_xy, split_train_test
from asi_project.model import build_baseline_pipeline


def split_prepared_data(
    prepared_data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Wybierz cechy modelu i wykonaj podział zgodny z kontraktem danych."""
    config = load_project_config()
    contract = get_contract(config)
    X, y = make_xy(prepared_data, contract, config.model_features)
    return split_train_test(X, y, contract, test_size, random_state)


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_parameters: dict[str, int],
) -> Pipeline:
    """Dopasuj preprocessing i klasyfikator wyłącznie na części treningowej."""
    config = load_project_config()
    model = build_baseline_pipeline(
        config.numeric_features,
        config.categorical_features,
        model_parameters=model_parameters,
    )
    return model.fit(X_train, y_train)


def create_training_nodes() -> list[Node]:
    """Zwróć węzły do połączenia z ewaluacją w potoku data_science."""
    return [
        node(
            split_prepared_data,
            inputs=["prepared_data", "params:split.test_size", "params:split.random_state"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_train_test",
        ),
        node(
            train_baseline_model,
            inputs=["X_train", "y_train", "params:logistic_regression"],
            outputs="baseline_model",
            name="train_baseline_model",
        ),
    ]
