import pandas as pd
from kedro.pipeline import Pipeline, node
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline as SklearnPipeline

from .training import create_training_nodes


def evaluate_model(
    model: SklearnPipeline, X_test: pd.DataFrame, y_test: pd.Series, split_params: dict
) -> dict:
    y_predict = model.predict(X_test)
    cm = confusion_matrix(y_test, y_predict)
    tn, fp, fn, tp = cm.ravel()

    return {
        "split": split_params,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_predict)),
            "precision": float(precision_score(y_test, y_predict, zero_division=0)),
            "recall": float(recall_score(y_test, y_predict, zero_division=0)),
            "f1": float(f1_score(y_test, y_predict, zero_division=0)),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        },
    }


def create_pipeline(**kwargs) -> Pipeline:
    evaluation_node = node(
        evaluate_model,
        inputs=["baseline-model", "X_test", "y_test", "params:split"],
        outputs="baseline_metrics",
        name="evaluate_model",
    )
    return Pipeline(create_training_nodes() + [evaluation_node])
