"""Eksperymenty AutoGluona uruchamiane jako część pipeline'u Kedro."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .tracking import get_data_fingerprint, get_git_sha, log_wandb_run


def _classification_metrics(y_true: pd.Series, y_predict: pd.Series) -> dict[str, Any]:
    """Policz metryki binarne w formacie zgodnym z raportem baseline'u."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_predict)),
        "precision": float(precision_score(y_true, y_predict, zero_division=0)),
        "recall": float(recall_score(y_true, y_predict, zero_division=0)),
        "f1": float(f1_score(y_true, y_predict, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _validate_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = config.get("experiments")
    if not isinstance(experiments, list) or len(experiments) < 2:
        raise ValueError("AutoGluon wymaga co najmniej dwóch konfiguracji eksperymentu.")

    names: set[str] = set()
    validated: list[dict[str, Any]] = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("Każdy eksperyment AutoGluona musi być mapą parametrów.")
        name = str(experiment.get("name", "")).strip()
        presets = str(experiment.get("presets", "")).strip()
        time_limit = experiment.get("time_limit")
        if not name or not presets or not isinstance(time_limit, int | float) or time_limit <= 0:
            raise ValueError("Eksperyment wymaga name, presets i dodatniego time_limit.")
        if name in names:
            raise ValueError(f"Powtórzona nazwa eksperymentu AutoGluona: {name}.")
        names.add(name)
        validated.append({"name": name, "presets": presets, "time_limit": time_limit})
    return validated


def _create_predictor(*, label: str, eval_metric: str, path: Path):
    from autogluon.tabular import TabularPredictor

    return TabularPredictor(
        label=label,
        problem_type="binary",
        eval_metric=eval_metric,
        positive_class=1,
        path=str(path),
    )


def _flat_wandb_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    matrix = metrics["confusion_matrix"]
    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "cm_tn": matrix["tn"],
        "cm_fp": matrix["fp"],
        "cm_fn": matrix["fn"],
        "cm_tp": matrix["tp"],
    }


def run_autogluon_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    autogluon_config: dict[str, Any],
    wandb_config: dict[str, str],
    split_config: dict[str, Any],
    *,
    predictor_factory: Callable[..., Any] | None = None,
    run_logger: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Wytrenuj jawnie skonfigurowane warianty AutoGluona i zapisz ich runy."""
    if not bool(autogluon_config.get("enabled", False)):
        return {"enabled": False, "experiments": []}

    experiments = _validate_experiments(autogluon_config)
    eval_metric = str(autogluon_config.get("eval_metric", "f1"))
    output_dir = Path(str(autogluon_config.get("output_dir", "data/06_models/autogluon")))
    output_dir.mkdir(parents=True, exist_ok=True)

    label = y_train.name or "target"
    train_data = X_train.copy()
    train_data[label] = y_train.to_numpy()

    git_sha = get_git_sha()
    data_fingerprint = get_data_fingerprint()
    predictor_factory = predictor_factory or _create_predictor
    run_logger = run_logger or log_wandb_run

    results: list[dict[str, Any]] = []
    for experiment in experiments:
        model_path = output_dir / experiment["name"]
        if model_path.exists():
            raise FileExistsError(
                f"Katalog modelu {model_path} już istnieje. Usuń go albo zmień nazwę runu."
            )

        predictor = predictor_factory(
            label=label,
            eval_metric=eval_metric,
            path=model_path,
        )
        predictor.fit(
            train_data=train_data,
            presets=experiment["presets"],
            time_limit=experiment["time_limit"],
        )
        predictions = pd.Series(predictor.predict(X_test), index=y_test.index)
        metrics = _classification_metrics(y_test, predictions)

        run_config = {
            "model_type": "autogluon",
            "experiment": experiment["name"],
            "presets": experiment["presets"],
            "time_limit": experiment["time_limit"],
            "eval_metric": eval_metric,
            "split": split_config,
            "git_sha": git_sha,
            "data_fingerprint": data_fingerprint,
        }
        run_logger(
            wandb_config=wandb_config,
            run_name=f"autogluon-{experiment['name']}",
            run_config=run_config,
            metrics=_flat_wandb_metrics(metrics),
            tags=["autogluon", "sprint-3", experiment["name"]],
        )

        results.append(
            {
                "name": experiment["name"],
                "presets": experiment["presets"],
                "time_limit": experiment["time_limit"],
                "best_model": str(predictor.model_best),
                "model_path": model_path.as_posix(),
                "metrics": metrics,
            }
        )

    return {
        "enabled": True,
        "eval_metric": eval_metric,
        "git_sha": git_sha,
        "data_fingerprint": data_fingerprint,
        "experiments": results,
    }
