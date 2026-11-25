import importlib.metadata
from typing import Any

import pandas as pd

import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from asi_project.config import get_contract, load_project_config
from asi_project.data import (
    assert_valid,
    filter_training_rows,
    load_dataset,
    make_xy,
    normalize_input,
    split_train_test,
)
from asi_project.model_package import save_model_package
from asi_project.pipelines.data_science.tracking import get_data_fingerprint, get_git_sha

def _get_library_versions() -> dict[str,str]:
    versions: dict[str,str] = {}
    for pkg in ("scikit-learn", "pandas", "numpy", "joblib", "autogluon.tabular", "wandb"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions

def _compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

def main() -> None:
    import wandb
    from autogluon.tabular import TabularPredictor

    config = load_project_config()
    contract = get_contract(config)
    params = yaml.safe_load((config.root / "conf" / "base" / "parameters.yml").read_text(encoding="utf-8"))

    df_raw = load_dataset(config, contract, mode="full")
    assert_valid(df_raw, contract, config)
    df_norm = normalize_input(df_raw)
    df_clean = filter_training_rows(df_norm, contract)
    X, y = make_xy(df_clean, contract, config.model_features)
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        contract,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
    )

    ag_dir = config.root / "data" / "06_models" / "autogluon" / "medium-60s"
    if ag_dir.is_dir():
        predictor = TabularPredictor.load(str(ag_dir))
    else:
        ag_dir.parent.mkdir(parents=True, exist_ok=True)
        train_data = X_train.copy()
        train_data[contract.target] = y_train.to_numpy()
        predictor = TabularPredictor(
            label=contract.target,
            problem_type="binary",
            eval_metric="f1",
            positive_class=1,
            path=str(ag_dir),
        )
        predictor.fit(train_data=train_data, presets="medium_quality", time_limit=60)

    preds = pd.Series(predictor.predict(X_test), index=y_test.index)
    metrics = _compute_metrics(y_test, preds)

    package_dir = config.root / "data" / "06_models" / "production_package"
    metadata = {
        "model_name": "autogluon-medium-60s",
        "best_submodel": str(predictor.model_best),
        "features": list(config.model_features),
        "target": contract.target,
        "positive_class": str(contract.positive_class),
        "metrics": metrics,
        "git_sha": get_git_sha(),
        "data_fingerprint": get_data_fingerprint(),
        "library_versions": _get_library_versions(),
    }
    save_model_package(predictor, metadata, package_dir)

    wandb_cfg = params["wandb"]
    with wandb.init(
        entity=wandb_cfg["entity"],
        project=wandb_cfg["project"],
        job_type="publish-production-model",
        name="publish-production-autogluon-medium-60s",
        tags=["sprint-3", "production-artifact"],
        config=metadata,
    ) as run:
        run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        artifact = wandb.Artifact(
            name="iranian-churn-model",
            type="model",
            description="Produkcyjny pakiet modelu AutoGluon (medium-60s) dla predykcji odejścia klientów.",
            metadata=metadata,
        )
        artifact.add_dir(str(package_dir))
        run.log_artifact(artifact, aliases=["production"])

    print(f"Zapisano pakiet w {package_dir} i opublikowano w W&B z aliasem 'production'.")


if __name__ == "__main__":
    main()