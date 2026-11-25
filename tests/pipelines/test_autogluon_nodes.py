import json

import pandas as pd
import pytest

from asi_project.pipelines.data_science import autogluon_nodes
from asi_project.pipelines.data_science.autogluon_nodes import run_autogluon_experiments
from asi_project.pipelines.data_science.tracking import get_data_fingerprint


class FakePredictor:
    def __init__(self, path, predictions):
        self.path = path
        self.predictions = predictions
        self.model_best = "WeightedEnsemble_L2"
        self.fit_arguments = None

    def fit(self, **kwargs):
        self.fit_arguments = kwargs
        return self

    def predict(self, X):
        return self.predictions[: len(X)]


def _config(tmp_path):
    return {
        "enabled": True,
        "eval_metric": "f1",
        "output_dir": str(tmp_path / "models"),
        "experiments": [
            {"name": "medium-60s", "presets": "medium_quality", "time_limit": 60},
            {"name": "good-120s", "presets": "good_quality", "time_limit": 120},
        ],
    }


def test_disabled_autogluon_skips_training():
    empty_X = pd.DataFrame()
    empty_y = pd.Series(dtype=int)

    result = run_autogluon_experiments(
        empty_X,
        empty_y,
        empty_X,
        empty_y,
        {"enabled": False},
        {},
        {},
    )

    assert result == {"enabled": False, "experiments": []}


def test_two_experiments_are_trained_evaluated_and_logged(tmp_path, monkeypatch):
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_train = pd.Series([0, 1, 0, 1], name="Churn")
    X_test = pd.DataFrame({"feature": [5, 6, 7, 8]})
    y_test = pd.Series([0, 1, 0, 1], name="Churn")

    created_predictors = []
    logged_runs = []

    def predictor_factory(**kwargs):
        predictor = FakePredictor(kwargs["path"], pd.Series([0, 1, 0, 1]))
        created_predictors.append(predictor)
        return predictor

    def run_logger(**kwargs):
        logged_runs.append(kwargs)

    monkeypatch.setattr(autogluon_nodes, "get_git_sha", lambda: "abc123")
    monkeypatch.setattr(autogluon_nodes, "get_data_fingerprint", lambda: "a" * 64)

    result = run_autogluon_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        _config(tmp_path),
        {"entity": "team", "project": "project"},
        {"test_size": 0.2, "random_state": 42},
        predictor_factory=predictor_factory,
        run_logger=run_logger,
    )

    assert result["enabled"] is True
    assert result["git_sha"] == "abc123"
    assert len(result["experiments"]) == 2
    assert len(created_predictors) == 2
    assert len(logged_runs) == 2
    assert {run["run_name"] for run in logged_runs} == {
        "autogluon-medium-60s",
        "autogluon-good-120s",
    }
    assert all(item["metrics"]["f1"] == 1.0 for item in result["experiments"])
    assert created_predictors[0].fit_arguments["presets"] == "medium_quality"
    assert created_predictors[1].fit_arguments["time_limit"] == 120
    assert list(created_predictors[0].fit_arguments["train_data"].columns) == [
        "feature",
        "Churn",
    ]


def test_enabled_autogluon_requires_two_experiments(tmp_path):
    X = pd.DataFrame({"feature": [1, 2]})
    y = pd.Series([0, 1], name="Churn")
    config = _config(tmp_path)
    config["experiments"] = config["experiments"][:1]

    with pytest.raises(ValueError, match="co najmniej dwóch"):
        run_autogluon_experiments(X, y, X, y, config, {}, {})


def test_data_fingerprint_is_read_from_json(tmp_path):
    fingerprint_path = tmp_path / "fingerprint.json"
    expected_sha = "90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1"
    fingerprint_path.write_text(json.dumps({"sha256": expected_sha}), encoding="utf-8")

    assert get_data_fingerprint(fingerprint_path) == expected_sha
