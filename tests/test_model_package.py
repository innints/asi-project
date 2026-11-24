from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

# Będziemy na modelu Dummy Testować zapis i wczytanie modelu
from asi_project.model_package import (
    REQUIRED_METADATA_KEYS,
    load_model_package,
    save_model_package,
)


def _sample_metadata() -> dict:
    return {
        "features": ["feat_1", "feat_2"],
        "target": "target",
        "positive_class": 1,
        "metrics": {"f1": 0.75, "accuracy": 0.8},
        "git_sha": "abc1234",
        "data_fingerprint": "0123456789abcdef",
        "library_versions": {"scikit-learn": "1.7.2", "pandas": "2.3.3"},
    }


def test_save_and_load_model_package(tmp_path):
    X = pd.DataFrame({"feat_1": [1.0, 2.0, 3.0, 4.0], "feat_2": [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)

    package_dir = tmp_path / "test_package"
    save_model_package(model=model, metadata=_sample_metadata(), output_dir=package_dir)

    assert (package_dir / "model.joblib").is_file()
    assert (package_dir / "metadata.json").is_file()

    loaded_model, loaded_metadata = load_model_package(package_dir)

    assert REQUIRED_METADATA_KEYS.issubset(loaded_metadata.keys())
    assert loaded_metadata["features"] == ["feat_1", "feat_2"]
    assert loaded_metadata["model_format"] == "joblib"

    preds = loaded_model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_save_model_package_missing_metadata_raises(tmp_path):
    X = pd.DataFrame({"feat_1": [1.0, 2.0], "feat_2": [0, 1]})
    y = pd.Series([0, 1])

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)

    incomplete_metadata = {"features": ["feat_1", "feat_2"]}

    with pytest.raises(ValueError, match="Brakuje wymaganych pol"):
        save_model_package(
            model=model,
            metadata=incomplete_metadata,
            output_dir=tmp_path / "bad_package",
        )


def test_save_and_load_autogluon_package_mock(tmp_path):
    # 1. Udajemy folder wytrenowanego modelu AutoGluona (bez prawdziwego treningu)
    fake_ag_dir = tmp_path / "fake_ag_source"
    fake_ag_dir.mkdir()
    (fake_ag_dir / "predictor.pkl").write_text("dummy", encoding="utf-8")

    fake_ag_model = MagicMock()
    fake_ag_model.path = str(fake_ag_dir)

    # 2. Zapis przez wspolna funkcje
    package_dir = tmp_path / "ag_package"
    save_model_package(
        model=fake_ag_model,
        metadata=_sample_metadata(),
        output_dir=package_dir,
    )

    assert (package_dir / "model" / "predictor.pkl").is_file()
    assert (package_dir / "metadata.json").is_file()

    # 3. Odczyt z podmienionym TabularPredictor.load (zeby nie ladowac prawdziwego AG)
    mock_loaded = MagicMock()
    mock_loaded.predict.return_value = pd.Series([0, 1])

    with patch("autogluon.tabular.TabularPredictor.load", return_value=mock_loaded) as mock_load:
        loaded_model, loaded_metadata = load_model_package(package_dir)

    mock_load.assert_called_once_with(str(package_dir / "model"))
    assert loaded_metadata["model_format"] == "autogluon"

    X = pd.DataFrame({"feat_1": [1.0, 2.0], "feat_2": [0, 1]})
    preds = loaded_model.predict(X)
    assert len(preds) == 2
