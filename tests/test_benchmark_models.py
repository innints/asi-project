"""Testy skryptu porównującego modele bez wykonywania kosztownego treningu."""

import importlib.util

import pytest


@pytest.fixture
def benchmark_module(repo_root):
    path = repo_root / "scripts" / "benchmark_models.py"
    spec = importlib.util.spec_from_file_location("benchmark_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_size_bytes_counts_all_files(benchmark_module, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "first.bin").write_bytes(b"1234")
    nested = model_dir / "nested"
    nested.mkdir()
    (nested / "second.bin").write_bytes(b"123456")

    assert benchmark_module.artifact_size_bytes(model_dir) == 10


def test_render_markdown_contains_three_models_and_required_dimensions(benchmark_module):
    base = {
        "f1": 0.75,
        "prediction_time_ms": 10.0,
        "prediction_time_per_row_ms": 0.1,
        "artifact_size_mb": 2.0,
        "ram_after_load_mb": 30.0,
        "test_rows": 100,
        "repeats": 5,
    }
    results = [
        {**base, "name": "baseline-logistic-regression"},
        {**base, "name": "autogluon-medium-60s"},
        {**base, "name": "autogluon-good-120s"},
    ]

    markdown = benchmark_module.render_markdown(results)

    assert all(result["name"] in markdown for result in results)
    assert "F1" in markdown
    assert "Czas predykcji" in markdown
    assert "Rozmiar artefaktu" in markdown
    assert "RAM po wczytaniu" in markdown


def test_render_markdown_rejects_inconsistent_measurements(benchmark_module):
    first = {
        "name": "first",
        "f1": 0.5,
        "prediction_time_ms": 1.0,
        "prediction_time_per_row_ms": 0.1,
        "artifact_size_mb": 1.0,
        "ram_after_load_mb": 1.0,
        "test_rows": 10,
        "repeats": 5,
    }
    second = {**first, "name": "second", "test_rows": 11}

    with pytest.raises(ValueError, match="tych samych warunkach"):
        benchmark_module.render_markdown([first, second])
