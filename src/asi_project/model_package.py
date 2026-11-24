import json
import shutil
from pathlib import Path
from typing import Any

import joblib

# Wymagane rzeczy do zapisania
REQUIRED_METADATA_KEYS = {
    "features",
    "target",
    "positive_class",
    "metrics",
    "git_sha",
    "data_fingerprint",
    "library_versions",
}  # +model_format


# funkcja do sprawdzenia wymagań przy save i load
def _validate_metadata(metadata: dict[str, Any]) -> None:
    missing = REQUIRED_METADATA_KEYS - set(metadata.keys())
    if missing:
        raise ValueError(f"Brakuje wymaganych pol w metadata.json:{missing}")


def save_model_package(
    model: Any,
    metadata: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    _validate_metadata(metadata)

    # Tworzenie folderu do zapisu
    package_dir = Path(output_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    # AutoGluon zapisuje model jako katalog (ma atrybut .path wskazujacy na folder),a zwykle modele (np. sklearn) zapisujemy jako pojedynczy plik .joblib
    if hasattr(model, "path") and Path(model.path).is_dir():
        model_format = "autogluon"
        # operacje na folderach, czyszczenie i kopiowanie
        model_dest = package_dir / "model"
        if model_dest.exists():
            shutil.rmtree(model_dest)
        shutil.copytree(model.path, model_dest)
    else:
        model_format = "joblib"
        joblib.dump(model, package_dir / "model.joblib")

    full_metadata = {**metadata, "model_format": model_format}
    metadata_path = package_dir / "metadata.json"
    metadata_path.write_text(json.dumps(full_metadata, indent=2), encoding="utf-8")

    return package_dir


def load_model_package(package_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    package_path = Path(package_dir)
    metadata_path = package_path / "metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _validate_metadata(metadata)

    model_format = metadata.get("model_format", "joblib")
    if model_format == "autogluon":
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor.load(str(package_path / "model"))
    else:
        model = joblib.load(package_path / "model.joblib")

    return model, metadata
