"""Wspólne informacje o pochodzeniu eksperymentów i logowanie do W&B."""

import json
import subprocess
from pathlib import Path
from typing import Any

from asi_project.config import load_project_config


def get_git_sha() -> str:
    """Zwróć commit, na którym uruchomiono eksperyment."""
    root = load_project_config().root
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def get_data_fingerprint(fingerprint_path: Path | None = None) -> str:
    """Odczytaj SHA-256 pełnego pliku danych z ``fingerprint.json``."""
    if fingerprint_path is None:
        config = load_project_config()
        if not config.dataset_id:
            raise ValueError("Nie wybrano datasetu w project.yaml.")
        fingerprint_path = config.root / "data" / "01_raw" / config.dataset_id / "fingerprint.json"

    payload = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    sha256 = payload.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(f"Niepoprawne pole sha256 w {fingerprint_path}.")
    try:
        int(sha256, 16)
    except ValueError as error:
        raise ValueError(f"Niepoprawne pole sha256 w {fingerprint_path}.") from error
    return sha256.lower()


def log_wandb_run(
    *,
    wandb_config: dict[str, str],
    run_name: str,
    run_config: dict[str, Any],
    metrics: dict[str, float | int],
    tags: list[str],
) -> None:
    """Zapisz jeden eksperyment w projekcie W&B skonfigurowanym przez zespół."""
    import wandb

    with wandb.init(
        entity=wandb_config["entity"],
        project=wandb_config["project"],
        name=run_name,
        tags=tags,
        config=run_config,
    ) as run:
        run.log(metrics)
