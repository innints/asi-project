import hashlib
import json
import subprocess
from pathlib import Path

import wandb
import yaml


# identyfikator ostatniego commita, żeby było wiadomo na jakim stanie repo był run
def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# sprawdzenie czy wszyscy pracują na tych samych danych
def get_data_fingerprint(raw_dir: Path = Path("data/01_raw")) -> str:
    hasher = hashlib.sha256()
    for path in sorted(raw_dir.glob("*")):
        if path.is_file():
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


# zapisanie danych do W&B
def main() -> None:
    metrics_report = json.loads(Path("data/08_reporting/metrics.json").read_text())
    params = yaml.safe_load(Path("conf/base/parameters.yml").read_text())

    wandb_cfg = params["wandb"]
    metrics = metrics_report["metrics"]
    cm = metrics["confusion_matrix"]

    config = {
        "model_type": "baseline",
        "split": metrics_report["split"],
        "git_sha": get_git_sha(),
        "data_fingerprint": get_data_fingerprint(),
    }

    with wandb.init(
        entity=wandb_cfg["entity"],
        project=wandb_cfg["project"],
        name="baseline",
        tags=["baseline", "sprint-3"],
        config=config,
    ) as run:
        run.log(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "cm_tn": cm["tn"],
                "cm_fp": cm["fp"],
                "cm_fn": cm["fn"],
                "cm_tp": cm["tp"],
            }
        )


if __name__ == "__main__":
    main()
