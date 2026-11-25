import json
from pathlib import Path

import wandb
import yaml

from asi_project.pipelines.data_science.tracking import get_data_fingerprint, get_git_sha


# zapisanie danych do W&B
def main() -> None:
    metrics_report = json.loads(Path("data/08_reporting/metrics.json").read_text(encoding="utf-8"))
    params = yaml.safe_load(Path("conf/base/parameters.yml").read_text(encoding="utf-8"))

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
