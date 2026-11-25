import tempfile
from pathlib import Path

import pandas as pd
import yaml

from asi_project.config import get_contract, load_project_config
from asi_project.data import load_dataset, normalize_input
from asi_project.model_package import load_model_package


def run_smoke_test() -> None:
    import wandb
    config = load_project_config()
    contract = get_contract(config)
    params = yaml.safe_load((config.root / "conf" / "base" / "parameters.yml").read_text(encoding="utf-8"))

    entity = params["wandb"]["entity"]
    project = params["wandb"]["project"]
    artifact_ref = f"{entity}/{project}/iranian-churn-model:production"

    with tempfile.TemporaryDirectory() as tmp_dir:
        api = wandb.Api()
        artifact = api.artifact(artifact_ref, type="model")
        downloaded_dir = Path(artifact.download(root=tmp_dir))

        model, metadata = load_model_package(downloaded_dir)

        mode = "sample" if config.ci_sample_path.is_file() else "full"
        df_sample = load_dataset(config, contract, mode=mode).head(5)
        features: list[str] = metadata["features"]
        X_five = normalize_input(df_sample[features])

        predictions = pd.Series(model.predict(X_five))

        assert len(predictions) == 5, f"Oczekiwano 5 predykcji, otrzymano: {len(predictions)}"
        assert not predictions.isna().any(), "Predykcje zawierają wartości NaN!"
        allowed_values = {0, 1}
        actual_values = set(predictions.astype(int).tolist())
        assert actual_values.issubset(allowed_values), (
            f"Nieoczekiwane klasy w predykcjach: {actual_values} (dozwolone: {allowed_values})"
        )

    
        print(f"Artefakt: {artifact_ref}")
        print(f"Format modelu: {metadata.get('model_format')} | Git SHA: {metadata.get('git_sha')}")
        print(f"Predykcje dla 5 rekordów: {predictions.astype(int).tolist()}")


if __name__ == "__main__":
    run_smoke_test()