"""Porównaj baseline i modele AutoGluona na tym samym zbiorze testowym.

Każdy model jest ładowany w osobnym procesie. Dzięki temu pomiar pamięci nie
dziedziczy bibliotek ani modeli załadowanych podczas wcześniejszego pomiaru.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMN = "__benchmark_target__"
MIB = 1024 * 1024

MODEL_SPECS = (
    {
        "name": "baseline-logistic-regression",
        "kind": "pickle",
        "path": Path("data/06_models/baseline_model.pkl"),
    },
    {
        "name": "autogluon-medium-60s",
        "kind": "autogluon",
        "path": Path("data/06_models/autogluon/medium-60s"),
    },
    {
        "name": "autogluon-good-120s",
        "kind": "autogluon",
        "path": Path("data/06_models/autogluon/good-120s"),
    },
)


def artifact_size_bytes(path: Path) -> int:
    """Zwróć rozmiar pliku albo wszystkich plików w katalogu modelu."""
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    raise FileNotFoundError(f"Nie znaleziono artefaktu modelu: {path}")


def _load_model(kind: str, path: Path) -> Any:
    if kind == "pickle":
        import pickle

        with path.open("rb") as stream:
            return pickle.load(stream)
    if kind == "autogluon":
        from autogluon.tabular import TabularPredictor

        return TabularPredictor.load(str(path))
    raise ValueError(f"Nieobsługiwany rodzaj modelu: {kind}")


def _measure_worker(args: argparse.Namespace) -> int:
    import pandas as pd
    import psutil

    test_data = pd.read_csv(args.test_data)
    y_test = test_data.pop(TARGET_COLUMN)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    peak_rss = [rss_before]
    stop_sampling = threading.Event()

    def sample_memory() -> None:
        while not stop_sampling.wait(0.005):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    load_started = time.perf_counter()
    try:
        model = _load_model(args.model_kind, args.model_path)
    finally:
        load_time_seconds = time.perf_counter() - load_started
        rss_after = process.memory_info().rss
        peak_rss[0] = max(peak_rss[0], rss_after)
        stop_sampling.set()
        sampler.join()

    model.predict(test_data)  # rozgrzewka: nie wliczamy jednorazowego przygotowania predykcji
    durations: list[float] = []
    predictions = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        predictions = model.predict(test_data)
        durations.append(time.perf_counter() - started)

    assert predictions is not None
    from sklearn.metrics import f1_score

    prediction_seconds = statistics.median(durations)
    result = {
        "name": args.model_name,
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "prediction_time_ms": prediction_seconds * 1000,
        "prediction_time_per_row_ms": prediction_seconds * 1000 / len(test_data),
        "artifact_size_mb": artifact_size_bytes(args.model_path) / MIB,
        "ram_after_load_mb": max(0, rss_after - rss_before) / MIB,
        "peak_ram_during_load_mb": max(0, peak_rss[0] - rss_before) / MIB,
        "load_time_seconds": load_time_seconds,
        "test_rows": len(test_data),
        "repeats": args.repeats,
    }
    args.result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def _prepare_test_data(path: Path) -> int:
    import pandas as pd
    import yaml

    from asi_project.config import get_contract, load_project_config
    from asi_project.data import make_xy, split_train_test

    prepared_data = pd.read_csv(PROJECT_ROOT / "data/02_intermediate/prepared_data.csv")
    config = load_project_config()
    contract = get_contract(config)
    parameters = yaml.safe_load(
        (PROJECT_ROOT / "conf/base/parameters.yml").read_text(encoding="utf-8")
    )
    X, y = make_xy(prepared_data, contract, config.model_features)
    _, X_test, _, y_test = split_train_test(X, y, contract, **parameters["split"])
    benchmark_data = X_test.copy()
    benchmark_data[TARGET_COLUMN] = y_test.to_numpy()
    benchmark_data.to_csv(path, index=False)
    return len(benchmark_data)


def _run_clean_process(
    spec: dict[str, Any], test_data_path: Path, result_path: Path, repeats: int
) -> dict[str, Any]:
    model_path = (PROJECT_ROOT / spec["path"]).resolve()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model-name",
        str(spec["name"]),
        "--model-kind",
        str(spec["kind"]),
        "--model-path",
        str(model_path),
        "--test-data",
        str(test_data_path),
        "--result-path",
        str(result_path),
        "--repeats",
        str(repeats),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        raise RuntimeError(f"Pomiar modelu {spec['name']} nie powiódł się:\n{details}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def render_markdown(results: list[dict[str, Any]]) -> str:
    """Zbuduj czytelną tabelę oraz opis wspólnej metody pomiaru."""
    if not results:
        raise ValueError("Brak wyników do zapisania.")
    test_rows = {int(result["test_rows"]) for result in results}
    repeats = {int(result["repeats"]) for result in results}
    if len(test_rows) != 1 or len(repeats) != 1:
        raise ValueError("Wszystkie modele muszą być zmierzone w tych samych warunkach.")

    rows = []
    for result in results:
        rows.append(
            "| {name} | {f1:.4f} | {prediction_time_ms:.2f} | "
            "{prediction_time_per_row_ms:.4f} | {artifact_size_mb:.3f} | "
            "{ram_after_load_mb:.2f} |".format(**result)
        )

    best = max(results, key=lambda result: float(result["f1"]))

    return "\n".join(
        [
            "# Porównanie modeli — Sprint 3",
            "",
            "Wszystkie modele oceniono na tym samym zbiorze testowym utworzonym przy użyciu "
            "parametrów `test_size: 0.2` i `random_state: 42` z konfiguracji projektu.",
            "",
            "| Model | F1 | Czas predykcji zbioru testowego [ms] | "
            "Czas na rekord [ms] | Rozmiar artefaktu [MB] | RAM po wczytaniu [MB] |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Metoda pomiaru",
            "",
            f"- zbiór testowy: {test_rows.pop()} rekordów;",
            f"- czas predykcji: mediana z {repeats.pop()} powtórzeń po jednej rozgrzewce;",
            "- rozmiar artefaktu: suma rozmiarów wszystkich plików modelu na dysku;",
            "- RAM po wczytaniu: przyrost pamięci RSS po załadowaniu modelu w osobnym, "
            "czystym procesie;",
            "- główna metryka projektu: F1 dla klasy pozytywnej `Churn=1`.",
            "",
            "## Wniosek",
            "",
            f"Najwyższe F1 osiągnął `{best['name']}`: {best['f1']:.4f}. Ostateczny wybór "
            "modelu produkcyjnego powinien uwzględniać również koszt pamięci, rozmiar "
            "artefaktu i czas predykcji przedstawione w tabeli.",
            "",
        ]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs/model_comparison.md",
        help="Plik Markdown z tabelą porównawczą.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-name", help=argparse.SUPPRESS)
    parser.add_argument("--model-kind", help=argparse.SUPPRESS)
    parser.add_argument("--model-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--test-data", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-path", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("Liczba powtórzeń musi być dodatnia.")
    if args.worker:
        return _measure_worker(args)

    with tempfile.TemporaryDirectory(prefix="asi-model-benchmark-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        test_data_path = temporary_path / "test_data.csv"
        _prepare_test_data(test_data_path)
        results = []
        for index, spec in enumerate(MODEL_SPECS):
            result_path = temporary_path / f"result-{index}.json"
            print(f"Pomiar: {spec['name']}")
            results.append(_run_clean_process(spec, test_data_path, result_path, args.repeats))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(results), encoding="utf-8")
    print(f"Zapisano tabelę: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
