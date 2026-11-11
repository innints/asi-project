"""Rejestr pipeline'ów Kedro i pełnego przebiegu Sprintu 2."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Połącz przygotowanie danych z treningiem i ewaluacją w przebiegu domyślnym."""
    pipelines = find_pipelines()
    if "__default__" in pipelines:
        pipelines["__default__"] = pipelines["data_processing"] + pipelines["data_science"]
    return pipelines
