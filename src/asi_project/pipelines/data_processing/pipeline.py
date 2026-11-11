"""Kolejność kroków przygotowania danych w Kedro."""

from kedro.pipeline import Pipeline, node

from asi_project.data import filter_training_rows, normalize_input

from .nodes import validate_raw_data


def create_pipeline() -> Pipeline:
    """Zbuduj potok od surowych danych do danych gotowych do podziału."""
    return Pipeline(
        [
            node(
                validate_raw_data,
                inputs="raw_data",
                outputs=["validated_data", "dataset_contract"],
                name="validate_data_contract",
            ),
            node(
                normalize_input,
                inputs="validated_data",
                outputs="normalized_data",
                name="normalize_input",
            ),
            node(
                filter_training_rows,
                inputs=["normalized_data", "dataset_contract"],
                outputs="prepared_data",
                name="filter_training_rows",
            ),
        ]
    )
