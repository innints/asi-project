"""Połączenie kontraktu projektu z wejściem potoku Kedro."""

import pandas as pd

from asi_project.config import DatasetContract, get_contract, load_project_config
from asi_project.data import assert_valid, normalize_column_names


def validate_raw_data(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, DatasetContract]:
    """Sprawdź wejście i przekaż kontrakt do filtrowania danych treningowych."""
    config = load_project_config()
    contract = get_contract(config)
    data = normalize_column_names(raw_data)
    assert_valid(data, contract, config)
    return data, contract
