"""Reguły przygotowania danych wynikające z EDA Iranian Churn."""

import numpy as np
import pandas as pd
import pytest

from asi_project.data import filter_training_rows, normalize_input


@pytest.mark.parametrize("text_dtype", ["object", "string"])
def test_normalization_is_inference_safe_for_dirty_records(text_dtype):
    """Brak targetu, spacje i braki nie zmieniają wyniku zależnie od wielkości partii."""
    raw = pd.DataFrame(
        {
            "Charge Amount": pd.Series([" 10 ", " \t ", "0"], dtype=text_dtype),
            "Status": pd.Series([" 2", None, "1 "], dtype=text_dtype),
            "Seconds of Use": [0, 1182, 3530],
        }
    )
    original = raw.copy(deep=True)
    expected = pd.DataFrame(
        {
            "Charge Amount": ["10", np.nan, "0"],
            "Status": ["2", raw.at[1, "Status"], "1"],
            "Seconds of Use": [0, 1182, 3530],
        }
    )

    normalized = normalize_input(raw)

    pd.testing.assert_frame_equal(raw, original)
    pd.testing.assert_frame_equal(normalized, expected, check_dtype=False)
    pd.testing.assert_frame_equal(normalize_input(normalized), normalized, check_dtype=False)
    for position in range(len(raw)):
        pd.testing.assert_frame_equal(
            normalize_input(raw.iloc[[position]]), normalized.iloc[[position]], check_dtype=False
        )


@pytest.mark.parametrize("missing_target", [None, np.nan, pd.NA, "", " \t\n "])
def test_training_filter_removes_missing_target_without_prior_normalization(
    registry, missing_target
):
    """Także pusta etykieta nie może trafić do treningu jako dodatkowa klasa."""
    contract = registry["iranian_churn"]
    raw = pd.DataFrame(
        {"Seconds of Use": [0, 100, 200], "Churn": [" 0 ", missing_target, " 1 "]},
        index=[10, 20, 30],
    )
    original = raw.copy(deep=True)

    filtered = filter_training_rows(raw, contract)

    pd.testing.assert_frame_equal(filtered, raw.loc[[10, 30]])
    pd.testing.assert_frame_equal(raw, original)


def test_training_filter_keeps_duplicates_conflicts_zeros_and_missing_features(registry):
    """Filtrowanie nie usuwa poprawnego targetu z powodu identycznych lub niepełnych cech."""
    contract = registry["iranian_churn"]
    raw = pd.DataFrame(
        {
            "Seconds of Use": [0, 0, 120, 120, np.nan],
            "Charge Amount": [10, 10, 0, 0, 1],
            "Churn": [0, 0, 0, 1, 1],
        },
        index=[10, 20, 30, 40, 50],
    )
    original = raw.copy(deep=True)

    filtered = filter_training_rows(raw, contract)

    pd.testing.assert_frame_equal(filtered, original)
    pd.testing.assert_frame_equal(raw, original)
