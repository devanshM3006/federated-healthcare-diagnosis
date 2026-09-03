"""Scientifically constrained future comorbidity-dataset analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"patient_id", "diabetes", "heart_disease", "ckd", "breast_cancer"}


def analyse_comorbidity_dataset(path: str | Path) -> pd.DataFrame:
    """Return disease-status Pearson correlations from a valid shared-patient dataset.

    This function deliberately rejects the four independent training datasets:
    every row must represent one patient and include each binary disease status.
    Correlation describes association only; it never establishes causation.
    """
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            "A comorbidity dataset is required. Missing columns: " + ", ".join(sorted(missing))
        )
    if data["patient_id"].duplicated().any():
        raise ValueError("patient_id must be unique for a patient-level correlation analysis.")
    disease_columns = sorted(REQUIRED_COLUMNS - {"patient_id"})
    values = data[disease_columns]
    if not values.isin([0, 1]).all().all():
        raise ValueError("Disease indicator columns must contain binary 0/1 values.")
    return values.corr(method="pearson")
