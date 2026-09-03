"""Append-only storage for federated research measurements."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

RESULT_COLUMNS = [
    "run_id", "timestamp_utc", "disease", "round", "record_type", "client_id",
    "num_samples", "samples", "positive_samples", "negative_samples", "loss",
    "accuracy", "precision", "recall", "f1", "specificity", "tp", "tn", "fp", "fn",
]


def normalise_disease_name(disease: str) -> str:
    return disease.lower().replace(" ", "_")


def append_results(disease: str, rows: list[Mapping[str, object]], run_id: str) -> Path:
    """Append rows rather than overwriting experiment history."""
    results_dir = Path("RESULTS")
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"{normalise_disease_name(disease)}_results.csv"
    write_header = not path.exists()
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({
                column: row.get(column, "") for column in RESULT_COLUMNS
            } | {"run_id": run_id, "timestamp_utc": timestamp, "disease": disease})
    return path
