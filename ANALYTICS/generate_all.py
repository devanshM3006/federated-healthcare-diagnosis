"""Generate report-ready research analytics from persisted federated results.

Run from the project root with: ``python -m ANALYTICS.generate_all``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Allows table generation in a minimal environment.
    plt = None

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "RESULTS"
TABLES_DIR = RESULTS_DIR / "tables"
GRAPHS_DIR = ROOT / "GRAPHS"
DISEASES = {
    "CKD": "ckd",
    "Diabetes": "diabetes",
    "Heart Disease": "heart_disease",
    "Breast Cancer": "breast_cancer",
}
DATASET_FILES = {
    "CKD": "ckd.csv",
    "Diabetes": "diabetes.csv",
    "Heart Disease": "heart.csv",
    "Breast Cancer": "breast_cancer.csv",
}


def dataset_characteristics() -> pd.DataFrame:
    rows = []
    for disease, filename in DATASET_FILES.items():
        dataset = pd.read_csv(ROOT / "DATASETS" / filename)
        rows.append({
            "Disease": disease,
            "Samples": len(dataset),
            "Features": 24 if disease == "CKD" else len(dataset.columns) - 1 - (1 if disease == "Breast Cancer" else 0),
            "Classes": 2,
            "Source dataset": filename,
        })
    return pd.DataFrame(rows)


def latest_run(data: pd.DataFrame) -> pd.DataFrame:
    """Select the newest complete experiment run without destroying history."""
    if data.empty:
        return data
    data = data.copy()
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True)
    newest_run = data.sort_values("timestamp_utc").iloc[-1]["run_id"]
    return data[data["run_id"] == newest_run].copy()


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    """Write a Markdown table without adding a tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.fillna("").itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_performance(disease: str, records: pd.DataFrame) -> None:
    if plt is None:
        return
    final_round = records["round"].max()
    rows = records[records["round"] == final_round].copy()
    rows["Label"] = rows["record_type"].map({"client": "Client "}) + rows["client_id"].astype(str)
    rows.loc[rows["record_type"] == "global", "Label"] = "Global (weighted)"
    metrics = ["accuracy", "precision", "recall", "f1", "specificity"]
    values = rows.set_index("Label")[metrics].T
    ax = values.plot(kind="bar", figsize=(10, 6), ylim=(0, 1))
    ax.set_title(f"{disease}: Federated Performance at Round {int(final_round)}")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score (0–1)")
    ax.legend(title="Evaluation scope")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output = GRAPHS_DIR / disease.replace(" ", "_") / "performance_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


def plot_class_distribution(disease: str, records: pd.DataFrame) -> None:
    if plt is None:
        return
    clients = records[records["record_type"] == "client"].drop_duplicates("client_id")
    if clients.empty or not {"positive_samples", "negative_samples"}.issubset(clients.columns):
        return
    values = clients.set_index("client_id")[["negative_samples", "positive_samples"]]
    ax = values.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#4C78A8", "#F58518"])
    ax.set_title(f"{disease}: Simulated Client Training-Class Distribution")
    ax.set_xlabel("Simulated client")
    ax.set_ylabel("Training samples")
    ax.legend(["Negative class", "Positive class"])
    plt.tight_layout()
    output = GRAPHS_DIR / disease.replace(" ", "_") / "client_class_distribution.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


def client_correlation(disease: str, records: pd.DataFrame) -> dict[str, object]:
    clients = records[records["record_type"] == "client"]
    trajectory = clients.pivot_table(index="round", columns="client_id", values="accuracy", aggfunc="mean")
    row = {"Disease": disease, "Metric": "Accuracy across federated rounds", "Observations": len(trajectory), "Pearson r": "Not available", "Interpretation": "Insufficient varying paired rounds"}
    if trajectory.shape[1] == 2 and len(trajectory) >= 3 and trajectory.nunique().min() > 1:
        correlation = trajectory.corr().iloc[0, 1]
        row.update({"Pearson r": correlation, "Interpretation": "Client/model-behaviour correlation; not a medical correlation"})
        if plt is not None:
            ax = trajectory.plot(marker="o", figsize=(8, 5))
            ax.set_title(f"{disease}: Client Accuracy Trajectories")
            ax.set_xlabel("Federated round")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            output = GRAPHS_DIR / disease.replace(" ", "_") / "client_accuracy_trajectories.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output, dpi=160)
            plt.close()
    return row


def generate() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    characteristics = dataset_characteristics()
    characteristics.to_csv(TABLES_DIR / "dataset_characteristics.csv", index=False)
    summaries, client_rows, correlations, client_global_rows = [], [], [], []
    for disease, slug in DISEASES.items():
        path = RESULTS_DIR / f"{slug}_results.csv"
        if not path.exists():
            continue
        records = latest_run(pd.read_csv(path))
        if records.empty:
            continue
        final_round = records["round"].max()
        final = records[records["round"] == final_round]
        global_record = final[final["record_type"] == "global"]
        if not global_record.empty:
            summary = global_record.iloc[0][["accuracy", "precision", "recall", "f1", "specificity"]].to_dict()
            summaries.append({"Disease": disease} | summary)
        client_final = final[final["record_type"] == "client"].copy()
        client_final.insert(0, "Disease", disease)
        client_rows.append(client_final)
        accuracy_row = {"Disease": disease}
        for _, client in client_final.iterrows():
            accuracy_row[f"Client {int(client['client_id'])} Accuracy"] = client["accuracy"]
        if not global_record.empty:
            accuracy_row["Global Accuracy"] = global_record.iloc[0]["accuracy"]
        client_global_rows.append(accuracy_row)
        plot_performance(disease, records)
        plot_class_distribution(disease, records)
        correlations.append(client_correlation(disease, records))

    if summaries:
        performance = pd.DataFrame(summaries)
        performance.to_csv(TABLES_DIR / "overall_disease_performance.csv", index=False)
        write_markdown(performance, TABLES_DIR / "overall_disease_performance.md")
    if client_rows:
        client_comparison = pd.concat(client_rows, ignore_index=True)
        columns = ["Disease", "client_id", "samples", "positive_samples", "negative_samples", "loss", "accuracy", "precision", "recall", "f1", "specificity"]
        client_comparison = client_comparison[[column for column in columns if column in client_comparison]]
        client_comparison.to_csv(TABLES_DIR / "client_comparison.csv", index=False)
        write_markdown(client_comparison, TABLES_DIR / "client_comparison.md")
    if client_global_rows:
        client_global = pd.DataFrame(client_global_rows)
        client_global.to_csv(TABLES_DIR / "client_global_accuracy_comparison.csv", index=False)
        write_markdown(client_global, TABLES_DIR / "client_global_accuracy_comparison.md")
    if correlations:
        correlation_table = pd.DataFrame(correlations)
        correlation_table.to_csv(TABLES_DIR / "client_accuracy_correlations.csv", index=False)
        write_markdown(correlation_table, TABLES_DIR / "client_accuracy_correlations.md")

    limitations = """# Disease-correlation limitation

The CKD, diabetes, heart-disease, and breast-cancer training CSVs are independent
patient populations. They contain no shared patient identifier or common disease
indicators, so this project intentionally produces **no disease-to-disease medical
correlation** from them. A valid future comorbidity dataset must contain one row per
patient and `patient_id` plus binary `diabetes`, `heart_disease`, `ckd`, and
`breast_cancer` columns. `ANALYTICS.disease_correlation.analyse_comorbidity_dataset`
validates that structure before calculating a Pearson association matrix. Correlation
does not imply causation.
"""
    (TABLES_DIR / "disease_correlation_limitation.md").write_text(limitations, encoding="utf-8")
    print(f"Analytics generated in: {TABLES_DIR}")
    if plt is None:
        print("Chart generation skipped: matplotlib is not installed in this interpreter.")


if __name__ == "__main__":
    generate()
