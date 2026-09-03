import torch
from collections import OrderedDict

from disease_registry import DISEASES
from UTILS.model_factory import get_model
from UTILS.model_saver import save_model
import flwr as fl
from flwr.common import parameters_to_ndarrays
import matplotlib.pyplot as plt
import os
from datetime import datetime, timezone
from UTILS.model_utils import get_parameters, set_parameters
from ANALYTICS.results_store import append_results

print("=" * 50)
print("FEDERATED HEALTHCARE SERVER")
print("=" * 50)

print("\nSelect Disease")

print("1. CKD")
print("2. Diabetes")
print("3. Heart Disease")
print("4. Breast Cancer")

choice = input("\nEnter Choice: ")

disease_names = {
    "1": "ckd",
    "2": "diabetes",
    "3": "heart_disease",
    "4": "breast_cancer"
}

if choice not in disease_names:
    print("Invalid Choice")
    exit()

disease = disease_names[choice]

round_accuracies = []
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

RESEARCH_METRICS = ["loss", "accuracy", "precision", "recall", "f1", "specificity"]
COUNT_METRICS = ["tp", "tn", "fp", "fn", "num_samples", "samples", "positive_samples", "negative_samples"]


def weighted_average(metrics):

    accuracies = [
        num_examples * m["accuracy"]
        for num_examples, m in metrics
    ]

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    round_accuracy = (
        sum(accuracies)
        / total_examples
    )

    round_accuracies.append(round_accuracy)

    return {"accuracy": round_accuracy}


def aggregate_client_metrics(results):
    """Create a sample-weighted global summary from client evaluation results."""
    client_metrics = [result.metrics for _, result in results]
    total_examples = sum(result.num_examples for _, result in results)
    global_metrics = {"record_type": "global", "client_id": "global"}
    for name in RESEARCH_METRICS:
        global_metrics[name] = sum(
            result.num_examples * float(result.metrics.get(name, 0.0))
            for _, result in results
        ) / total_examples
    for name in COUNT_METRICS:
        global_metrics[name] = sum(float(metrics.get(name, 0.0)) for metrics in client_metrics)
    return global_metrics


class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if not results:
            return aggregated

        research_rows = []
        for client, result in results:
            row = dict(result.metrics)
            row.update({
                "round": server_round,
                "record_type": "client",
                "client_id": int(row.get("client_id", client.cid)),
                "num_samples": int(result.num_examples),
            })
            research_rows.append(row)

        global_row = aggregate_client_metrics(results)
        global_row["round"] = server_round
        research_rows.append(global_row)
        result_path = append_results(DISEASES[choice]["name"], research_rows, RUN_ID)
        print(f"Research metrics saved to: {result_path}")
        return aggregated

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):

        aggregated_result = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated_result is None:
            return None

        parameters, metrics = aggregated_result

        # Save only after the final round
        if server_round == 10:

            model = get_model(choice)

            parameter_ndarrays = parameters_to_ndarrays(
    parameters
)

            set_parameters(
                model,
                parameter_ndarrays
            )

            save_model(
                model,
                DISEASES[choice]["name"]
            )

        return aggregated_result

strategy = SaveModelStrategy(
    evaluate_metrics_aggregation_fn=weighted_average
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)

os.makedirs("GRAPHS", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(round_accuracies) + 1),
    round_accuracies,
    marker="o"
)

plt.title(
    f"Federated {disease.replace('_', ' ').title()} Accuracy"
)

plt.xlabel("Federated Training Round")
plt.ylabel("Global Model Accuracy")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"GRAPHS/{disease}_accuracy.png"
)

plt.show()
