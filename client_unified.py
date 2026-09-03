import flwr as fl
import torch
import numpy as np
from UTILS.model_utils import get_parameters, set_parameters
from UTILS.evaluation_metrics import binary_classification_metrics, loader_class_distribution

from disease_registry import DISEASES

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLIENTS = 2

print("=" * 50)
print("FEDERATED HEALTHCARE CLIENT")
print("=" * 50)

print("\nSelect Disease")

print("1. CKD")
print("2. Diabetes")
print("3. Heart Disease")
print("4. Breast Cancer")

choice = input("\nEnter Choice: ")

if choice not in DISEASES:
    print("Invalid Choice")
    exit()

CLIENT_ID = int(
    input("\nEnter Client ID (0 or 1): ")
)

selected = DISEASES[choice]

DiseaseModel = selected["model"]
loader = selected["loader"]

model = DiseaseModel().to(DEVICE)

trainloader, testloader = loader(
    CLIENT_ID,
    NUM_CLIENTS
)

print(
    f"\nLoaded {selected['name']} successfully"
)


def train(net, trainloader, epochs=10):

    net.train()

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=0.001
    )

    loss_fn = torch.nn.BCELoss()

    for _ in range(epochs):

        for X, y in trainloader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            outputs = net(X).view(-1)

            loss = loss_fn(
                outputs,
                y.view(-1)
            )

            loss.backward()

            optimizer.step()


def test(net, testloader):

    net.eval()

    loss_total = 0
    all_labels = []
    all_predictions = []

    loss_fn = torch.nn.BCELoss()

    with torch.no_grad():

        for X, y in testloader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            outputs = net(X).view(-1)

            loss = loss_fn(
                outputs,
                y.view(-1)
            )

            preds = outputs > 0.5

            loss_total += loss.item()
            all_labels.extend(y.view(-1).detach().cpu().numpy().astype(int))
            all_predictions.extend(preds.detach().cpu().numpy().astype(int))

    metrics = binary_classification_metrics(all_labels, all_predictions)
    metrics["loss"] = float(loss_total / len(testloader))
    return metrics


class FederatedClient(
    fl.client.NumPyClient
):

    def get_parameters(
        self,
        config
    ):
        return get_parameters(model)

    def fit(
        self,
        parameters,
        config
    ):

        set_parameters(
            model,
            parameters
        )

        train(
            model,
            trainloader,
            epochs=10
        )

        return (
            get_parameters(model),
            len(trainloader.dataset),
            {}
        )

    def evaluate(
        self,
        parameters,
        config
    ):

        set_parameters(
            model,
            parameters
        )

        metrics = test(
            model,
            testloader
        )
        metrics.update(loader_class_distribution(trainloader))
        metrics["client_id"] = CLIENT_ID

        print(
            f"[Client {CLIENT_ID}] "
            f"Loss={metrics['loss']:.4f} "
            f"Accuracy={metrics['accuracy']:.4f} "
            f"Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"Specificity={metrics['specificity']:.4f}"
        )

        return (
            float(metrics["loss"]),
            len(testloader.dataset),
            {key: float(value) for key, value in metrics.items()}
        )


fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FederatedClient()
)
