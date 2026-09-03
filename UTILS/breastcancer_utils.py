import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(client_id, num_clients=2):

    # Load breast cancer dataset
    df = pd.read_csv("DATASETS/breast_cancer.csv")

    # Features (skip ID column and label column)
    X = df.drop(columns=["id", "diagnosis"]).values

    # Labels
    y = (df["diagnosis"] == "M").astype(int).values.reshape(-1, 1)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Split data among clients
    chunk_size = len(X_train) // num_clients

    start = client_id * chunk_size

    end = (
        (client_id + 1) * chunk_size
        if client_id < num_clients - 1
        else len(X_train)
    )

    X_client = X_train[start:end]
    y_client = y_train[start:end]

    trainset = TensorDataset(
        torch.tensor(X_client, dtype=torch.float32),
        torch.tensor(y_client, dtype=torch.float32)
    )

    testset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    return (
        DataLoader(trainset, batch_size=16, shuffle=True),
        DataLoader(testset, batch_size=16)
    )
