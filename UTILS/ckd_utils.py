import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def load_data(client_id, num_clients=2):

    # Load dataset
    df = pd.read_csv("DATASETS/ckd.csv")

    # The source file includes one structurally empty trailing column.
    # It is retained in the CSV for data fidelity but excluded from training.
    df = df.drop(columns=["unused_empty"])

    # Replace missing values
    df.replace("?", pd.NA, inplace=True)

    # Convert columns where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # Fill missing values
    for col in df.columns:

        if col == "classification":
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns
    for col in df.columns:

        if col == "classification":
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.factorize(df[col])[0]

    # Encode target column
    df["classification"] = (
        df["classification"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["classification"] = df["classification"].map({
        "ckd": 1,
        "notckd": 0
    })

    df = df.dropna(subset=["classification"])

    print("\nTarget Distribution:")
    print(df["classification"].value_counts())

    # Features and labels
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float).reshape(-1, 1)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Standardize
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Federated split
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
