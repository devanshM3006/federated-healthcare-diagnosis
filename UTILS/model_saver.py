import os
import torch


def save_model(model, disease_name):
    """
    Save a trained PyTorch model.
    """

    os.makedirs("trained_models", exist_ok=True)

    filename = (
        disease_name.lower()
        .replace(" ", "_")
        + ".pth"
    )

    save_path = os.path.join(
        "trained_models",
        filename
    )

    torch.save(
        model.state_dict(),
        save_path
    )

    print(f"\nModel saved successfully:")
    print(save_path)