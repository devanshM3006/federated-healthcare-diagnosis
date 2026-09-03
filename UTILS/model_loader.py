import os
import torch

from UTILS.model_factory import get_model
from UTILS.model_utils import set_parameters


def load_model(choice: str):
    """
    Load a trained model from the trained_models folder.
    """

    model = get_model(choice)

    filename = (
        model.__class__.__name__
        .replace("Net", "")
        .replace("Disease", "_disease")
        .lower()
        + ".pth"
    )

    model_path = os.path.join(
        "trained_models",
        filename
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu")
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model