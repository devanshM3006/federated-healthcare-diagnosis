from disease_registry import DISEASES


def get_model(choice: str):
    """
    Returns an instance of the selected disease model.
    """

    if choice not in DISEASES:
        raise ValueError(f"Invalid disease choice: {choice}")

    model_class = DISEASES[choice]["model"]

    return model_class()