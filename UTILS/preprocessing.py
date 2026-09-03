import torch


def preprocess_heart(request):
    """
    Convert HeartPredictionRequest into
    the tensor expected by HeartDiseaseNet.
    """

    features = [
        request.age,
        request.sex,
        request.cp,
        request.trestbps,
        request.chol,
        request.fbs,
        request.restecg,
        request.thalach,
        request.exang,
        request.oldpeak,
        request.slope,
        request.ca,
        request.thal,
    ]

    tensor = torch.tensor(
        [features],
        dtype=torch.float32
    )

    return tensor