from MODELS.ckd_model import CKDNet
from MODELS.diabetes_model import DiabetesNet
from MODELS.heartdisease_model import HeartDiseaseNet
from MODELS.breastcancer_model import BreastCancerNet

from UTILS.ckd_utils import load_data as load_ckd
from UTILS.diabetes_utils import load_data as load_diabetes
from UTILS.heartdisease_utils import load_data as load_heart
from UTILS.breastcancer_utils import load_data as load_breast


DISEASES = {

    "1": {
        "name": "CKD",
        "model": CKDNet,
        "loader": load_ckd
    },

    "2": {
        "name": "Diabetes",
        "model": DiabetesNet,
        "loader": load_diabetes
    },

    "3": {
        "name": "Heart Disease",
        "model": HeartDiseaseNet,
        "loader": load_heart
    },

    "4": {
        "name": "Breast Cancer",
        "model": BreastCancerNet,
        "loader": load_breast
    }
}