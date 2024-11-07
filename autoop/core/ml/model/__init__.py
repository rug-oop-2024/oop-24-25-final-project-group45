from autoop.core.ml.model.classification.knn import (
    KNearestNeighbors,
)

from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_regression import (
    MultipleLinearRegression,
)

REGRESSION_MODELS = [
    "MultipleLinearRegression",
]  # add your models as str here

CLASSIFICATION_MODELS = [
    "KNearestNeighbors",
]  # add your models as str here


def get_model(model_name: str) -> Model:
    """Get a model by name using this Factory Function."""
    match model_name:
        case "MultipleLinearRegression":
            return MultipleLinearRegression()
        case "KNearestNeighbors":
            return KNearestNeighbors()
    raise ValueError(f"Model {model_name} doesn't exist.")
