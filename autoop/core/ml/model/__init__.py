from autoop.core.ml.model.model import Model

from autoop.core.ml.model.regression.multiple_regression import MultipleLinearRegression
from autoop.core.ml.model.regression.lasso_wrap import LassoWrapper
from autoop.core.ml.model.regression.poly_regression import PolynomialRegression

from autoop.core.ml.model.classification.knn import KNearestNeighbors
from autoop.core.ml.model.classification.logistic_regression import MultinomialLogisticRegression
from autoop.core.ml.model.classification.svc_linear import SVCLinear

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "LassoWrapper",
    "PolynomialRegression"
]

CLASSIFICATION_MODELS = [
    "KNearestNeighbors",
    "LogisticRegression",
    "LinearSVC"
]


def get_model(model_name: str) -> Model:
    """Get a model by name using this Factory Function."""
    match model_name:
        case "MultipleLinearRegression":
            return MultipleLinearRegression()
        case "KNearestNeighbors":
            return KNearestNeighbors()
        case "LogisticRegression":
            return MultinomialLogisticRegression
        case "LinearSVC":
            return SVCLinear
        case "LassoWrapper":
            return LassoWrapper
        case "PolynomialRegression":
            return PolynomialRegression
    raise ValueError(f"Model {model_name} doesn't exist.")
