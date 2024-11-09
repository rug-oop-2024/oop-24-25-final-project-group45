from typing import TYPE_CHECKING

from autoop.core.ml.model.classification.decision_tree_classifier import (
    DecisionTreeModel,
)
from autoop.core.ml.model.classification.k_nearest_neighbors import (
    KNearestNeighbors,
)
from autoop.core.ml.model.classification.logistic_regression import (
    LogisticRegressor,
)

if TYPE_CHECKING:
    from autoop.core.ml.model.model import Model

from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.support_vector_regression import (
    SupportVectorRegressor,
)

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "Lasso",
    "SupportVectorRegressor",
]

CLASSIFICATION_MODELS = [
    "KNearestNeighbors",
    "LogisticRegression",
    "DecisionTreeClassifier",
]


def get_model(model_name: str) -> "Model":
    """Get a model by name using this Factory Function."""
    match model_name:
        case "MultipleLinearRegression":
            return MultipleLinearRegression()
        case "Lasso":
            return Lasso()
        case "SupportVectorRegressor":
            return SupportVectorRegressor()
        case "KNearestNeighbors":
            return KNearestNeighbors()
        case "LogisticRegression":
            return LogisticRegressor()
        case "DecisionTreeClassifier":
            return DecisionTreeModel()
    raise ValueError(f"Model {model_name} doesn't exist.")
