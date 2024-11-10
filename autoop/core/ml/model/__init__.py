from typing import TYPE_CHECKING

from autoop.core.ml.model.classification.decision_tree_classifier import (
    DecisionTreeModel,
)
from autoop.core.ml.model.classification.random_forest import (
    RandomForest,
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
    "RandomForest",
    "LogisticRegression",
    "DecisionTreeClassifier",
]


def get_model(model_name: str) -> "Model":
    """Get a model by name using this Factory Function."""
    model_classes = {
        "MultipleLinearRegression": MultipleLinearRegression,
        "Lasso": Lasso,
        "SupportVectorRegressor": SupportVectorRegressor,
        "RandomForest": RandomForest,
        "LogisticRegression": LogisticRegressor,
        "DecisionTreeClassifier": DecisionTreeModel,
    }

    return model_classes[model_name]()
