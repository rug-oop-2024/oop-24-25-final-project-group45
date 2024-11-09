import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any
import numpy as np
from autoop.core.ml.artifact import Artifact


class Model(ABC):
    """Abstract base class for models used in this project."""

    def __init__(self):
        """Initialize abc"""
        self._parameters = {}
        self._type = None
        self._n_features = None
        self._fitted = False

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        """Retrieve model parameters as a dictionary.

        Returns:
            dict[str, np.ndarray]: A deep copy of model parameters.
        """
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, new_params: dict[str, Any]) -> None:
        """Set and validate new parameters for the model.

        Args:
            new_params (dict[str, np.ndarray]): Parameters to update.

        Raises:
            TypeError: If the parameters are not provided as a dictionary.
        """
        if not isinstance(new_params, dict):
            raise TypeError("Parameters must be in dictionary format.")
        for key in new_params.keys():
            self._validate_key(key)
        self._parameters.update(new_params)

    @property
    def type(self) -> str:
        """Return model type"""
        return self._type

    @type.setter
    def type(self, model_type: str) -> None:
        """Set the type of the model, ensuring it is a string.

        Args:
            model_type (str): The type of the model.

        Raises:
            TypeError: If model_type is not a string.
        """
        if not isinstance(model_type, str):
            raise TypeError("Model type must be a string.")
        self._type = model_type

    def _validate_key(self, key: str) -> None:
        """Check that parameter keys are valid strings.

        Args:
            key (str): Key to validate.

        Raises:
            TypeError: If key is not a string.
        """
        if not isinstance(key, str):
            raise TypeError("Parameter keys must be strings.")

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """Train the model on observations and ground_truths.

        Args:
            observations (np.ndarray): Training data.
            ground_truths (np.ndarray): Labels corresponding to the data.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Generate predictions from the model.

        Args:
            observations (np.ndarray): Data to predict.

        Returns:
            np.ndarray: Predictions.
        """
        pass

    def to_artifact(self, name: str) -> Artifact:
        """Convert the model into an artifact.

        Args:
            name (str): Name of the artifact.

        Returns:
            Artifact: Serialized model as an artifact.
        """
        model_data = {
            "parameters": self.parameters,
            "features": self._n_features,
            "fitted": self._fitted,
            "model": self.__class__.__name__,
        }
        return Artifact(
            name=name, data=pickle.dumps(model_data), artifact_type="model"
        )
