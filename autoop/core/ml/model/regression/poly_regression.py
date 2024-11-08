from autoop.core.ml.model import Model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import numpy as np


# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures.fit_transform
# Documentation: https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html

class PolynomialRegression(Model):
    def __init__(self, type):
        super().__init__(type)
        self._model = None
        self._degree = 2

    @property
    def degree(self):
        """
        Getter for the degree of the polynomial.
        """
        return self._degree

    @degree.setter
    def degree(self, degree: int):
        """
        Setter for the degree of the polynomial.
        """
        self._degree = degree

    def fit(
            self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> np.ndarray:
        """Fit the regression model with observations and ground truths.

        Args:
            observations (np.ndarray): Observations matrix (samples x variables).
            ground_truth (np.ndarray): Ground truths vector corresponding to X (samples).
        """
        self._validate_input(observations, ground_truth)

        X_val = self._preprocess_data(observations)

        self._model = LinearRegression()
        self._model.fit(X_val, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts observations based on the fitted model.

        Args:
            observations (np.ndarray): Observations matrix for prediction.

        Returns:
            np.ndarray: Predicted values for each observation in X.
        """
        self._validate_num_features(observations)

        X_val = self._preprocess_data(observations)

        predictions = self._model.predict(X_val)

        return predictions

    def _preprocess_data(self, observations: np.ndarray) -> np.ndarray:
        """
        Preprocess data to perform polynomial regression.

        Args:
            observations (np.ndarray): Observations to be preprocessed.

        Returns:
            observations_poly (np.ndarray): Preprocessed observations
        """
        poly = PolynomialFeatures(degree=self._degree)
        observations_poly = poly.fit_transform(observations)
        return observations_poly
