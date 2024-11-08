import io
import pickle
from typing import List, TYPE_CHECKING

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric

if TYPE_CHECKING:
    from autoop.core.ml.model import Model

from autoop.functional.feature import detect_feature_types
from autoop.functional.preprocessing import preprocess_features

import numpy as np
import pandas as pd

from exceptions import DatasetValidationError


class Pipeline:
    """Pipeline class for orchestrating data processing and model training."""

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: "Model",
        input_features: List[Feature],
        target_feature: Feature,
        split: float = 0.8,
    ) -> None:
        """Set up the pipeline with dataset, model, and necessary components.

        Args:
            metrics (List[Metric]): List of metrics for model evaluation.
            dataset (Dataset): Data to be processed.
            model (Model): Machine learning model to use.
            input_features (List[Feature]): List describing input features.
            target_feature (Feature): Feature to predict.
            split (float, optional): Fraction of data for training. Default is 0.8.

        Raises:
            ValueError: Raised if model type does not match target type.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (
            target_feature.type == "categorical"
            and model.type != "classification"
        ):
            raise ValueError(
                "Model should be classification for categorical targets."
            )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model should be regression for continuous targets."
            )

    def __str__(self) -> str:
        """Generate a string summary of key pipeline attributes."""
        return f"""
            Pipeline(
                model={self._model.type},\n
                input_features={list(map(str, self._input_features))},\n
                target_feature={str(self._target_feature)},\n
                split={self._split},\n
                metrics={list(map(str, self._metrics))}
        )
        """

    @property
    def model(self) -> "Model":
        """Return the model used in the current pipeline."""
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Retrieve artifacts created during pipeline execution."""
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type == "OneHotEncoder":
                data = pickle.dumps(artifact["encoder"])
                artifacts.append(Artifact(name=name, data=data))
            elif artifact_type == "StandardScaler":
                data = pickle.dumps(artifact["scaler"])
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "dataset": self._dataset,
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
            "metrics": self._metrics,
        }
        # Save training and testing metric results if available
        if hasattr(self, "_train_metrics_results"):
            pipeline_data.update(
                {
                    "train_results": self._train_metrics_results,
                    "test_results": self._metrics_results,
                }
            )

        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self.model.to_artifact(name=f"pipeline_model_{self.model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """Add an artifact to the registry by its name."""
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Handle preprocessing for input and target features.

        Steps:
        1. Process the target feature and save the artifact.
        2. Process the input features and save respective artifacts.
        3. Prepare data vectors for model training and evaluation.
        """
        target_feature_name, target_data, artifact = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for feature_name, _data, artifact in input_results:
            self._register_artifact(feature_name, artifact)

        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self) -> None:
        """Divide the data into training and testing sets based on split ratio."""
        split = self._split
        self._train_X = [
            vector[: int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)) :] for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            : int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)) :
        ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Concatenate multiple vectors into a single array.

        Args:
            vectors (List[np.array]): List of individual vectors to merge.

        Returns:
            np.array: Combined array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Train the model with the processed training data."""
        observations = self._compact_vectors(self._train_X)
        ground_truth = self._train_y
        self._model.fit(observations, ground_truth)

    def _evaluate(self) -> None:
        """Compute predictions and evaluate using specified metrics."""
        observations = self._compact_vectors(self._train_X)
        ground_truth = self._train_y
        self._train_metrics_results = []
        predictions = self._model.predict(observations)
        for metric in self._metrics:
            result = metric.evaluate(predictions, ground_truth)
            self._train_metrics_results.append((metric, result))

        observations = self._compact_vectors(self._test_X)
        ground_truth = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(observations)
        for metric in self._metrics:
            result = metric.evaluate(predictions, ground_truth)
            self._metrics_results.append((metric, result))
        encoder = list(self._artifacts[self._target_feature.name].values())[1]
        if encoder.__class__.__name__ == "StandardScaler":
            predictions = encoder.inverse_transform(predictions)
        self._predictions = predictions

    def execute(self) -> dict[str, list]:
        """Run the pipeline end-to-end and gather the results.

        Returns:
            dict[str, list]: Contains training/test metrics and predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "train_metrics": self._train_metrics_results,
            "test_metrics": self._metrics_results,
            "predictions": self._predictions,
        }

    def _validate_prediction_features(self, new_dataset: Dataset) -> None:
        """Ensure new data has matching features for making predictions.

        Args:
            new_data (Dataset): The new dataset to validate.

        Raises:
            DatasetValidationError: Raised if necessary features are missing or incorrect.
        """
        # Gather feature names and types from the provided data
        new_features = {
            feature.name: feature.type
            for feature in detect_feature_types(new_dataset)
        }

        # Track the expected feature names and types
        required_features = [feature.name for feature in self._input_features]
        required_types = {
            feature.name: feature.type for feature in self._input_features
        }

        # Verify if all expected features are present in new data
        missing_features = [
            feature
            for feature in required_features
            if feature not in new_features
        ]
        if missing_features:
            raise DatasetValidationError(missing_features=missing_features)

        # Check for any unexpected additional features
        extra_features = [
            feature
            for feature in new_features
            if feature not in required_features
        ]
        if extra_features:
            raise DatasetValidationError(extra_features=extra_features)

        # Ensure feature types match the expected types
        incorrect_types = {
            feature: (new_features[feature], expected_type)
            for feature, expected_type in required_types.items()
            if new_features[feature] != expected_type
        }
        if incorrect_types:
            raise DatasetValidationError(incorrect_types=incorrect_types)

    def _preprocess_prediction_columns(self, new_dataset: Dataset) -> None:
        """Reorder columns in new data to match training configuration.

        Args:
            new_data (Dataset): The dataset to reorder.
        """
        csv = new_dataset.data.decode()
        full_data = pd.read_csv(io.StringIO(csv))
        expected_column_order = [
            feature.name for feature in self._input_features
        ]
        # Arrange columns according to the expected sequence
        new_data_reordered = full_data[expected_column_order]
        new_dataset.data = new_data_reordered.to_csv(index=False).encode()
        input_results = preprocess_features(self._input_features, new_dataset)
        for feature_name, _data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def make_predictions(self, new_dataset: Dataset) -> np.ndarray:
        """Generate predictions on new data using the trained model.

        Args:
            new_data (Dataset): Dataset with data for prediction.

        Returns:
            np.ndarray: Predictions for the provided dataset.
        """
        # Verify feature alignment and preprocess data for predictions
        self._validate_prediction_features(new_dataset)
        self._preprocess_prediction_columns(new_dataset)

        observations = self._compact_vectors(self._input_vectors)
        encoder = self._artifacts[self._target_feature.name]
        predictions = self._model.predict(observations)
        if encoder.__class__.__name__ == "StandardScaler":
            # Apply inverse transform if applicable
            return encoder.inverse_transform(predictions)
        return predictions
