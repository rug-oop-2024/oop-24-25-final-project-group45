import pickle
from typing import List

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.core.ml.model import Model
from autoop.functional.preprocessing import preprocess_features

import numpy as np


class Pipeline:
    """A class that integrates data processing and modeling components."""

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split_ratio=0.8,
    ):
        """Initialize the pipeline with dataset, model, and configuration.

        Args:
            metrics (List[Metric]): Metrics used for evaluation.
            dataset (Dataset): The input data for processing.
            model (Model): The model to train on the data.
            input_features (List[Feature]): List specifying features.
            target_feature (Feature): The feature to predict.
            split_ratio (float, optional): Proportion for training data.
                Defaults to 0.8.

        Raises:
            ValueError: If an incompatible model-target type pairing is used.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split_ratio
        if (
            target_feature.type == "categorical"
            and model.type != "classification"
        ):
            raise ValueError(
                "Model must be classification type for categorical targets."
            )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model must be regression type for continuous targets."
            )

    def __str__(self) -> str:
        """Represent key attributes of the pipeline as a string."""
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
    def model(self) -> Model:
        """Get the model used in this Pipeline."""
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Return artifacts generated during execution."""
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
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """Register an artifact in the pipeline by name."""
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Prepare the input and target features from the dataset.

        Steps:
        1. Preprocess target feature and store artifact.
        2. Preprocess input features and store artifacts.
        3. Store data vectors for training and evaluation.
        """
        target_feature_name, target_data, artifact = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)

        self._output_vector = target_data
        self._input_vectors = [data for _, data, _ in input_results]

    def _split_data(self) -> None:
        """Divide the data into training and testing sets based on split."""
        split = self._split
        self._train_X = [
            vector[: int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)) :] for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[: int(split * len(self._output_vector))]
        self._test_y = self._output_vector[int(split * len(self._output_vector)) :]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Merge a list of vectors into one array."""
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Train the model using the prepared training data."""
        observations = self._compact_vectors(self._train_X)
        ground_truth = self._train_y
        self._model.fit(observations, ground_truth)

    def _evaluate(self) -> None:
        """Evaluate model performance and calculate metric results."""
        observations_train = self._compact_vectors(self._train_X)
        ground_truth_train = self._train_y
        self._train_metrics_results = []
        predictions_train = self._model.predict(observations_train)
        for metric in self._metrics:
            result = metric.evaluate(predictions_train, ground_truth_train)
            self._train_metrics_results.append((metric, result))

        observations_test = self._compact_vectors(self._test_X)
        ground_truth_test = self._test_y
        self._metrics_results = []
        predictions_test = self._model.predict(observations_test)
        for metric in self._metrics:
            result = metric.evaluate(predictions_test, ground_truth_test)
            self._metrics_results.append((metric, result))
        self._predictions = predictions_test

    def execute(self) -> dict[str, list]:
        """Run the pipeline steps and gather the results.

        Returns:
            dict[str, list]: Contains training and test metrics, and predictions.
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