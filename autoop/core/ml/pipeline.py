import io
import pickle
from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features

if TYPE_CHECKING:
    from autoop.core.ml.model import Model


class Pipeline:
    """A pipeline class to manage data processing and model training."""

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: "Model",
        input_features: List[Feature],
        target_feature: Feature,
        split: float = 0.8,
    ) -> None:
        """Initialize Pipeline with model, dataset, and configuration."""
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split

        self._validate_model_type()

    def __str__(self) -> str:
        """Represent the Pipeline configuration as a string."""
        return (
            f"Pipeline(model={self._model.type}, input_features="
            f"{list(map(str, self._input_features))}, "
            f"target_feature={str(self._target_feature)}, "
            f"split={self._split}, "
            f"metrics={list(map(str, self._metrics))})"
        )

    def _validate_model_type(self):
        """Check that model type is compatible with the target feature type."""
        if self._target_feature.type == "categorical":
            if self._model.type != "classification":
                raise ValueError(
                    "Model type must be classification for "
                    "categorical target feature"
                )
        elif self._target_feature.type == "continuous":
            if self._model.type != "regression":
                raise ValueError(
                    "Model type must be regression for continuous target "
                    "feature"
                )

    @property
    def model(self) -> "Model":
        """Returns model."""
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Collect and return artifacts generated during pipeline execution."""
        artifacts = []
        for name, artifact in self._artifacts.items():
            data = pickle.dumps(
                artifact.get(
                    "encoder"
                    if "OneHotEncoder" in artifact["type"]
                    else "scaler"
                )
            )
            artifacts.append(Artifact(name=name, data=data))
        artifacts.extend(
            [
                Artifact(
                    name="pipeline_config",
                    data=pickle.dumps(self._get_pipeline_data()),
                ),
                self.model.to_artifact(
                    name=f"pipeline_model_{self.model.type}"
                ),
            ]
        )
        return artifacts

    def _get_pipeline_data(self) -> dict:
        """Return core pipeline configuration as a dictionary."""
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
        return pipeline_data

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Prepare input and target features from the dataset for
        model training.
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
        self._input_vectors = [data for (_, data, _) in input_results]

    def _split_data(self) -> None:
        """Split the data into training and testing sets."""
        split_index = int(self._split * len(self._output_vector))
        self._train_X, self._test_X = [
            v[:split_index] for v in self._input_vectors
        ], [v[split_index:] for v in self._input_vectors]
        self._train_y, self._test_y = (
            self._output_vector[:split_index],
            self._output_vector[split_index:],
        )

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Train the model on training data."""
        observations, ground_truth = (
            self._compact_vectors(self._train_X),
            self._train_y,
        )
        self._model.fit(observations, ground_truth)

    def _evaluate(self) -> None:
        """Evaluate the model and collect results."""
        self._train_metrics_results, self._metrics_results = [], []

        train_predictions = (self._model.predict
                             (self._compact_vectors(self._train_X)))
        for metric in self._metrics:
            self._train_metrics_results.append(
                (metric, metric.compute(train_predictions, self._train_y))
            )

        test_predictions = self._model.predict(self._compact_vectors
                                               (self._test_X))
        for metric in self._metrics:
            self._metrics_results.append(
                (metric, metric.compute(test_predictions, self._test_y))
            )

        encoder = list(self._artifacts[self._target_feature.name].values())[1]
        self._predictions = (
            encoder.inverse_transform(test_predictions)
            if encoder.__class__.__name__ == "StandardScaler"
            else test_predictions
        )

    def execute(self) -> dict[str, list]:
        """
        Execute pipeline steps to process data and return
        evaluation results.
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

    def _reorder_columns(self, new_dataset: Dataset) -> None:
        """Reorder columns of a dataset to match expected order."""
        expected_columns = [feature.name for feature in self._input_features]
        new_data_df = pd.read_csv(io.StringIO(new_dataset.data.decode()))[
            expected_columns
        ]
        new_dataset.data = new_data_df.to_csv(index=False).encode()
        input_results = preprocess_features(self._input_features, new_dataset)
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        self._input_vectors = [data for (_, data, _) in input_results]

    def make_predictions(self, new_dataset: Dataset) -> np.ndarray:
        """Predict on new data after checking column order."""
        self._reorder_columns(new_dataset)
        observations = self._compact_vectors(self._input_vectors)
        encoder = self._artifacts[self._target_feature.name]
        predictions = self._model.predict(observations)
        return (
            encoder.inverse_transform(predictions)
            if encoder.__class__.__name__ == "StandardScaler"
            else predictions
        )
