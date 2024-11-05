from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detect the type of each feature in the dataset as either categorical or numerical.
    Binary numeric columns are considered categorical.

    Args:
        dataset (Dataset): The dataset to analyze.

    Returns:
        List[Feature]: A list of Feature objects, each with a name and a detected type.
    """
    df = dataset.read()

    features = []

    for column_name in df.columns:
        if pd.api.types.is_numeric_dtype(df[column_name]):
            # Check if it's binary (two unique values, e.g., 0 and 1)
            unique_values = df[column_name].unique()
            if len(unique_values) == 2 and sorted(unique_values) in [[0, 1], [1, 0]]:
                feature_type = "categorical"  # Treat binary numeric as categorical
            else:
                feature_type = "numerical"
        else:
            feature_type = "categorical"

        # Create a Feature object with the detected type
        feature = Feature(name=column_name, type=feature_type)
        features.append(feature)

    return features
