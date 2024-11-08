from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Identify each column type in the dataset as either categorical or numerical.
    Columns with binary values (e.g., 0 and 1) are categorized as categorical.

    Args:
        dataset (Dataset): The dataset to evaluate.

    Returns:
        List[Feature]: A list of Feature objects, each containing the column's
        name and its inferred type.
    """
    data_frame = dataset.read()
    feature_list = []

    for col_name in data_frame.columns:
        if pd.api.types.is_numeric_dtype(data_frame[col_name]):
            unique_vals = data_frame[col_name].unique()
            if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
                col_type = "categorical"
            else:
                col_type = "numerical"
        else:
            col_type = "categorical"

        feature_list.append(Feature(col_name, col_type))

    return feature_list
