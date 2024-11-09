from autoop.core.ml.artifact import Artifact
from abc import ABC, abstractmethod
import pandas as pd
import io


class Dataset(Artifact):
    """A class to represent an ML dataset"""
    def __init__(self, *args, **kwargs):
        super().__init__(artifact_type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str,
                       asset_path: str, version: str = "1.0.0"):
        """ Create a dataset from a pandas dataframe."""
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read_df(self) -> pd.DataFrame:
        """ Function that reads data in bytes """
        bytes = super().read_data()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save_df(self, data: pd.DataFrame) -> bytes:
        """ Function that saves data in bytes"""
        bytes = data.to_csv(index=False).encode()
        return super().save_data(bytes)
