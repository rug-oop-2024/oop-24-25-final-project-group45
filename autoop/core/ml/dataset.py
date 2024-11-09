from autoop.core.ml.artifact import Artifact

import io
import pandas as pd


class Dataset(Artifact):
    """Represent a machine learning dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize the dataset as a type of Artifact."""
        super().__init__(artifact_type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ):
        """Generate a Dataset instance from a pandas DataFrame."""
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read_df(self) -> pd.DataFrame:
        """Read and decode data into a DataFrame."""
        data_bytes = super().read_data()
        csv = data_bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save_df(self, data: pd.DataFrame) -> bytes:
        """Encode and save DataFrame as bytes."""
        data_bytes = data.to_csv(index=False).encode()
        return super().save_data(data_bytes)
