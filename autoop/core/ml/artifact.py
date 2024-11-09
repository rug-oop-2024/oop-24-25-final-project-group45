import base64
from typing import Dict, List, Optional


class Artifact:
    """Represents a digital asset with metadata and data storage."""

    def __init__(
        self,
        name: str,
        data: bytes,
        artifact_type: Optional[str] = None,
        asset_path: Optional[str] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the artifact with its properties.

        Args:
            name (str): Name of the artifact.
            data (bytes): Binary data for the artifact.
            artifact_type (Optional[str]): Type or category of the artifact.
            asset_path (Optional[str]): Path for the artifact asset.
            version (Optional[str]): Version of the artifact.
            tags (Optional[List[str]]): Tags for categorizing the artifact.
            metadata (Optional[Dict[str, str]]): Metadata associated with
            the artifact.
        """
        self.metadata = metadata or {}
        self.name = name
        self.data = data
        self.type = artifact_type or "other"
        self.asset_path = asset_path or name
        self.version = version or "1.0.0"
        self.id = f"{self._encode_base64(self.asset_path)}-{self.version}"
        self.tags = tags or []

    def save_data(self, new_data: bytes) -> bytes:
        """Save new data to the artifact and return it.

        Args:
            new_data (bytes): Data to be saved.

        Returns:
            bytes: Saved data.
        """
        self.data = new_data
        return self.data

    def add_metadata(self, artifact: "Artifact") -> None:
        """Add metadata by storing another artifact's ID.

        Args:
            artifact (Artifact): Artifact whose metadata ID to store.
        """
        self.metadata[artifact.name] = artifact.id

    @staticmethod
    def _encode_base64(value: str) -> str:
        """Encode a string in base64 for ID generation.

        Args:
            value (str): String to encode.

        Returns:
            str: Base64 encoded string.
        """
        return base64.urlsafe_b64encode(value.encode()).decode()

    def read_data(self) -> bytes:
        """Get the artifact's data.

        Returns:
            bytes: The artifact's binary data.
        """
        return self.data
