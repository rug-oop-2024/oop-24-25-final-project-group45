from typing import List

from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import LocalStorage, Storage

import streamlit as st


class ArtifactRegistry:
    """Handles the registration and management of artifacts."""

    def __init__(self, database: Database, storage: Storage):
        """Initialize ArtifactRegistry with storage and database.

        Args:
            database (Database): The database for metadata storage.
            storage (Storage): The storage for saving artifact files.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact):
        """Register an artifact in both storage and database.

        Args:
            artifact (Artifact): The artifact to be registered.
        """
        self._storage.save(artifact.data, artifact.asset_path)

        metadata_entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set_data("artifacts", artifact.id, metadata_entry)

    def list(self, artifact_type: str = None) -> List[Artifact]:
        """Retrieve a list of artifacts from the registry.

        Args:
            artifact_type (str, optional): Filter artifacts by type.
                Defaults to None.

        Returns:
            List[Artifact]: A list of registered artifacts.
        """
        artifact_entries = self._database.data_list("artifacts")
        artifacts = []
        for artifact_id, metadata in artifact_entries:
            if artifact_type and metadata["type"] != artifact_type:
                continue
            artifact = Artifact(
                name=metadata["name"],
                version=metadata["version"],
                asset_path=metadata["asset_path"],
                tags=metadata["tags"],
                metadata=metadata["metadata"],
                data=self._storage.load(metadata["asset_path"]),
                artifact_type=metadata["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """Fetch a specific artifact by ID.

        Args:
            artifact_id (str): The unique ID of the artifact.

        Returns:
            Artifact: The retrieved artifact.
        """
        metadata = self._database.get("artifacts", artifact_id)
        st.write(metadata)
        return Artifact(
            name=metadata["name"],
            version=metadata["version"],
            asset_path=metadata["asset_path"],
            tags=metadata["tags"],
            metadata=metadata["metadata"],
            data=self._storage.load(metadata["asset_path"]),
            artifact_type=metadata["type"],
        )

    def delete(self, artifact_id: str):
        """Remove an artifact from both storage and database.

        Args:
            artifact_id (str): The ID of the artifact to delete.
        """
        metadata = self._database.get("artifacts", artifact_id)
        self._storage.delete(metadata["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """Main class representing the AutoML system."""

    _instance = None

    def __init__(self, storage: LocalStorage, database: Database):
        """Initialize the AutoML system with storage and database.

        Args:
            storage (LocalStorage): The system's storage mechanism.
            database (Database): The system's metadata database.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance():
        """Return a singleton instance of AutoMLSystem.

        Returns:
            AutoMLSystem: The singleton instance of this class.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(LocalStorage("./assets/dbo")),
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self):
        """Access the artifact registry.

        Returns:
            ArtifactRegistry: The current artifact registry instance.
        """
        return self._registry
