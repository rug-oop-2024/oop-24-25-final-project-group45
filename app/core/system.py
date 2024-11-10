from typing import List
import streamlit as st
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import LocalStorage, Storage


class ArtifactRegistry:
    """Manages the registration, retrieval, and deletion of artifacts."""

    def __init__(self, database: Database, storage: Storage):
        """
        Initializes the ArtifactRegistry with specified database and storage.

        Args:
            database (Database): Database instance used for
            recording artifact metadata.
            storage (Storage): Storage backend to handle
            artifact data persistence.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact):
        """
        Saves and registers a new artifact in the storage and database.

        Args:
            artifact (Artifact): The artifact instance to
            be saved and recorded.
        """
        self._storage.save(artifact.data, artifact.asset_path)
        entry_data = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.insert("artifacts", artifact.id, entry_data)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Retrieves a list of artifacts from the registry,
        optionally filtered by type.

        Args:
            type (str, optional): Filter artifacts by specified type
            (e.g., 'model', 'dataset').

        Returns:
            List[Artifact]: List of Artifact instances matching
            the specified type, if provided.
        """
        entries = self._database.list_entries("artifacts")
        artifacts = [
            Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                artifact_type=data["type"],
            )
            for artifact_id, data in entries if
            type is None or data["type"] == type
        ]
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Fetches a specific artifact from the registry using its unique ID.

        Args:
            artifact_id (str): The unique identifier of the desired artifact.

        Returns:
            Artifact: The artifact instance corresponding to the provided ID.
        """
        data = self._database.fetch("artifacts", artifact_id)
        st.write(data)  # Debugging/logging in Streamlit
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            artifact_type=data["type"],
        )

    def delete(self, artifact_id: str):
        """
        Removes a specified artifact from the registry and storage.

        Args:
            artifact_id (str): The unique identifier of the artifact
            to delete.
        """
        data = self._database.fetch("artifacts", artifact_id)
        if data:
            self._storage.remove(data["asset_path"])
            self._database.remove("artifacts", artifact_id)


class AutoMLSystem:
    """Central system managing artifact storage, registration, and
    retrieval."""

    _instance = None

    def __init__(self, storage: LocalStorage, database: Database):
        """
        Initializes the AutoML system with specified storage and database.

        Args:
            storage (LocalStorage): Storage backend for saving
            artifacts and metadata.
            database (Database): Database instance for managing
            artifact records.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Returns a singleton instance of the AutoMLSystem,
        initializing it if necessary.

        Returns:
            AutoMLSystem: The system instance for artifact management.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(LocalStorage("./assets/dbo")),
            )
        AutoMLSystem._instance._database.reload()
        return AutoMLSystem._instance

    @property
    def registry(self):
        """
        Provides access to the ArtifactRegistry for managing artifacts.

        Returns:
            ArtifactRegistry: The registry handling artifact operations.
        """
        return self._registry
