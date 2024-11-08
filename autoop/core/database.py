import json
import os
from typing import List, Optional, Tuple

from autoop.core.storage import Storage


class Database:
    """Represents a database for storing and managing data collections."""

    def __init__(self, storage: Storage):
        """Initialize the Database with a storage backend.

        Args:
            storage (Storage): Storage instance for data persistence.
        """
        self._storage = storage
        self._data = {}
        self._load()

    def set_data(self, collection: str, data_id: str, entry: dict) -> dict:
        """Store a data entry within a specified collection.

        Args:
            collection (str): Collection to hold the data entry.
            data_id (str): Unique identifier for the data entry.
            entry (dict): The data content to store.

        Returns:
            dict: The data entry that was stored.
        """
        assert isinstance(entry, dict), "Data should be provided as a dictionary."
        assert isinstance(collection, str), "Collection name must be a string."
        assert isinstance(data_id, str), "Data ID must be a string."

        if collection not in self._data:
            self._data[collection] = {}
        self._data[collection][data_id] = entry
        self._persist()
        return entry

    def get(self, collection: str, data_id: str) -> Optional[dict]:
        """Retrieve a data entry by its ID within a collection.

        Args:
            collection (str): The collection name.
            data_id (str): The ID of the data entry to retrieve.

        Returns:
            Optional[dict]: The data entry if it exists, otherwise None.
        """
        return self._data.get(collection, {}).get(data_id)

    def delete(self, collection: str, data_id: str) -> None:
        """Remove a specific data entry from a collection.

        Args:
            collection (str): Collection from which to delete the entry.
            data_id (str): ID of the entry to delete.
        """
        if self._data.get(collection) and data_id in self._data[collection]:
            del self._data[collection][data_id]
            self._persist()

    def data_list(self, collection: str) -> List[Tuple[str, dict]]:
        """Get a list of all entries within a collection.

        Args:
            collection (str): The collection to list entries from.

        Returns:
            List[Tuple[str, dict]]: List of (ID, data) pairs in the collection.
        """
        return list(self._data.get(collection, {}).items())

    def refresh(self) -> None:
        """Reload data from storage to refresh the database state."""
        self._load()

    def _persist(self) -> None:
        """Save the current state of all collections to the storage."""
        for collection, data in self._data.items():
            if not data:
                continue
            for data_id, item in data.items():
                self._storage.save(
                    json.dumps(item).encode(), f"{collection}{os.sep}{data_id}"
                )

        # Remove entries from storage if they are no longer in the database
        stored_keys = self._storage.list("")
        for key in stored_keys:
            collection, data_id = key.split(os.sep)[-2:]
            if collection not in self._data or data_id not in self._data[collection]:
                self._storage.delete(f"{collection}{os.sep}{data_id}")

    def _load(self) -> None:
        """Load all data entries from storage into the database."""
        self._data.clear()
        for key in self._storage.list(""):
            collection, data_id = key.split(os.sep)[-2:]
            data = self._storage.load(f"{collection}{os.sep}{data_id}")
            if collection not in self._data:
                self._data[collection] = {}
            self._data[collection][data_id] = json.loads(data.decode())
