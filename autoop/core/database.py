import json
from typing import List, Optional, Tuple

from autoop.core.storage import Storage


class Database:
    """A lightweight database interface for structured storage and
    retrieval."""

    def __init__(self, storage: Storage) -> None:
        """Initialize the database with storage backend.

        Args:
            storage (Storage): Storage backend instance to persist data.
        """
        self._storage = storage
        self._data = {}
        self._load_data()

    def insert(self, collection: str, data_id: str, entry: dict) -> dict:
        """Insert or update an entry in a collection.

        Args:
            collection (str): Name of the collection.
            data_id (str): Unique identifier for the entry.
            entry (dict): Data to be stored.

        Returns:
            dict: The stored entry.
        """
        assert isinstance(entry, dict), "Entry must be a dictionary."
        if collection not in self._data:
            self._data[collection] = {}
        self._data[collection][data_id] = entry
        self._save_data()
        return entry

    def fetch(self, collection: str, data_id: str) -> Optional[dict]:
        """Retrieve an entry from a collection by its ID.

        Args:
            collection (str): Collection name.
            data_id (str): Entry identifier.

        Returns:
            Optional[dict]: Retrieved entry or None if it doesn't exist.
        """
        return self._data.get(collection, {}).get(data_id)

    def remove(self, collection: str, data_id: str) -> None:
        """Remove an entry from a collection.

        Args:
            collection (str): Collection name.
            data_id (str): Entry identifier.
        """
        if collection in self._data and data_id in self._data[collection]:
            del self._data[collection][data_id]
            self._save_data()

    def list_entries(self, collection: str) -> List[Tuple[str, dict]]:
        """Get all entries from a specific collection.

        Args:
            collection (str): Name of the collection.

        Returns:
            List[Tuple[str, dict]]: List of entries in the form (ID, entry).
        """
        return list(self._data.get(collection, {}).items())

    def reload(self) -> None:
        """Reload all data from storage."""
        self._load_data()

    def _save_data(self) -> None:
        """Persist data to storage backend."""
        for collection, items in self._data.items():
            for data_id, entry in items.items():
                self._storage.save(
                    json.dumps(entry).encode(), f"{collection}/{data_id}"
                )

        existing_keys = set(self._storage.list(""))
        memory_keys = {
            f"{coll}/{id}"
            for coll, items in self._data.items()
            for id in items
        }
        stale_keys = existing_keys - memory_keys
        for key in stale_keys:
            self._storage.delete(key)

    def _load_data(self) -> None:
        """Load data from storage backend into memory."""
        self._data = {}
        for key in self._storage.list(""):
            collection, data_id = key.rsplit("/", 2)[-2:]
            entry_data = self._storage.load(key)
            if collection not in self._data:
                self._data[collection] = {}
            self._data[collection][data_id] = json.loads(entry_data.decode())
