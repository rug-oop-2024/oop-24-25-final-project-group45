import os
from abc import ABC, abstractmethod
from glob import glob
from typing import List


class PathNotFoundError(Exception):
    """Custom error for handling missing paths in storage operations."""

    def __init__(self, path: str) -> None:
        """Raise error when a specified path is missing.

        Args:
            path (str): Path that was not found.
        """
        super().__init__(f"Specified path does not exist: {path}")


class Storage(ABC):
    """Abstract storage interface to define data operations."""

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """Store data at the designated path.

        Args:
            data (bytes): The binary data to be saved.
            path (str): Path where data will be stored.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """Retrieve data from a specific path.

        Args:
            path (str): Path to retrieve data from.

        Returns:
            bytes: Data read from the specified path.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """Remove data at the specified path.

        Args:
            path (str): Path to the data for deletion.
        """
        pass

    @abstractmethod
    def list(self, path: str) -> List[str]:
        """List all files under a directory path.

        Args:
            path (str): Directory path to search.

        Returns:
            List[str]: Filenames located under the specified path.
        """
        pass


class LocalStorage(Storage):
    """Local file-based storage for managing file operations."""

    def __init__(self, base_directory: str = "./data_storage") -> None:
        """Set up a local storage instance at the given base path.

        Args:
            base_directory (str, optional): Base directory path for storage.
                Defaults to "./data_storage".
        """
        self.base_directory = os.path.normpath(base_directory)
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)

    def save(self, data: bytes, filename: str) -> None:
        """Write binary data to a file in the base directory.

        Args:
            data (bytes): Data to be saved.
            filename (str): Filename within the base directory.
        """
        full_path = self._construct_path(filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as file:
            file.write(data)

    def load(self, filename: str) -> bytes:
        """Read binary data from a file.

        Args:
            filename (str): File name to read data from.

        Returns:
            bytes: Binary data stored in the file.
        """
        full_path = self._construct_path(filename)
        self._ensure_path_exists(full_path)
        with open(full_path, "rb") as file:
            return file.read()

    def delete(self, filename: str = "/") -> None:
        """Remove a file from the storage.

        Args:
            filename (str, optional): Filename to delete. Defaults to "/".
        """
        full_path = self._construct_path(filename)
        self._ensure_path_exists(full_path)
        os.remove(full_path)

    def list(self, directory_prefix: str = "/") -> List[str]:
        """Get a list of files in a directory under base storage.

        Args:
            directory_prefix (str): Directory prefix to start file search.

        Returns:
            List[str]: List of file paths relative to base directory.
        """
        full_path = self._construct_path(directory_prefix)
        self._ensure_path_exists(full_path)
        file_paths = glob(os.path.join(full_path, "**", "*"), recursive=True)
        return [
            os.path.relpath(path, self.base_directory)
            for path in file_paths
            if os.path.isfile(path)
        ]

    def _ensure_path_exists(self, path: str) -> None:
        """Validate existence of a given path.

        Args:
            path (str): Path to verify.

        Raises:
            PathNotFoundError: Raised if path is not found.
        """
        if not os.path.exists(path):
            raise PathNotFoundError(path)

    def _construct_path(self, path_fragment: str) -> str:
        """Build full path by combining base directory with a path fragment.

        Args:
            path_fragment (str): File or directory name to append to base.

        Returns:
            str: Full constructed path.
        """
        return os.path.normpath(os.path.join(self.base_directory,
                                             path_fragment))
