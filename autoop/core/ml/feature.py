class Feature:
    """Represents a feature, either categorical or numerical, within a dataset."""

    def __init__(self, name: str, column_type: str):
        """Initialize a feature with its name and type.

        Args:
            name (str): The feature's name.
            column_type (str): Type of the feature, either 'categorical' or 'numerical'.
        """
        self.name = name
        self.type = column_type

    def __str__(self) -> str:
        """Provide a string representation of the feature's name and type.

        Returns:
            str: Formatted string indicating the feature's name and type.
        """
        return f"{self.name} [{self.type}]"
