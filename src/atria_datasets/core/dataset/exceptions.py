"""
Dataset Exceptions Module

This module defines custom exceptions for handling errors related to datasets
in the Atria application. These exceptions provide meaningful error messages
for issues such as missing dataset splits or configurations.

Classes:
    - SplitNotFoundError: Raised when a requested dataset split is not found.
    - ConfigurationNotFoundError: Raised when a requested dataset configuration is not found.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""


class SplitNotFoundError(ValueError):
    """
    Exception raised when a requested dataset split is not found.

    Attributes:
        split_value (str): The name of the missing dataset split.
    """

    def __init__(self, split_value: str):
        """
        Initializes the SplitNotFoundError.

        Args:
            split_value (str): The name of the missing dataset split.
        """
        super().__init__(f"Split '{split_value}' not found in the dataset.")


class ConfigurationNotFoundError(ValueError):
    """
    Exception raised when a requested dataset configuration is not found.

    Attributes:
        config_name (str): The name of the missing configuration.
        available_configs (list[str]): A list of available configurations.
    """

    def __init__(self, config_name: str, available_configs: list[str]):
        """
        Initializes the ConfigurationNotFoundError.

        Args:
            config_name (str): The name of the missing configuration.
            available_configs (list[str]): A list of available configurations.
        """
        super().__init__(
            f"Configuration '{config_name}' not found in the dataset. "
            f"Available configurations: {', '.join(available_configs)}"
        )
