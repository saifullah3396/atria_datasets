"""
Datasets Module

This module serves as the entry point for importing various dataset-related classes and configurations
used in the system. It provides access to foundational classes for datasets, dataset configurations,
metadata, and dataset splits.

Exports:
    - AtriaDataset: The main class for managing datasets.
    - AtriaDatasetConfig: A class for configuring datasets.
    - DatasetMetadata: A class for managing metadata associated with datasets.
    - DatasetSplit: A class for representing dataset splits.
    - SplitConfig: A class for configuring dataset splits.

Dependencies:
    - atria_datasets.core.datasets.atria_dataset: For the `AtriaDataset` class.
    - atria_datasets.core.datasets.config: For the `AtriaDatasetConfig` class.
    - atria_datasets.core.datasets.metadata: For the `DatasetMetadata` class.
    - atria_datasets.core.datasets.splits: For the `DatasetSplit` and `SplitConfig` classes.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

__all__ = [
    "AtriaDataset",
    "AtriaDatasetConfig",
    "DatasetMetadata",
    "DatasetSplit",
    "SplitConfig",
    "AtriaIterableDataset",
    "AtriaIndexableDataset",
    "DownloadFileInfo",
]
