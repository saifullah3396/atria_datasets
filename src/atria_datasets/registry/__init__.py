"""
Registry Initialization Module

This module initializes the registry system for the Atria application. It imports
and initializes various registry groups from the `ModuleRegistry` class, making
them accessible as module-level constants. These registry groups are used to
manage datasets, data pipelines, data transformations, and other components
within the application.

The registry system provides a centralized way to register and retrieve components
such as datasets, models, transformations, and pipelines throughout the application.

Constants:
    DATASET: Registry group for dataset components
    DATA_PIPELINE: Registry group for data pipeline components
    DATA_TRANSFORM: Registry group for data transformation components
    BATCH_SAMPLER: Registry group for batch sampling components
    MODEL_PIPELINE: Registry group for model pipeline components
    MODEL: Registry group for model components
    TASK_PIPELINE: Registry group for task pipeline components
    METRIC: Registry group for metric components
    LR_SCHEDULER: Registry group for learning rate scheduler components
    OPTIMIZER: Registry group for optimizer components
    ENGINE: Registry group for engine components

Example:
    >>> from atria_registry import DATA_TRANSFORM, MODEL
    >>> # Register a new data transform
    >>> @DATA_TRANSFORM.register()
    >>> class MyTransform:
    ...     pass
    >>> # Get a registered model
    >>> model_cls = MODEL.get("my_model")

Dependencies:
    atria_registry.module_registry: Provides the ModuleRegistry class
    atria_registry.registry_group: Provides registry group classes

Author: Atria Development Team
Date: 2025-07-10
Version: 1.2.0
License: MIT
"""

from atria_registry import ModuleRegistry

from atria_datasets.registry.module_registry import init_registry
from atria_datasets.registry.registry_groups import (
    DatasetRegistryGroup,
    DataTransformRegistryGroup,
)

init_registry()

DATASET: DatasetRegistryGroup = ModuleRegistry().DATASET
"""Registry group for datasets.

Used to register and manage dataset-related components throughout the application.
Provides methods to register new datasets and retrieve existing ones by name.
"""

DATA_PIPELINE = ModuleRegistry().DATA_PIPELINE
"""Registry group for data pipelines.

Used to register and manage data pipeline components that handle data processing
workflows and transformations.
"""

DATA_TRANSFORM: DataTransformRegistryGroup = ModuleRegistry().DATA_TRANSFORM
"""Registry group for data transformations.

Used to register and manage data transformation components that modify or process
input data. Includes preprocessing, augmentation, and normalization operations.
"""

BATCH_SAMPLER = ModuleRegistry().BATCH_SAMPLER
"""Registry group for batch samplers.

Used to register and manage batch sampling strategies that determine how data
is grouped into batches during training and inference.
"""


__all__ = ["DATASET", "DATA_PIPELINE", "DATA_TRANSFORM", "BATCH_SAMPLER"]
