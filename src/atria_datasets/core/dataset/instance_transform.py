"""
Atria Dataset Module

This module defines the `AtriaDataset` class, which serves as a base class for datasets
used in the Atria application. It provides functionality for managing dataset splits,
configurations, metadata, and runtime transformations.

Classes:
    - AtriaDataset: Base class for datasets in the Atria application.

Dependencies:
    - torch: For tensor operations.
    - pathlib.Path: For handling file paths.
    - typing: For type annotations and generic types.
    - atria_core.logger: For logging utilities.
    - atria_core.utilities.file: For resolving file paths.
    - atria_core.utilities.repr: For rich object representations.
    - atria_datasets.core.datasets.config: For dataset configuration classes.
    - atria_datasets.core.datasets.exceptions: For custom dataset-related exceptions.
    - atria_datasets.core.datasets.metadata: For dataset metadata management.
    - atria_datasets.core.datasets.splits: For dataset split management.
    - atria_datasets.core.storage.dataset_storage_manager: For dataset storage management.
    - atria_datasets.core.transforms.base: For runtime data transformations.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from collections.abc import Callable
from typing import Any

from atria_core.logger import get_logger
from atria_core.types import BaseDataInstance
from atria_core.utilities.repr import RepresentationMixin

from atria_datasets.core.typing.common import T_BaseDataInstance

logger = get_logger(__name__)


class InstanceTransform(RepresentationMixin):
    def __init__(
        self,
        input_transform: Callable,
        data_model: T_BaseDataInstance,
        output_transform: Callable | None = None,
        apply_output_transform: bool = True,
    ):
        self._input_transform = input_transform
        self._data_model = data_model
        self._output_transform = output_transform
        self._apply_output_transform = apply_output_transform

    def __call__(self, index: int, sample: Any) -> BaseDataInstance:
        # apply input transformation
        data_instance: BaseDataInstance = self._input_transform(sample)

        # assert that the transformed instance is of the expected data model type
        assert isinstance(data_instance, self._data_model), (
            f"self._input_transform(sample) should return {self._data_model}, but got {type(data_instance)}"
        )

        # apply index
        if data_instance.index is None:
            data_instance.index = index

        # load the data instance from disk if needed
        data_instance.load()

        # yield the transformed data instance if output transform is enabled
        return (
            self._output_transform(data_instance)
            if self._output_transform is not None and self._apply_output_transform
            else data_instance
        )
