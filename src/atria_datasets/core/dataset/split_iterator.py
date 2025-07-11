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

import inspect
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

import rich
import rich.pretty
from atria_core.logger import get_logger
from atria_core.types import DatasetSplitType
from atria_core.utilities.repr import RepresentationMixin

from atria_datasets.core.dataset.instance_transform import InstanceTransform
from atria_datasets.core.typing.common import T_BaseDataInstance

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)


class SplitIterator(Sequence[T_BaseDataInstance], RepresentationMixin):
    def __init__(
        self,
        split: DatasetSplitType,
        base_iterator: Sequence | Generator,
        data_model: type[T_BaseDataInstance],
        input_transform: Callable,
        output_transform: Callable | None = None,
        max_len: int | None = None,
    ):
        self._split = split
        self._base_iterator = base_iterator
        self._max_len = max_len
        self._apply_output_transform: bool = True
        self._subset_indices: list[int] | None = None
        self._tf = InstanceTransform(
            input_transform=input_transform,
            data_model=data_model,
            output_transform=output_transform,
            apply_output_transform=self._apply_output_transform,
        )
        self._is_generator = inspect.isgeneratorfunction(base_iterator)
        if not self._is_generator:
            assert hasattr(self._base_iterator, "__len__"), (
                f"T he base iterator {self._base_iterator} must implement __len__ to support indexing. "
            )
            self._suports_indexing = hasattr(self._base_iterator, "__getitem__")
            self._supports_multi_indexing = hasattr(self._base_iterator, "__getitems__")
            assert self._suports_indexing or self._supports_multi_indexing, (
                f"The base iterator {self._base_iterator} must implement either __getitem__ or __getitems__ "
            )

    @property
    def split(self) -> DatasetSplitType:
        return self._split

    @property
    def base_iterator(self) -> Iterable:
        return self._base_iterator

    @property
    def input_transform(self) -> Callable:
        return self._tf._input_transform

    @property
    def output_transform(self) -> Callable | None:
        return self._tf._output_transform

    @output_transform.setter
    def output_transform(self, value: Callable) -> None:
        self._tf._output_transform = value

    @property
    def subset_indices(self) -> list[int] | None:
        """
        Returns the subset indices if available, otherwise None.
        """
        return self._subset_indices

    @subset_indices.setter
    def subset_indices(self, indices: list[int]) -> None:
        """
        Sets the subset indices for the iterator.

        Args:
            indices (list[int]): A list of indices to set as the subset.
        """
        self._subset_indices = indices

    @property
    def data_model(self) -> T_BaseDataInstance:
        return self._tf._data_model

    def disable_output_transform(self) -> None:
        self._apply_output_transform = False

    def enable_output_transform(self) -> None:
        self._apply_output_transform = True

    def dataframe(self) -> "pd.DataFrame":
        """
        Displays the dataset split information in a rich format.
        """
        if hasattr(self._base_iterator, "dataframe"):
            return self._base_iterator.dataframe()
        else:
            raise RuntimeError(
                "This dataset is not backed by a DataFrame or does not support dataframe representation."
            )

    def __iter__(self) -> Iterator[T_BaseDataInstance]:
        if self._is_generator:
            if self._subset_indices is not None:
                raise RuntimeError(
                    "You are trying to iterate over a subset of the dataset, "
                    "but the base iterator does not support indexing. "
                )

            #
            for index, sample in enumerate(self._base_iterator):
                yield self._tf(index, sample)

        # fall back to sequence iteration of self
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> T_BaseDataInstance:  # type: ignore[override]
        assert self._suports_indexing, (
            "The base iterator does not support multi-indexing. "
            "Please use __getitem__ for single index access."
        )
        if self._subset_indices is not None:
            index = self._subset_indices[index]
        return self._tf(index, self._base_iterator[index])  # type: ignore

    def __getitems__(self, indices: list[int]) -> list[T_BaseDataInstance]:  # type: ignore
        if self._subset_indices is not None:
            indices = [self._subset_indices[idx] for idx in indices]
        if self._supports_multi_indexing:
            data_instances = self._base_iterator[indices]  # type: ignore
        else:
            assert self._suports_indexing, (
                "The base iterator does not support multi-indexing. "
                "Please use __getitem__ for single index access."
            )
            data_instances = [self._base_iterator[index] for index in indices]  # type: ignore
        return [
            self._tf(index, data_instance)
            for index, data_instance in zip(indices, data_instances, strict=True)
        ]

    def __len__(self) -> int:
        if hasattr(self._base_iterator, "__len__"):
            iterator = (
                self._subset_indices
                if self._subset_indices is not None
                else self._base_iterator
            )
            if self._max_len is not None:
                return min(self._max_len, len(iterator))  # type: ignore
            return len(iterator)  # type: ignore
        raise RuntimeError(
            "The dataset does not support length calculation. "
            "Please implement the `__len__` method in your dataset class."
        )

    def __rich_repr__(self) -> rich.pretty.RichReprResult:
        """
        Generates a rich representation of the object.

        Yields:
            RichReprResult: A generator of key-value pairs or values for the object's attributes.
        """
        ignored_fields = ["_split", "_max_len", "_apply_output_transform"]

        for name, field_repr in self.__dict__.items():
            if name not in ignored_fields:
                if name is None:
                    yield field_repr
                else:
                    yield name, field_repr
        try:
            yield "num_rows", len(self)
        except Exception:
            yield "num_rows", "unknown"
