"""
Data Transforms Module

This module defines the `DataTransform` and `DataTransformsDict` classes, which provide
a framework for applying transformations to data. The `DataTransform` class allows for
customizable transformations on data models, while the `DataTransformsDict` class organizes
transformations for training and evaluation workflows.

Classes:
    - DataTransform: A class for applying transformations to data models.
    - DataTransformsDict: A class for managing transformations for training and evaluation.

Dependencies:
    - typing: For type annotations.
    - rich.pretty: For pretty-printing representations.
    - atria_registry: For loading transformations from a registry.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

from atria_core.utilities.repr import RepresentationMixin
from pydantic import BaseModel, ConfigDict


class DataTransform(RepresentationMixin):
    """
    A class for applying transformations to data models.

    This class provides functionality for applying transformations to data models,
    validating input data, and managing transformation pipelines.

    Attributes:
        input_path (Optional[str]): The path to the input attribute in the data model. Defaults to None.
    """

    def __init__(self, input_path: str | None = None):
        """
        Initializes the `DataTransform` class.

        Args:
            input_path (Optional[str]): The path to the input attribute in the data model. Defaults to None.
        """
        self.input_path = input_path

    @staticmethod
    def load_from_registry(
        name: str,
        provider: str | None = "atria",
        config: Mapping[str, Any] | None = None,
    ):
        """
        Loads a transformation from the registry.

        Args:
            name (str): The name of the transformation to load.
            provider (Optional[str]): The package name for the registry. Defaults to "atria".
            config (Optional[Mapping[str, Any]]): Configuration for the transformation. Defaults to None.

        Returns:
            Any: The loaded transformation.
        """
        from atria_registry import DATA_TRANSFORM

        return DATA_TRANSFORM.load_from_registry(
            name=name, provider=provider, config=config
        )

    @abstractmethod
    def _apply_transforms(self, input: Any) -> Any:
        """
        Apply the necessary transformations to the input data.

        This method must be implemented by subclasses to define the specific
        transformations to be applied to the input.

        Args:
            input (T): The input data to be transformed.

        Returns:
            Any: The transformed data.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the _apply_transforms method."
        )

    def _validate_and_apply_transforms(
        self, input: Any | Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """
        Validates the input data and applies the transformation.

        Args:
            input (Union[Any, Mapping[str, Any]]): The input data to validate and transform.

        Returns:
            Mapping[str, Any]: The transformed data.

        Raises:
            AssertionError: If the input data does not meet the validation criteria.
        """
        if self.input_path is not None:
            attrs = self.input_path.split(".")
            obj = input
            for attr in attrs[:-1]:
                obj = getattr(obj, attr)
            current_attr = getattr(obj, attrs[-1])
            assert current_attr is not None, (
                f"{self.__class__.__name__} transform requires {self.input_path} to be present in the sample."
            )
            setattr(obj, attrs[-1], self._apply_transforms(current_attr))
            return input
        else:
            return self._apply_transforms(input)

    def __call__(
        self, input: Any | Mapping[str, Any] | list[Mapping[str, Any]]
    ) -> Any | Mapping[str, Any] | list[Mapping[str, Any]]:
        """
        Applies the transformation to the input data.

        Args:
            input (Union[Any, Mapping[str, Any], List[Mapping[str, Any]]]): The input data to transform.

        Returns:
            Union[Any, Mapping[str, Any], List[Mapping[str, Any]]]: The transformed data.
        """
        if isinstance(input, list):
            return [self(s) for s in input]
        return self._validate_and_apply_transforms(input)


class DataTransformsDict(BaseModel):
    """
    A class for managing transformations for training and evaluation.

    This class organizes transformations into separate pipelines for training and evaluation workflows.

    Attributes:
        train (partial[DataTransform] | OrderedDict[str, partial[DataTransform]] | None):
            The transformation(s) to apply during training. Defaults to None.
        evaluation (partial[DataTransform] | OrderedDict[str, partial[DataTransform]] | None):
            The transformation(s) to apply during evaluation. Defaults to None.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=False, extra="forbid"
    )

    train: DataTransform | OrderedDict[str, DataTransform] | None = None
    evaluation: DataTransform | OrderedDict[str, DataTransform] | None = None

    def compose(self, type: str) -> "DataTransform" | Callable:
        from torchvision.transforms import Compose  # type: ignore

        tf = getattr(self, type, None)
        if tf is None:
            raise ValueError(
                f"Transformations for type '{type}' are not defined in {self.__class__.__name__}."
            )
        if isinstance(tf, dict):
            return Compose(list(tf.values()))
        return tf


class Compose(RepresentationMixin):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t(input)
        return input
