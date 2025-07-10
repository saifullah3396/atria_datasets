"""
Torchvision Transforms Module

This module provides indirect registration of common torchvision transformations into the
`DATA_TRANSFORM` registry. These transformations include resizing, cropping, normalization,
and random horizontal flipping, which are commonly used in image preprocessing pipelines.

Transformations:
    - Resize: Resizes the input tensor to the specified size.
    - CenterCrop: Crops the input tensor at the center to the specified size.
    - Normalize: Normalizes the input tensor using the specified mean and standard deviation.
    - RandomHorizontalFlip: Randomly flips the input tensor horizontally with a default probability of 0.5.

Dependencies:
    - torch: For tensor operations.
    - torchvision.transforms: For image transformation utilities.
    - atria_registry: For registering transformations into the `DATA_TRANSFORM` registry.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import TYPE_CHECKING

from atria_core.utilities.imports import _resolve_module_from_path
from atria_registry import DATA_TRANSFORM

from atria_datasets.core.transforms.base import DataTransform

if TYPE_CHECKING:
    import torch


class TorchvisionTransform(DataTransform):
    """
    A wrapper class for applying a specified transformation to a PyTorch tensor.

    This class extends the `DataTransform` base class and is designed to apply a
    user-defined transformation to a PyTorch tensor. The transformation logic is
    encapsulated in the `transform` parameter passed during initialization.

    Attributes:
        transform (Callable): The transformation function to be applied to the input tensor.

    Args:
        transform (Callable): A callable object (e.g., function or class) that defines
            the transformation to be applied to the input tensor.

    Methods:
        _apply_transform(image: torch.Tensor) -> torch.Tensor:
            Applies the specified transformation to the input tensor.

    Example:
        >>> import torch
        >>> from some_module import WrappedTransform
        >>> def example_transform(tensor):
        ...     return tensor * 2
        >>> wrapped_transform = WrappedTransform(transform=example_transform)
        >>> input_tensor = torch.tensor([1, 2, 3])
        >>> output_tensor = wrapped_transform._apply_transform(input_tensor)
        >>> print(output_tensor)
        tensor([2, 4, 6])
    """

    def __init__(self, transform: str, **kwargs):
        super().__init__()
        self.transform = _resolve_module_from_path(
            f"torchvision.transforms.{transform}"
        )(**kwargs)

    def _apply_transforms(self, image: "torch.Tensor") -> "torch.Tensor":
        return self.transform(image)


# Registering torchvision transformations into the DATA_TRANSFORM registry
DATA_TRANSFORM.register_torchvision_transform(
    "Resize", interpolation=2, size=(224, 224)
)
DATA_TRANSFORM.register_torchvision_transform("ToTensor")
DATA_TRANSFORM.register_torchvision_transform("CenterCrop")
DATA_TRANSFORM.register_torchvision_transform(
    "Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
DATA_TRANSFORM.register_torchvision_transform("RandomHorizontalFlip")
