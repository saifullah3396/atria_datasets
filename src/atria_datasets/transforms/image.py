"""
Image Transforms Module

This module defines various transformations for image data. These transformations include
converting grayscale tensors to RGB, applying general image transformations, CIFAR-10-specific
image preprocessing, and resizing images while maintaining their aspect ratio.

Classes:
    - TensorGrayToRgb: Converts grayscale tensors to RGB format.
    - ImageTransform: Applies a series of transformations to an `Image` object.
    - Cifar10ImageTransform: Preprocesses CIFAR-10 images with normalization, padding, and cropping.
    - FixedAspectRatioResize: Resizes images while maintaining their aspect ratio.

Dependencies:
    - typing: For type annotations.
    - numpy: For numerical operations.
    - torch: For tensor operations.
    - PIL.Image: For handling image files.
    - torchvision.transforms: For image transformation utilities.
    - atria_core.logger: For logging utilities.
    - atria_registry: For registering transformations.
    - atria_core.types: For the `Image` class.
    - atria_datasets.core.transforms.base: For the `DataTransform` base class.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from atria_core.logger import get_logger
from atria_registry import DATA_TRANSFORM

from atria_datasets.core.transforms.base import Compose, DataTransform

if TYPE_CHECKING:
    import torch
    from atria_core.types.tensor_types import TensorImage


logger = get_logger(__name__)


@DATA_TRANSFORM.register("tensor_gray_to_rgb")
class TensorGrayToRgb(DataTransform):
    """
    A transformation that converts grayscale tensors to RGB format.
    """

    def _apply_transforms(self, image: "torch.Tensor") -> "torch.Tensor":
        """
        Converts a grayscale tensor to RGB format by repeating the single channel.

        Args:
            image (torch.Tensor): The input tensor, either 2D or with a single channel.

        Returns:
            torch.Tensor: The RGB tensor with three channels.
        """
        if len(image.shape) == 2 or image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        else:
            return image


@DATA_TRANSFORM.register("image_transform")
class ImageTransform(DataTransform):
    """
    A transformation that applies a series of transformations to an `Image` object.

    Attributes:
        transforms (Callable): The transformation pipeline to apply.
    """

    _REGISTRY_CONFIGS = {
        "default": {
            "hydra_defaults": [
                "_self_",
                {"/data_transform@transforms.resize": "resize"},
                {"/data_transform@transforms.to_tensor": "to_tensor"},
                {"/data_transform@transforms.tensor_gray_to_rgb": "tensor_gray_to_rgb"},
                {"/data_transform@transforms.normalize": "normalize"},
            ]
        }
    }

    def __init__(
        self,
        input_path: str | None = "image",
        transforms: dict[str, Callable] | Callable | None = None,
    ):
        """
        Initializes the `ImageTransform` class.

        Args:
            input_path (Optional[str]): The path to the input attribute in the data model. Defaults to None.
            transforms (Optional[Union[Dict[str, Callable], Callable]]): The transformation pipeline to apply.
        """
        super().__init__(input_path=input_path)
        if isinstance(transforms, dict):
            self.transforms = Compose(list(transforms.values()))
        elif isinstance(transforms, partial):
            transforms = transforms()
            self.transforms = Compose([transforms])

    def _apply_transforms(self, input: TensorImage) -> TensorImage:
        """
        Applies the transformation pipeline to the `Image` object.

        Args:
            input (Image): The input `Image` object.

        Returns:
            Image: The transformed `Image` object.
        """
        input.content = self.transforms(input.content)
        return input


@DATA_TRANSFORM.register("cifar10_image_transform")
class Cifar10ImageTransform(DataTransform):
    """
    A transformation for preprocessing CIFAR-10 images.

    Attributes:
        mean (Union[float, List[float]]): The mean values for normalization.
        std (Union[float, List[float]]): The standard deviation values for normalization.
        pad_size (int): The padding size for the image.
        crop_size (int): The crop size for the image.
        train (bool): Whether the transformation is for training or evaluation.
        transforms (Callable): The transformation pipeline to apply.
    """

    def __init__(
        self,
        input_path: str | None = None,
        mean: float | list[float] | None = None,
        std: float | list[float] | None = None,
        pad_size: int = 4,
        crop_size: int = 32,
        train: bool = False,
    ):
        """
        Initializes the `Cifar10ImageTransform` class.

        Args:
            input_path (Optional[str]): The path to the input attribute in the data model.
            mean (Union[float, List[float]]): The mean values for normalization. Defaults to CIFAR-10 values.
            std (Union[float, List[float]]): The standard deviation values for normalization. Defaults to CIFAR-10 values.
            pad_size (int): The padding size for the image. Defaults to 4.
            crop_size (int): The crop size for the image. Defaults to 32.
            train (bool): Whether the transformation is for training or evaluation. Defaults to False.
        """
        from torchvision import transforms as T

        super().__init__(input_path=input_path)
        self.mean = [0.4914, 0.4822, 0.4465] if mean is None else mean
        self.std = [0.247, 0.243, 0.261] if std is None else std
        self.pad_size = pad_size
        self.crop_size = crop_size
        self.train = train

        if self.train:
            self.transforms = T.Compose(
                [
                    T.Pad(self.pad_size, padding_mode="reflect"),
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(self.crop_size),
                    T.Normalize(np.array(self.mean), np.array(self.std)),
                ]
            )
        else:
            self.transforms = T.Compose(
                [T.Normalize(np.array(self.mean), np.array(self.std))]
            )

    def _apply_transforms(self, image: "torch.Tensor") -> "torch.Tensor":
        """
        Applies the CIFAR-10 preprocessing pipeline to the input tensor.

        Args:
            image (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The preprocessed tensor.
        """
        return self.transforms(image)


@DATA_TRANSFORM.register("fixed_aspect_ratio_resize")
class FixedAspectRatioResize(DataTransform):
    """
    A transformation for resizing images while maintaining their aspect ratio.

    Attributes:
        max_size (int): The maximum size for the longer dimension of the image.
        interpolation (str): The interpolation method to use for resizing.
        antialias (bool): Whether to apply antialiasing during resizing.
        pad_value (int): The value to use for padding.
        transforms (Optional[Callable]): Additional transformations to apply after resizing.
    """

    def __init__(
        self,
        input_path: str | None = "image",
        max_size: int = 512,
        transforms: Callable | None = None,
    ):
        """
        Initializes the `FixedAspectRatioResize` class.

        Args:
            input_path (Optional[str]): The path to the input attribute in the data model.
            max_size (int): The maximum size for the longer dimension of the image.
            transforms (Optional[Callable]): Additional transformations to apply after resizing. Defaults to None.
        """
        super().__init__(input_path=input_path)
        self.max_size = max_size
        self.transforms = transforms

    def _apply_transforms(self, image: "TensorImage") -> "TensorImage":
        """
        Resizes the image while maintaining its aspect ratio.

        Args:
            image (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The resized tensor.
        """

        _, h, w = image.shape
        if h > w:
            updated_h = self.max_size
            updated_w = int(self.max_size / h * w)
        else:
            updated_h = int(self.max_size / w * h)
            updated_w = self.max_size
        image.resize(width=updated_w, height=updated_h)
        return image
