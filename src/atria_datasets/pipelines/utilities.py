"""
Utilities for Data Pipelines

This module provides utility functions for managing data pipelines, including creating
data loaders with distributed configurations and a default collate function for batching
data instances.

Functions:
    - auto_dataloader: Automatically configures a DataLoader for distributed training.
    - default_collate: Default collate function for batching data instances.

Dependencies:
    - warnings: For issuing warnings about configuration issues.
    - typing: For type annotations.
    - torch.utils.data: For DataLoader and Dataset utilities.
    - ignite.distributed: For distributed training utilities.
    - atria_core.logger: For logging utilities.
    - atria_core.types: For the `BaseDataInstance` class.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import TYPE_CHECKING, Any

from atria_core.logger import get_logger
from atria_core.types import BaseDataInstance

if TYPE_CHECKING:
    from atria_datasets.transforms.mmdet import MMDetInput

logger = get_logger(__name__)


def auto_dataloader(dataset: Any, **kwargs: Any) -> Any:
    """
    Automatically configures a DataLoader for distributed training.

    This function adjusts DataLoader settings based on the distributed training configuration,
    including rank, world size, and device type. It supports XLA devices and provides warnings
    for incompatible configurations.

    Args:
        iterator (Iterator): The dataset split iterator to load data from.
        **kwargs (Any): Additional arguments for configuring the DataLoader.

    Returns:
        DataLoader: A configured DataLoader instance.

    Raises:
        ValueError: If incompatible configurations are detected.
    """
    from ignite.distributed import DistributedProxySampler
    from ignite.distributed import utils as idist
    from ignite.distributed.comp_models import xla as idist_xla
    from torch.utils.data import DataLoader, IterableDataset
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data.sampler import Sampler
    from wids import ChunkedSampler, DistributedChunkedSampler

    rank = idist.get_rank()
    world_size = idist.get_world_size()

    if world_size > 1:
        if "batch_size" in kwargs and kwargs["batch_size"] >= world_size:
            kwargs["batch_size"] //= world_size

        nproc = idist.get_nproc_per_node()
        if "num_workers" in kwargs and kwargs["num_workers"] >= nproc:
            kwargs["num_workers"] = (kwargs["num_workers"] + nproc - 1) // nproc

        if "batch_sampler" not in kwargs:
            if isinstance(dataset, IterableDataset):
                logger.info(
                    "Found iterable dataset, dataloader will be created without any distributed sampling. "
                    "Please, make sure that the dataset itself produces different data on different ranks."
                )
            else:
                sampler: DistributedProxySampler | DistributedSampler | Sampler | None
                sampler = kwargs.get("sampler", None)
                if isinstance(sampler, DistributedSampler):
                    if sampler.rank != rank:
                        logger.warning(
                            f"Found distributed sampler with rank={sampler.rank}, but process rank is {rank}"
                        )
                    if sampler.num_replicas != world_size:
                        logger.warning(
                            f"Found distributed sampler with num_replicas={sampler.num_replicas}, "
                            f"but world size is {world_size}"
                        )
                elif isinstance(sampler, ChunkedSampler):
                    sampler = DistributedChunkedSampler(
                        dataset,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=sampler.shuffle,
                        shufflefirst=sampler.shufflefirst,
                        seed=sampler.seed,
                        drop_last=sampler.drop_last,
                        chunk_size=sampler.chunk_size,
                    )
                elif sampler is None:
                    shuffle = kwargs.pop("shuffle", True)
                    sampler = DistributedSampler(
                        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
                    )
                else:
                    sampler = DistributedProxySampler(
                        sampler, num_replicas=world_size, rank=rank
                    )
                kwargs["sampler"] = sampler
        else:
            logger.warning(
                "Found batch_sampler in provided kwargs. Please, make sure that it is compatible "
                "with distributed configuration"
            )

    if (
        idist.has_xla_support
        and idist.backend() == idist_xla.XLA_TPU
        and kwargs.get("pin_memory", False)
    ):
        logger.warning(
            "Found incompatible options: xla support and pin_memory args equal True. "
            "Argument `pin_memory=False` will be used to construct data loader."
        )
        kwargs["pin_memory"] = False
    else:
        kwargs["pin_memory"] = kwargs.get("pin_memory", "cuda" in idist.device().type)

    dataloader = DataLoader(dataset, **kwargs)
    if (
        idist.has_xla_support
        and idist.backend() == idist_xla.XLA_TPU
        and world_size > 1
    ):
        logger.info("DataLoader is wrapped by `MpDeviceLoader` on XLA")

        from torch_xla.distributed.parallel_loader import MpDeviceLoader  # type: ignore

        mp_device_loader_cls = MpDeviceLoader
        mp_dataloader = mp_device_loader_cls(dataloader, idist.device())
        mp_dataloader.sampler = dataloader.sampler  # type: ignore[attr-defined]
        return mp_dataloader

    return dataloader


def default_collate(batch: BaseDataInstance):
    """
    Default collate function for DataLoader.

    This function collates a batch of data instances into a single batch. It is used when
    the `collate_fn` argument is not provided to the DataLoader.

    Args:
        batch (BaseDataInstance): A batch of data instances.

    Returns:
        Any: The collated batch.

    Raises:
        ValueError: If the batch is empty or not a list.
    """
    if isinstance(batch, list) and len(batch) > 0:
        return batch[0].batched([sample.to_tensor() for sample in batch])
    else:
        raise ValueError("Batch is empty or not a list.")


def mmdet_pseudo_collate(batch: list["MMDetInput"]):
    """
    Default collate function for MMDetInput inputs.

    This function collates a batch of data instances into a single batch. It is used when
    the `collate_fn` argument is not provided to the DataLoader.

    Args:
        batch (List[MMDetInput]): A batch of data instances.

    Returns:
        Any: The collated batch.

    Raises:
        ValueError: If the batch is empty or not a list.
    """
    from atria_datasets.core.transforms.mmdet import MMDetInput
    from mmengine.dataset.utils import pseudo_collate

    return MMDetInput.model_construct(
        **pseudo_collate([sample.model_dump() for sample in batch])
    )
