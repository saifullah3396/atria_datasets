"""
Data Module

This module provides data-related utilities and components for the Atria framework. It includes dataset management, data transformations, and storage utilities.

Submodules:
    - batch_samplers: Batch sampling strategies for datasets.
    - dataset_splitters: Utilities for splitting datasets into training, validation, and test sets.
    - datasets: Dataset definitions and metadata management.
    - pipelines: Data pipelines for preprocessing and augmentation.
    - storage: File storage management for datasets.
    - structures: Data structures for representing various data types.
    - transforms: Data transformation utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""
from atria_datasets.core.dataset.atria_dataset import AtriaDataset
from atria_datasets.core.dataset.atria_huggingface_dataset import (
    AtriaHuggingfaceDataset,
)
from atria_datasets.core.dataset.split_iterator import SplitIterator
from atria_datasets.core.dataset_splitters.standard_splitter import StandardSplitter
from atria_datasets.core.download_manager.download_file_info import DownloadFileInfo
from atria_datasets.core.download_manager.download_manager import DownloadManager
from atria_datasets.core.download_manager.file_downloader import (
    FileDownloader,
    FTPFileDownloader,
    GoogleDriveDownloader,
    HTTPDownloader,
)
from atria_datasets.core.storage.deltalake_reader import DeltalakeReader
from atria_datasets.core.storage.deltalake_storage_manager import (
    DeltalakeStorageManager,
)
from atria_datasets.core.storage.msgpack_shard_writer import (
    MsgpackFileWriter,
    MsgpackShardWriter,
)
from atria_datasets.core.storage.sharded_dataset_storage_manager import (
    ShardedDatasetStorageManager,
)

__all__ = [
    "AtriaDataset",
    "AtriaHuggingfaceDataset",
    "SplitIterator",
    "StandardSplitter",
    "DownloadManager",
    "DownloadFileInfo",
    "FileDownloader",
    "HTTPDownloader",
    "GoogleDriveDownloader",
    "FTPFileDownloader",
    "DeltalakeStorageManager",
    "DeltalakeReader",
    "MsgpackFileWriter",
    "MsgpackShardWriter",
    "ShardedDatasetStorageManager",
]
