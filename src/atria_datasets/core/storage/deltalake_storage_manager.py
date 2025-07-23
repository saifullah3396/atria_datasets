"""
File Storage Manager Module

This module defines the `FileStorageManager` class, which provides functionality for managing
the storage of datasets in various formats, such as Msgpack and WebDataset. It includes methods
for writing datasets to storage, reading datasets from storage, shuffling datasets, and managing
dataset shards.

Classes:
    - FileStorageManager: A class for managing dataset storage in file-based formats.

Dependencies:
    - glob: For file pattern matching.
    - itertools: For iterator utilities.
    - os: For file system operations.
    - shutil: For file and directory operations.
    - pathlib.Path: For handling file paths.
    - typing: For type annotations.
    - ray: For distributed processing.
    - tqdm: For progress tracking.
    - webdataset: For handling WebDataset format.
    - atria_core.logger: For logging utilities.
    - atria_registry: For registering storage managers.
    - atria_datasets.core.datasets: For dataset-related classes and metadata.
    - atria_datasets.core.storage.dataset_storage_manager: For the base storage manager class.
    - atria_datasets.core.storage.shard_writer_actor: For managing shard writers.
    - atria_datasets.core.storage.utilities: For utility functions and constants.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import multiprocessing as mp
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import tqdm
from atria_core.logger.logger import get_logger

from atria_datasets.core.dataset.atria_dataset import DatasetLoadingMode
from atria_datasets.core.storage.deltalake_reader import DeltalakeReader

if TYPE_CHECKING:
    from atria_core.types import BaseDataInstance, DatasetSplitType

    from atria_datasets.core.dataset.atria_dataset import SplitIterator


logger = get_logger(__name__)


class SerializerWorker:
    def __init__(self, tf: Callable):
        """
        Initialize the SerializerWorker with a transformation function.
        """
        self._tf = tf

    def __call__(self, inputs: Any) -> Any:
        """
        Call the transformation function with the provided arguments.
        """
        return self._tf(*inputs).to_row()


class DeltalakeStorageManager:
    """
    A class for managing dataset storage in deltalake format

    Attributes:
        _streaming_mode (bool): Whether to enable streaming mode for WebDataset. Defaults to False.
    """

    def __init__(
        self,
        storage_dir: str,
        config_name: str,
        num_processes: int = 8,
        max_memory: int = 1000_000_000,
    ):
        self._storage_dir = Path(storage_dir)
        self._config_name = config_name
        self._num_processes = num_processes
        self._max_memory = max_memory

        if not self._storage_dir.exists():
            self._storage_dir.mkdir(parents=True, exist_ok=True)
        assert self._storage_dir.is_dir(), (
            f"Storage directory {self._storage_dir} must be a directory."
        )
        (self._storage_dir / config_name).mkdir(parents=True, exist_ok=True)

    def split_dir(self, split: "DatasetSplitType") -> Path:
        return Path(self._storage_dir) / f"{self._config_name}/delta/{split.value}"

    def dataset_exists(self) -> bool:
        return (Path(self._storage_dir) / f"{self._config_name}/delta/").exists()

    def split_exists(self, split: "DatasetSplitType") -> bool:
        return self.split_dir(split).exists()

    def purge_split(self, split: "DatasetSplitType") -> None:
        import shutil

        split_dir = self.split_dir(split)
        if split_dir.exists():
            logger.info(
                f"Purging dataset split {split.value} from storage {split_dir}."
            )
            shutil.rmtree(split_dir)

    def write_split(self, split_iterator: "SplitIterator") -> None:
        try:
            self._write(split_iterator=split_iterator)
        except Exception as e:
            self.purge_split(split_iterator.split)
            raise RuntimeError(
                f"Error while writing dataset split {split_iterator.split.value} to storage"
            ) from e
        except KeyboardInterrupt as e:
            self.purge_split(split_iterator.split)
            raise KeyboardInterrupt(
                "KeyboardInterrupt detected. Stopping dataset preparation..."
            ) from e

    def map_to_nullable_dtypes(
        self, row_serialization_types: dict[str, type]
    ) -> dict[str, str]:
        type_map = {str: "string", int: "Int64", float: "Float64", bool: "boolean"}
        return {
            col: type_map.get(tp, "object")
            for col, tp in row_serialization_types.items()
        }

    def read_split(
        self,
        split: "DatasetSplitType",
        data_model: "BaseDataInstance",
        output_transform: Callable | None = None,
        allowed_keys: set[str] | None = None,
        streaming_mode: bool = False,
    ) -> "SplitIterator":
        from atria_datasets.core.dataset.atria_dataset import SplitIterator

        if not self.split_exists(split):
            raise RuntimeError(
                f"Dataset split {split.value} not prepared. Please call `write_split()` first."
            )

        if allowed_keys is not None:
            allowed_keys.add("index")
            allowed_keys.add("sample_id")

        return SplitIterator(
            split=split,
            base_iterator=DeltalakeReader.from_mode(
                table_path=str(self.split_dir(split=split)),
                storage_dir=self._storage_dir,
                config_name=self._config_name,
                data_model=data_model,
                mode=DatasetLoadingMode.local_streaming
                if streaming_mode
                else DatasetLoadingMode.in_memory,
                allowed_keys=allowed_keys,
            ),
            output_transform=output_transform,
            data_model=data_model,
        )

    def get_splits(self) -> list["DatasetSplitType"]:
        """
        Get a list of available dataset splits in the storage directory.

        Returns:
            list[DatasetSplitType]: A list of available dataset splits.
        """
        from atria_core.types import DatasetSplitType

        return [split for split in DatasetSplitType if self.split_exists(split)]

    def prepare_split_files(self, data_dir: str) -> list[tuple[str, str]]:
        import deltalake
        from atria_core.types import DatasetSplitType

        # Get all delta files in the storage directory
        delta_files = list(
            (self._storage_dir / self._config_name / "delta").glob("**/*.*")
        )

        # Create a list of tuples (source, target) for delta files
        files_src_tgt = {
            (str(f), str(f.relative_to(self._storage_dir)))
            for f in delta_files
            if f.is_file()
        }

        def map_file_path(file_path):
            if pd.isna(file_path):
                return None  # skip NaN
            tgt = str(Path(self._config_name) / "raw")
            files_src_tgt.add((str(Path(data_dir) / file_path), tgt))
            return tgt

        for split in list(DatasetSplitType):
            if not self.split_exists(split):
                continue

            dt = deltalake.DeltaTable(self.split_dir(split=split))
            all_columns = [f.name for f in dt.schema().fields]

            # Find all file_path columns (case-insensitive)
            file_path_cols = [col for col in all_columns if "file_path" in col.lower()]
            content_cols = [
                file_path_col.replace("file_path", "content")
                for file_path_col in file_path_cols
            ]
            dataframe = dt.to_pandas(columns=content_cols + file_path_cols)
            for file_path_column, content_column in zip(
                file_path_cols, content_cols, strict=True
            ):
                if not dataframe[content_column].dropna().empty:
                    continue
                file_path_col_data = dataframe[file_path_column].dropna()
                if file_path_col_data.empty:
                    continue
                file_path_col_data.apply(map_file_path)
        return files_src_tgt

    def _write(self, split_iterator: "SplitIterator") -> Path:
        import itertools

        import deltalake
        import pyarrow as pa

        split_dir = self.split_dir(split=split_iterator.split)
        logger.info(
            f"Preprocessing dataset split {split_iterator.split.value} to cached deltalake storage {split_dir}"
        )

        split_iterator.disable_tf()

        first_batch = True

        def write_batch(batch, is_first_batch=False):
            mode = "overwrite" if is_first_batch else "append"
            deltalake.write_deltalake(
                split_dir,
                pa.Table.from_pylist(
                    batch, schema=split_iterator.data_model.pa_schema()
                ),
                mode=mode,
            )

        try:
            total = len(split_iterator)
        except RuntimeError:
            total = None

        # Choose the iterator, applying islice if _max_len is set
        iterator = iter(split_iterator)
        if total is not None:
            iterator = itertools.islice(
                iterator,
                total,  # type: ignore
            )

        write_batch_size = None
        if self._num_processes == 0:
            # Single process mode
            batch = []
            for result in tqdm.tqdm(
                map(SerializerWorker(split_iterator._tf), iterator),
                desc="Writing to Deltalake",
                total=total,
                unit="rows",
            ):
                batch.append(result)
                if write_batch_size is None:
                    write_batch_size = self._max_memory // len(pickle.dumps(result))
                    logger.info(
                        f"Setting write batch size to {write_batch_size} based on max memory {self._max_memory // 1000_000} MB"
                    )
                if len(batch) >= write_batch_size:
                    write_batch(batch, is_first_batch=first_batch)
                    first_batch = False
                    batch = []
            if batch:
                write_batch(batch, is_first_batch=first_batch)
        else:
            with mp.Pool(self._num_processes) as pool:
                batch = []
                for result in tqdm.tqdm(
                    pool.imap_unordered(SerializerWorker(split_iterator._tf), iterator),
                    desc="Writing to Deltalake",
                    total=total,
                    unit="rows",
                ):
                    batch.append(result)
                    if write_batch_size is None:
                        write_batch_size = self._max_memory // len(pickle.dumps(result))
                        logger.info(
                            f"Setting write batch size to {write_batch_size} based on max memory {self._max_memory}"
                        )
                    if len(batch) >= write_batch_size:
                        write_batch(batch, is_first_batch=first_batch)
                        first_batch = False
                        batch = []  # Reset batch

                if batch:
                    write_batch(batch, is_first_batch=first_batch)

        split_iterator.enable_tf()

        return split_dir
