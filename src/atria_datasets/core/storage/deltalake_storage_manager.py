# """
# File Storage Manager Module

# This module defines the `FileStorageManager` class, which provides functionality for managing
# the storage of datasets in various formats, such as Msgpack and WebDataset. It includes methods
# for writing datasets to storage, reading datasets from storage, shuffling datasets, and managing
# dataset shards.

# Classes:
#     - FileStorageManager: A class for managing dataset storage in file-based formats.

# Dependencies:
#     - glob: For file pattern matching.
#     - itertools: For iterator utilities.
#     - os: For file system operations.
#     - shutil: For file and directory operations.
#     - pathlib.Path: For handling file paths.
#     - typing: For type annotations.
#     - ray: For distributed processing.
#     - tqdm: For progress tracking.
#     - webdataset: For handling WebDataset format.
#     - atria_core.logger: For logging utilities.
#     - atria_registry: For registering storage managers.
#     - atria_datasets.core.datasets: For dataset-related classes and metadata.
#     - atria_datasets.core.storage.dataset_storage_manager: For the base storage manager class.
#     - atria_datasets.core.storage.shard_writer_actor: For managing shard writers.
#     - atria_datasets.core.storage.utilities: For utility functions and constants.

# Author: Your Name (your.email@example.com)
# Date: 2025-04-07
# Version: 1.0.0
# License: MIT
# """

# from collections.abc import Callable
# from pathlib import Path
# from typing import TYPE_CHECKING

# import tqdm
# from atria_core.logger.logger import get_logger

# if TYPE_CHECKING:
#     from atria_core.types import BaseDataInstance, DatasetSplitType

#     from atria_datasets.core.dataset.atria_dataset import SplitIterator


# tqdm.tqdm.pandas()

# logger = get_logger(__name__)


# class DeltalakeStorageManager:
#     """
#     A class for managing dataset storage in deltalake format

#     Attributes:
#         _streaming_mode (bool): Whether to enable streaming mode for WebDataset. Defaults to False.
#     """

#     def __init__(
#         self, storage_dir: str, num_processes: int = 8, write_batch_size: int = 100000
#     ):
#         self._storage_dir = Path(storage_dir)
#         self._num_processes = num_processes
#         self._write_batch_size = write_batch_size

#         if not self._storage_dir.exists():
#             self._storage_dir.mkdir(parents=True, exist_ok=True)
#         assert self._storage_dir.is_dir(), (
#             f"Storage directory {self._storage_dir} must be a directory."
#         )

#     def split_dir(self, split: "DatasetSplitType") -> Path:
#         return Path(self._storage_dir) / f"delta/{split.value}"

#     def dataset_exists(self) -> bool:
#         return (Path(self._storage_dir) / "delta/").exists()

#     def split_exists(self, split: "DatasetSplitType") -> bool:
#         return self.split_dir(split).exists()

#     def purge_split(self, split: "DatasetSplitType") -> None:
#         import shutil

#         split_dir = self.split_dir(split)
#         if split_dir.exists():
#             shutil.rmtree(split_dir)

#     def write_split(self, split_iterator: "SplitIterator") -> None:
#         from atria_datasets.core.dataset.exceptions import SplitNotFoundError

#         try:
#             split_iterator.disable_output_transform()
#             self._write(split_iterator=split_iterator)
#             split_iterator.enable_output_transform()
#         except SplitNotFoundError as e:
#             raise e
#         except Exception as e:
#             self.purge_split(split_iterator.split)
#             raise RuntimeError(
#                 f"Error while writing dataset split {split_iterator.split.value} to storage"
#             ) from e
#         except KeyboardInterrupt as e:
#             self.purge_split(split_iterator.split)
#             raise KeyboardInterrupt(
#                 "KeyboardInterrupt detected. Stopping dataset preparation..."
#             ) from e

#     def map_to_nullable_dtypes(
#         self, row_serialization_types: dict[str, type]
#     ) -> dict[str, str]:
#         type_map = {str: "string", int: "Int64", float: "Float64", bool: "boolean"}
#         return {
#             col: type_map.get(tp, "object")
#             for col, tp in row_serialization_types.items()
#         }

#     def read_split(
#         self,
#         split: "DatasetSplitType",
#         data_model: "BaseDataInstance",
#         output_transform: Callable | None = None,
#         allowed_keys: set[str] | None = None,
#         streaming_mode: bool = False,
#     ) -> "SplitIterator":
#         from atria_datasets.core.dataset.atria_dataset import SplitIterator
#         from atria_datasets.core.storage.deltalake_reader import DeltalakeReader
#         from atria_datasets.core.storage.deltalake_streamer import DeltalakeStreamer

#         if not self.split_exists(split):
#             raise RuntimeError(
#                 f"Dataset split {split.value} not prepared. Please call `write_split()` first."
#             )

#         if allowed_keys is not None:
#             allowed_keys.add("index")
#             allowed_keys.add("sample_id")

#         if streaming_mode:
#             from atria_hub.hub import AtriaHub

#             storage_options = AtriaHub().get_storage_options()
#             base_iterator = DeltalakeStreamer(
#                 path=str(self.split_dir(split=split)),
#                 data_model=data_model,
#                 allowed_keys=allowed_keys,
#                 storage_options=storage_options,
#             )
#         else:
#             base_iterator = DeltalakeReader(  # type: ignore
#                 path=str(self.split_dir(split=split)),
#                 data_model=data_model,
#                 allowed_keys=allowed_keys,
#             )
#         return SplitIterator(
#             split=split,
#             base_iterator=base_iterator,
#             output_transform=output_transform,
#             data_model=data_model,
#         )

#     def get_split_files(self) -> list[tuple[str, str]]:
#         import deltalake
#         from atria_core.types import DatasetSplitType

#         # Get all delta files in the storage directory
#         delta_files = list((self._storage_dir / "delta").glob("**/*.*"))

#         # Create a list of tuples (source, target) for delta files
#         files_src_tgt = []
#         files_src_tgt = [
#             (str(f.resolve()), str(f.relative_to(self._storage_dir)))
#             for f in delta_files
#             if f.is_file()
#         ]

#         # Iterate over each split and check for file paths
#         for split in DatasetSplitType:
#             if not self.split_exists(split):
#                 continue

#             dataframe = deltalake.DeltaTable(self.split_dir(split=split)).to_pandas()
#             for column in dataframe.columns:
#                 if "file_path" in column.lower():
#                     file_path_col = dataframe[column].dropna()
#                     if file_path_col.empty:
#                         continue
#                     for raw_path in file_path_col.tolist():
#                         src = str(Path(raw_path).resolve())
#                         tgt = str("raw" / Path(raw_path).relative_to(self._storage_dir))
#                         files_src_tgt.append((src, tgt))

#         return files_src_tgt

#     def _write(self, split_iterator: "SplitIterator") -> Path:
#         import itertools

#         import deltalake
#         import pandas as pd
#         import pyarrow as pa
#         from pandarallel import pandarallel

#         pandarallel.initialize(
#             nb_workers=self._num_processes, progress_bar=True, verbose=0
#         )

#         split_dir = self.split_dir(split=split_iterator.split)
#         logger.info(
#             f"Preprocessing dataset split {split_iterator.split.value} to cached deltalake storage {split_dir}."
#         )

#         # Choose the iterator, applying islice if _max_len is set
#         if split_iterator._max_len is None:
#             base_iter = split_iterator._base_iterator
#         else:
#             base_iter = itertools.islice(
#                 split_iterator._base_iterator,
#                 split_iterator._max_len,  # type: ignore
#             )

#         # Enumerate over the chosen iterator and convert to list
#         all_samples = list(enumerate(base_iter))

#         # Create a DataFrame from the enumerated samples
#         df_all = pd.DataFrame(all_samples, columns=["index", "sample"])

#         # Apply your transform in parallel using pandarallel
#         transform = split_iterator._tf
#         df_processed = df_all.parallel_apply(
#             lambda row: transform(row["index"], row["sample"]).to_row(),  # type: ignore
#             axis=1,
#             result_type="expand",
#         )

#         table = pa.Table.from_pandas(
#             df_processed, schema=split_iterator.data_model.pa_schema()
#         )
#         deltalake.write_deltalake(split_dir, table, mode="overwrite")

#         return split_dir

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

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tqdm
from atria_core.logger.logger import get_logger

if TYPE_CHECKING:
    from atria_core.types import BaseDataInstance, DatasetSplitType

    from atria_datasets.core.dataset.atria_dataset import SplitIterator


tqdm.tqdm.pandas()

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
        self, storage_dir: str, num_processes: int = 8, write_batch_size: int = 10000
    ):
        self._storage_dir = Path(storage_dir)
        self._num_processes = num_processes
        self._write_batch_size = write_batch_size

        if not self._storage_dir.exists():
            self._storage_dir.mkdir(parents=True, exist_ok=True)
        assert self._storage_dir.is_dir(), (
            f"Storage directory {self._storage_dir} must be a directory."
        )

    def split_dir(self, split: "DatasetSplitType") -> Path:
        return Path(self._storage_dir) / f"delta/{split.value}"

    def dataset_exists(self) -> bool:
        return (Path(self._storage_dir) / "delta/").exists()

    def split_exists(self, split: "DatasetSplitType") -> bool:
        return self.split_dir(split).exists()

    def purge_split(self, split: "DatasetSplitType") -> None:
        import shutil

        split_dir = self.split_dir(split)
        if split_dir.exists():
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
        from atria_datasets.core.storage.deltalake_reader import DeltalakeReader
        from atria_datasets.core.storage.deltalake_streamer import DeltalakeStreamer

        if not self.split_exists(split):
            raise RuntimeError(
                f"Dataset split {split.value} not prepared. Please call `write_split()` first."
            )

        if allowed_keys is not None:
            allowed_keys.add("index")
            allowed_keys.add("sample_id")

        if streaming_mode:
            from atria_hub.hub import AtriaHub

            storage_options = AtriaHub().get_storage_options()
            base_iterator = DeltalakeStreamer(
                path=str(self.split_dir(split=split)),
                data_model=data_model,
                allowed_keys=allowed_keys,
                storage_options=storage_options,
            )
        else:
            base_iterator = DeltalakeReader(  # type: ignore
                path=str(self.split_dir(split=split)),
                data_model=data_model,
                allowed_keys=allowed_keys,
            )
        return SplitIterator(
            split=split,
            base_iterator=base_iterator,
            output_transform=output_transform,
            data_model=data_model,
        )

    def get_split_files(self, data_dir: str) -> list[tuple[str, str]]:
        import deltalake
        from atria_core.types import DatasetSplitType

        # Get all delta files in the storage directory
        assert self._storage_dir.parent.name == "storage", (
            f"Expected storage directory to be in 'storage' parent directory, but got {self._storage_dir.parent}."
        )
        storage_base_dir = self._storage_dir.parent
        delta_files = list((self._storage_dir / "delta").glob("**/*.*"))

        # Create a list of tuples (source, target) for delta files
        files_src_tgt = []
        files_src_tgt = [
            (str(f.resolve()), str(f.relative_to(storage_base_dir)))
            for f in delta_files
            if f.is_file()
        ]

        # Iterate over each split and check for file paths
        for split in DatasetSplitType:
            if not self.split_exists(split):
                continue

            dataframe = deltalake.DeltaTable(self.split_dir(split=split)).to_pandas()
            for column in dataframe.columns:
                if "file_path" in column.lower():
                    assert (
                        column.lower().replace("file_path", "content")
                        in dataframe.columns
                    ), (
                        f"Column '{column}' not found in DataFrame. Expected 'content' column."
                    )
                    content_col = dataframe[column].dropna()
                    if not content_col.empty:
                        continue
                    file_path_col = dataframe[column].dropna()
                    if file_path_col.empty:
                        continue
                    for raw_path in file_path_col.tolist():
                        src = str(Path(raw_path).resolve())
                        tgt = str("raw" / Path(raw_path).relative_to(data_dir))
                        files_src_tgt.append((src, tgt))

        return files_src_tgt

    def _write(self, split_iterator: "SplitIterator") -> Path:
        import multiprocessing as mp

        import deltalake
        import pyarrow as pa

        split_dir = self.split_dir(split=split_iterator.split)
        logger.info(
            f"Preprocessing dataset split {split_iterator.split.value} to cached deltalake storage {split_dir}."
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

        with mp.Pool(self._num_processes) as pool:
            batch = []
            for result in tqdm.tqdm(
                pool.imap_unordered(
                    SerializerWorker(split_iterator._tf), split_iterator
                ),
                desc="Writing to Deltalake",
                total=len(split_iterator),
                unit="rows",
            ):
                batch.append(result)
                if len(batch) >= 10000:
                    write_batch(batch, is_first_batch=first_batch)
                    first_batch = False
                    batch = []  # Reset batch

            if batch:
                write_batch(batch, is_first_batch=first_batch)

        split_iterator.enable_tf()

        return split_dir
