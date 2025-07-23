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
import queue
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
    def __init__(
        self,
        write_dir: str,
        split: str,
        tf: Callable,
        process_id: int = 0,
        max_shard_size: int = 100000,
        write_binary_tars: bool = True,
    ):
        """
        Initialize the SerializerWorker with a transformation function.
        """
        import webdataset as wds

        self._process_id = process_id
        self._tf = tf
        self._wds_writer: wds.ShardWriter | None = None
        self._current_shard = 0
        self._current_shard_path = None
        self._write_dir = write_dir
        self._split = split
        self._max_shard_size = max_shard_size
        self._write_binary_tars = write_binary_tars

    def __call__(self, inputs: Any) -> Any:
        """
        Call the transformation function with the provided arguments.
        """

        import webdataset as wds
        from atria_core.types import BaseDataInstance

        if self._write_binary_tars and self._wds_writer is None:
            file_path = (
                Path(self._write_dir)
                / "shards"
                / self._split
                / f"{self._process_id:06d}-%06d.tar"
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self._wds_writer = wds.ShardWriter(
                str(file_path), maxcount=self._max_shard_size
            )
            self._current_shard = self._wds_writer.shard
            self._current_shard_path = (
                Path("shards") / self._split / Path(self._wds_writer.fname).name
            )
            logger.info(
                f"Initializing WebDataset writer at {self._wds_writer.fname} with max shard size {self._max_shard_size}"
            )

        if self._write_binary_tars and self._current_shard != self._wds_writer.shard:
            self._current_shard = self._wds_writer.shard
            self._current_shard_path = (
                Path("shards") / self._split / Path(self._wds_writer.fname).name
            )

        sample: BaseDataInstance = self._tf(*inputs)
        if self._wds_writer is not None:
            sample_row = sample.to_row()
            for key in sample_row:
                if "file_path" in key:
                    file_path_key = key
                    content_key = key.replace("file_path", "content")
                    assert content_key in sample_row, (
                        f"Column '{content_key}' not found in sample row. Expected 'content' column for the corresponding "
                        f"file path '{file_path_key}'."
                    )
                    content = sample_row[content_key]
                    assert content is not None, (
                        f"Content for key '{content_key}' is None. Expected a valid binary loaded content after load()."
                    )
                    tar_content = {
                        "__key__": str(sample.key),
                        f"{content_key}": content,
                    }
                    sample_row[file_path_key] = (
                        f"tar://{self._current_shard_path}?path={sample.key}.{content_key}"
                    )
                    sample_row[content_key] = None
                    self._wds_writer.write(tar_content)
            return sample_row
        else:
            sample = sample.to_row()
            for key in sample_row:
                if "file_path" in key:
                    sample[key] = "file://" + sample[key]
            return sample.to_row()

    def close(self) -> None:
        """
        Close the WebDataset writer if it is initialized.
        """
        if self._wds_writer is not None:
            self._wds_writer.close()
            self._wds_writer = None
            logger.info("Closed WebDataset writer.")


class ProducerWorker(mp.Process):
    def __init__(self, input_queue, output_queue, serializer_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.serializer = SerializerWorker(**serializer_kwargs)

    def run(self):
        while True:
            item = self.input_queue.get()
            if item == "STOP":
                break
            try:
                result = self.serializer(item)
                self.output_queue.put(result)
            except Exception as e:
                self.output_queue.put(e)


class ConsumerWorker(mp.Process):
    def __init__(self, output_queue, write_batch_func, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.output_queue = output_queue
        self.write_batch = write_batch_func
        self.batch_size = batch_size

    def run(self):
        batch = []
        first_batch = True

        while True:
            try:
                result = self.output_queue.get(timeout=5)
            except queue.Empty:
                continue

            if result == "STOP":
                break

            if isinstance(result, Exception):
                raise result

            batch.append(result)
            if len(batch) >= self.batch_size:
                logger.info(f"Writing batch of size {len(batch)} to storage.")
                self.write_batch(batch, is_first_batch=first_batch)
                first_batch = False
                batch = []

        if batch:
            self.write_batch(batch, is_first_batch=first_batch)


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
        write_batch_size: int = 100000,
    ):
        self._storage_dir = Path(storage_dir)
        self._config_name = config_name
        self._num_processes = num_processes
        self._write_batch_size = write_batch_size

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
            if file_path.startswith("tar://"):
                file_path = file_path.replace("tar://", "").split("?")[0]
                tgt = str(Path(self._config_name) / file_path)
                files_src_tgt.add(
                    (str(Path(self._storage_dir) / self._config_name / file_path), tgt)
                )
            elif file_path.startswith("file://"):
                tgt = str(
                    Path(self._config_name) / "raw" / file_path.replace("file://", "")
                )
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
        import multiprocessing as mp

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

        if self._num_processes <= 1:
            batch = []
            worker = SerializerWorker(
                write_dir=self._storage_dir / self._config_name,
                split=split_iterator.split.value,
                tf=split_iterator._tf,
                write_binary_tars=True,
            )
            for result in tqdm.tqdm(
                map(worker, iterator),
                desc="Writing to Deltalake",
                total=total,
                unit="rows",
            ):
                batch.append(result)
                if len(batch) >= self._write_batch_size:
                    write_batch(batch, is_first_batch=first_batch)
                    first_batch = False
                    batch = []
            if batch:
                write_batch(batch, is_first_batch=first_batch)
            worker.close()
        else:
            input_queue = mp.Queue(maxsize=100)
            output_queue = mp.Queue(maxsize=100)
            serializer_kwargs = {
                "write_dir": self._storage_dir / self._config_name,
                "split": split_iterator.split.value,
                "tf": split_iterator._tf,
                "write_binary_tars": True,
            }

            producers = [
                ProducerWorker(input_queue, output_queue, serializer_kwargs)
                for _ in range(self._num_processes - 1)
            ]

            consumer = ConsumerWorker(output_queue, write_batch, self._write_batch_size)

            consumer.start()
            for p in producers:
                p.start()

            for item in tqdm.tqdm(
                iterator, desc="Writing to Deltalake", total=total, unit="rows"
            ):
                input_queue.put(item)

            for _ in producers:
                input_queue.put("STOP")

            for p in producers:
                p.join()

            output_queue.put("STOP")

            consumer.join()

        split_iterator.enable_tf()

        return split_dir
