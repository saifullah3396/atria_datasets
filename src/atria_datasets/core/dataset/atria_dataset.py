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

from abc import abstractmethod
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import Any, Generic, cast

import rich
import rich.pretty
from atria_core.constants import _DEFAULT_ATRIA_DATASETS_CACHE_DIR
from atria_core.logger import get_logger
from atria_core.types import (
    BaseDataInstance,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    ImageInstance,
    SplitConfig,
)
from atria_core.utilities.repr import RepresentationMixin
from atria_registry.constants import _PROVIDER_NAME
from atria_registry.registry_config_mixin import RegistryConfigMixin
from omegaconf import OmegaConf

from atria_datasets.core.constants import _DEFAULT_DOWNLOAD_PATH
from atria_datasets.core.dataset.exceptions import SplitNotFoundError
from atria_datasets.core.dataset.split_iterator import SplitIterator
from atria_datasets.core.storage.utilities import FileStorageType
from atria_datasets.core.typing.common import T_BaseDataInstance

logger = get_logger(__name__)


class AtriaDataset(
    Generic[T_BaseDataInstance], RepresentationMixin, RegistryConfigMixin
):
    """
    Base class for datasets in the Atria application.

    This class provides functionality for managing dataset splits, configurations,
    metadata, runtime transformations, and storage management.

    Attributes:
        _dataset_name (str): The name of the dataset.
        _config_name (str): The name of the dataset configuration.
        _description (str | None): A brief description of the dataset.
        _version (str): The version of the dataset.
        _data_urls (Union[str, List[str], Dict[str, str]] | None): The URLs for accessing
            the dataset. Can be a single URL, a list of URLs, or a dictionary mapping
            keys to URLs.
        _streaming_mode (bool): Indicates whether the dataset should be loaded in
        _downloaded_files (Dict[str, Path]): A dictionary of downloaded files.
        _prepared_split_iterator (Iterator): The prepared iterator for the active split.
        _download_dir (Path): The directory for downloaded files.
        _download_manager (DownloadManager): The download manager for the dataset.
    """

    __data_model__ = BaseDataInstance
    __default_config_path__ = "conf/dataset/config.yaml"
    __default_metadata_path__ = "metadata.yaml"

    def __init__(
        self,
        dataset_name: str | None = None,
        config_name: str = "main",
        data_urls: str | list[str] | dict[str, str] | dict[str, tuple] | None = None,
        max_train_samples: int | None = None,
        max_validation_samples: int | None = None,
        max_test_samples: int | None = None,
        **kwargs,
    ):
        """
        Initializes the AtriaDataset.

        Args:
            name (str | None): The name of the dataset. Defaults to None.
            config_name (str): The name of the dataset configuration. Defaults to "default".
            data_urls (Union[str, List[str], Dict[str, str]] | None): The URLs for accessing
                the dataset. Can be a single URL, a list of URLs, or a dictionary mapping
                keys to URLs. Defaults to None.
            max_train_samples (Optional[int]): The maximum number of training samples to load.
            max_validation_samples (Optional[int]): The maximum number of validation samples to load.
            max_test_samples (Optional[int]): The maximum number of test samples to load.
        """
        from atria_datasets.core.dataset.split_iterator import SplitIterator

        # config parameters
        self._dataset_name = dataset_name or self.__class__.__name__.lower()
        self._config_name = config_name
        self._data_urls = data_urls
        self._max_train_samples = max_train_samples
        self._max_validation_samples = max_validation_samples
        self._max_test_samples = max_test_samples

        # internal parameters
        self._data_dir = self._setup_data_dir()
        self._storage_dir = self._setup_storage_dir()
        self._downloaded_files: dict[str, Path] = {}
        self._split_iterators: dict[DatasetSplitType, SplitIterator] = {}
        self._downloads_prepared: bool = False
        self._sharded_splits_prepared: bool = False

        RegistryConfigMixin.__init__(
            self,
            dataset_name=self._dataset_name,
            config_name=self._config_name,
            data_urls=self._data_urls,
            max_train_samples=self._max_train_samples,
            max_validation_samples=self._max_validation_samples,
            max_test_samples=self._max_test_samples,
            **kwargs,
        )

    @property
    def name(self) -> str:
        """
        Returns the name of the dataset.

        Returns:
            str: The name of the dataset.
        """
        return self._dataset_name

    @property
    def config_name(self) -> str:
        """
        Returns the configuration name of the dataset.

        Returns:
            str: The configuration name of the dataset.
        """
        return self._config_name

    @property
    def downloaded_files(self) -> dict[str, Path]:
        """
        Returns the dictionary of downloaded files.

        Returns:
            Dict[str, Path]: The downloaded files.
        """
        return self._downloaded_files

    @property
    def metadata(self) -> DatasetMetadata:
        """
        Returns the metadata for the dataset.

        Returns:
            DatasetMetadata: The dataset metadata.
        """
        metadata = self._metadata()
        return metadata

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        """
        Returns the data model class for the dataset.

        Returns:
            type[T_BaseDataInstance]: The data model class.
        """
        return self.__data_model__

    @property
    def data_dir(self) -> Path:
        """
        Returns the directory where the dataset files are stored.

        Returns:
            Path: The data directory.
        """
        return self._data_dir

    @classmethod
    def load_from_registry(
        cls,
        name: str,
        split: DatasetSplitType | None = None,
        config_name: str | None = None,
        data_dir: str | None = None,
        provider: str | None = _PROVIDER_NAME,
        preprocess_transform: Callable | None = None,
        shard_storage_type: FileStorageType | None = None,
        access_token: str | None = None,
        overwrite_existing_cached: bool = False,
        overwrite_existing_shards: bool = False,
        allowed_keys: set[str] | None = None,
        build_kwargs: dict[str, Any] | None = None,
        sharded_storage_kwargs: dict[str, Any] | None = None,
    ) -> "AtriaDataset[T_BaseDataInstance]":
        from atria_datasets import DATASET

        module_name = f"{name}/{config_name}" if config_name is not None else name
        logger.info(f"Loading dataset {module_name} from registry.")
        dataset: AtriaDataset[T_BaseDataInstance] = DATASET.load_from_registry(
            module_name=module_name,
            provider=provider,
            return_config=False,
            **(build_kwargs if build_kwargs else {}),
        )
        assert dataset.__data_model__ == cls.__data_model__, (
            "The data model of the loaded dataset does not match the expected data model. "
            f"Expected: {cls.__data_model__}, "
            f"Got: {dataset.__data_model__}."
        )
        dataset.build_split(
            split=split,
            data_dir=data_dir,
            preprocess_transform=preprocess_transform,
            shard_storage_type=shard_storage_type,
            access_token=access_token,
            overwrite_existing_cached=overwrite_existing_cached,
            overwrite_existing_shards=overwrite_existing_shards,
            allowed_keys=allowed_keys,
            **(sharded_storage_kwargs if sharded_storage_kwargs else {}),
        )
        return cast(AtriaDataset[T_BaseDataInstance], dataset)

    def build_split(
        self,
        split: DatasetSplitType,
        data_dir: str | None = None,
        runtime_transforms: Callable | None = None,
        preprocess_transform: Callable | None = None,
        access_token: str | None = None,
        overwrite_existing_cached: bool = False,
        overwrite_existing_shards: bool = False,
        allowed_keys: set[str] | None = None,
        **sharded_storage_kwargs,
    ) -> None:
        # prepare the dataset directory
        if data_dir is not None:
            self._data_dir = self._validate_data_dir(data_dir)
            self._storage_dir = self._setup_storage_dir()

        # prepare the cached splits in deltalake storage
        self._prepare_cached_splits(
            split=split,
            access_token=access_token,
            overwrite_existing=overwrite_existing_cached,
            allowed_keys=allowed_keys,
        )

        shard_storage_type = sharded_storage_kwargs.get("shard_storage_type", None)
        if shard_storage_type is not None:
            self._prepare_sharded_splits(
                shard_storage_type=shard_storage_type,
                preprocess_transform=preprocess_transform,
                overwrite_existing=overwrite_existing_shards,
                allowed_keys=allowed_keys,
                **sharded_storage_kwargs,
            )

        if runtime_transforms is not None:
            for split_iterator in self._split_iterators.values():
                split_iterator.output_transform = runtime_transforms

    def prepare_downloads(self, data_dir: str, access_token: str | None = None) -> None:
        """
        Prepares the dataset by downloading and extracting files if data URLs are provided.

        Args:
            data_dir (str): The directory where the dataset files are stored.

        Returns:
            None
        """

        if not self._downloads_prepared:
            from atria_datasets.core.download_manager.download_manager import (
                DownloadManager,
            )

            download_dir = Path(data_dir) / _DEFAULT_DOWNLOAD_PATH
            download_dir.mkdir(parents=True, exist_ok=True)
            download_manager = DownloadManager(
                data_dir=Path(data_dir), download_dir=download_dir
            )

            if self._data_urls is not None:
                self._downloaded_files = download_manager.download_and_extract(
                    str(self._data_urls)
                )

            self._downloads_prepared = True

    def save_dataset_info(self) -> None:
        import yaml

        def write_yaml_file(file_path: Path, data: dict):
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)

        # write the dataset config and metadata
        write_yaml_file(
            self._storage_dir / self.__default_config_path__,
            OmegaConf.to_container(self.config),  # type: ignore
        )

        # write the dataset metadata
        write_yaml_file(
            self._storage_dir / self.__default_metadata_path__,
            self.metadata.model_dump(),
        )

    def get_dataset_files_from_dir(
        self, storage_dir: Path | str | None = None
    ) -> list[tuple[str, str]]:
        from atria_datasets.core.storage.deltalake_storage_manager import (
            DeltalakeStorageManager,
        )

        # we use the provided data_dir if it is not None, otherwise we use the default data_dir
        storage_dir = Path(storage_dir) if storage_dir else self._storage_dir

        # first we check that a delta folder exists in the data_dir
        deltalake_storage_manager = DeltalakeStorageManager(
            storage_dir=str(storage_dir)
        )

        # prepare dataset split files
        dataset_files = deltalake_storage_manager.get_split_files()

        # prepare dataset metadata and config files
        dataset_files += [
            (
                str(self._storage_dir / self.__default_metadata_path__),
                self.__default_metadata_path__,
            ),
            (
                str(self._storage_dir / self.__default_config_path__),
                self.__default_config_path__,
            ),
        ]

        return dataset_files

    def get_split_config(self, split: DatasetSplitType) -> SplitConfig:
        from atria_datasets.core.dataset.exceptions import SplitNotFoundError

        split_config = next(
            (
                split_config
                for split_config in self._split_configs(data_dir=str(self._data_dir))
                if split_config.split == split
            ),
            None,
        )
        if split_config is None:
            raise SplitNotFoundError(split.value)
        return split_config

    def get_max_split_samples(self, split: DatasetSplitType) -> int | None:
        """
        Returns the maximum number of samples for the active split.

        Returns:
            Optional[int]: The maximum number of samples for the active split.
        """
        if split == DatasetSplitType.train:
            return self._max_train_samples
        elif split == DatasetSplitType.validation:
            return self._max_validation_samples
        elif split == DatasetSplitType.test:
            return self._max_test_samples
        return None

    def prepare_split_iterator(self, split_config: SplitConfig) -> SplitIterator:
        return SplitIterator(
            split=split_config.split,
            base_iterator=self._split_iterator(
                split=split_config.split, **split_config.gen_kwargs
            ),
            input_transform=self._input_transform,
            data_model=self.data_model,
            max_len=self.get_max_split_samples(split=split_config.split),
        )

    def _setup_data_dir(self) -> Path:
        return _DEFAULT_ATRIA_DATASETS_CACHE_DIR / self._dataset_name

    def _validate_data_dir(self, data_dir: str | Path) -> Path:
        if Path(data_dir).exists():
            assert Path(data_dir).is_dir(), (
                f"Data directory {data_dir} is not a directory."
            )
        else:
            logger.warning(
                f"Data directory {data_dir} does not exist. Creating a new directory."
            )
            Path(data_dir).mkdir(parents=True, exist_ok=True)
        return Path(data_dir)

    def _setup_storage_dir(self) -> Path:
        storage_dir = self._data_dir / self._config_name
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        return storage_dir

    def _prepare_sharded_splits(
        self,
        shard_storage_type: FileStorageType,
        preprocess_transform: Callable | None = None,
        overwrite_existing: bool = False,
        allowed_keys: set[str] | None = None,
        **kwargs,
    ) -> None:
        if not self._sharded_splits_prepared:
            from atria_datasets.core.dataset.exceptions import SplitNotFoundError
            from atria_datasets.core.storage.sharded_dataset_storage_manager import (
                ShardedDatasetStorageManager,
            )

            sharded_dataset_storage_manager = ShardedDatasetStorageManager(
                storage_dir=str(self._storage_dir),
                storage_type=shard_storage_type,
                **kwargs,
            )
            for split, split_iterator in self._split_iterators.items():
                split_exists = sharded_dataset_storage_manager.split_exists(split=split)
                if split_exists and overwrite_existing:
                    logger.warning(
                        f"Overwriting existing shraded dataset split {split.value} as overwrite_existing is set to True."
                    )
                    sharded_dataset_storage_manager.purge_split(split)

                if not split_exists or overwrite_existing:
                    try:
                        # write split to storage
                        sharded_dataset_storage_manager.write_split(
                            split_iterator=split_iterator,
                            preprocess_transform=preprocess_transform,
                        )
                    except SplitNotFoundError:
                        pass

                # read split from storage
                self._split_iterators[split] = (
                    sharded_dataset_storage_manager.read_split(
                        split=split,
                        data_model=self.data_model,
                        allowed_keys=allowed_keys,
                    )
                )

            self._sharded_splits_prepared = True

    def _prepare_cached_splits(
        self,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
        write_batch_size: int = 100000,
        overwrite_existing: bool = False,
        allowed_keys: set[str] | None = None,
    ) -> None:
        from atria_datasets.core.dataset.exceptions import SplitNotFoundError
        from atria_datasets.core.storage.deltalake_storage_manager import (
            DeltalakeStorageManager,
        )

        if split is None:
            splits = list(DatasetSplitType)
        else:
            splits = [split]

        logger.info(f"Caching dataset to storage dir: {self._storage_dir}")
        deltalake_storage_manager = DeltalakeStorageManager(
            storage_dir=str(self._storage_dir), write_batch_size=write_batch_size
        )

        for split in splits:
            try:
                split_config = self.get_split_config(split=split)
            except SplitNotFoundError:
                logger.warning(
                    f"Split {split.value} not found in dataset configuration. Skipping split preparation."
                )
                continue

            split_exists = deltalake_storage_manager.split_exists(split=split)
            if split_exists and overwrite_existing:
                logger.warning(
                    f"Overwriting existing dataset split {split.value} as overwrite_existing is set to True."
                )
                deltalake_storage_manager.purge_split(split)
                split_exists = False

            if not split_exists:
                # prepare the downloads
                self.prepare_downloads(
                    data_dir=str(self._data_dir), access_token=access_token
                )

                # prepare dataset metadata and config
                self.save_dataset_info()

                try:
                    # write split to storage
                    deltalake_storage_manager.write_split(
                        split_iterator=self.prepare_split_iterator(
                            split_config=split_config
                        )
                    )
                except SplitNotFoundError:
                    pass

            if split_exists:
                logger.info(
                    f"Loading dataset split {split.value} from cached storage: "
                    f"{deltalake_storage_manager.split_dir(split=split)}"
                )

            # read split from storage
            self._split_iterators[split] = deltalake_storage_manager.read_split(
                split=split, data_model=self.data_model, allowed_keys=allowed_keys
            )

    def _metadata(self) -> DatasetMetadata:
        """
        Prepares the metadata for the dataset.

        Returns:
            DatasetMetadata: The prepared metadata object.
        """
        return DatasetMetadata()

    def __rich_repr__(self) -> rich.pretty.RichReprResult:
        """
        Generates a rich representation of the object.

        Yields:
            RichReprResult: A generator of key-value pairs or values for the object's attributes.
        """
        ignored_fields = [
            "_active_split_config",
            "_split_iterator",
            "_download_dir",
            "_prepared_metadata",
            "_download_manager",
            "_downloaded_files",
            "_enable_minimal_repr",
            "_prepared_split_iterator",
            "_downloads_prepared",
            "_sharded_splits_prepared",
            "_config",
        ]

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

    def _input_transform(self, sample: Any | T_BaseDataInstance) -> T_BaseDataInstance:
        """
        Transforms a sample into the dataset's data model.

        Args:
            sample (Union[Any, T_BaseDataInstance]): The sample to transform.

        Returns:
            T_BaseDataInstance: The transformed sample.
        """
        if isinstance(sample, self.data_model):
            return sample
        elif isinstance(sample, dict):
            return self.data_model(**sample)
        else:
            raise TypeError(
                f"Sample type {type(sample)} is not compatible with the data model {self.data_model}."
            )

    @abstractmethod
    def _split_configs(self, data_dir: str) -> list[SplitConfig]:
        """
        Abstract method to return the split configurations for the dataset.

        Args:
            data_dir (str): The directory where the dataset files are stored.

        Returns:
            List[SplitConfig]: The split configurations.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the `_split_configs` method."
        )

    @abstractmethod
    def _split_iterator(
        self, split: DatasetSplitType, **kwargs
    ) -> Sequence | Generator:
        """
        Abstract method to return the iterator for a dataset split.

        Args:
            split (DatasetSplit): The dataset split.
            **kwargs: Additional arguments for the iterator.

        Returns:
            Union[Iterator, Generator[T_BaseDataInstance, None, None]]: The split iterator.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the `_split_iterator` method."
        )

    @property
    def train(self) -> SplitIterator[T_BaseDataInstance]:
        """
        Returns the training split iterator.

        Returns:
            SplitIterator: The training split iterator.
        """
        split = self._split_iterators.get(DatasetSplitType.train, None)
        if split is None:
            raise SplitNotFoundError(DatasetSplitType.train.value)
        return split

    @train.setter
    def train(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        """
        Sets the training split iterator.

        Args:
            value (SplitIterator): The training split iterator to set.
        """
        self._split_iterators[DatasetSplitType.train] = value

    @property
    def validation(self) -> SplitIterator[T_BaseDataInstance]:
        """
        Returns the validation split iterator.

        Returns:
            SplitIterator: The validation split iterator.
        """
        split = self._split_iterators.get(DatasetSplitType.validation, None)
        if split is None:
            raise SplitNotFoundError(DatasetSplitType.validation.value)
        return split

    @validation.setter
    def validation(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        """
        Sets the validation split iterator.

        Args:
            value (SplitIterator): The validation split iterator to set.
        """
        self._split_iterators[DatasetSplitType.validation] = value

    @property
    def test(self) -> SplitIterator[T_BaseDataInstance] | None:
        """
        Returns the test split iterator.

        Returns:
            SplitIterator: The test split iterator.
        """
        split = self._split_iterators.get(DatasetSplitType.test, None)
        if split is None:
            raise SplitNotFoundError(DatasetSplitType.test.value)
        return split

    @test.setter
    def test(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        """
        Sets the test split iterator.

        Args:
            value (SplitIterator): The test split iterator to set.
        """
        self._split_iterators[DatasetSplitType.test] = value


class AtriaHubDataset(AtriaDataset[T_BaseDataInstance]):
    def build_split(  # type: ignore[override]
        self,
        split: DatasetSplitType,
        data_dir: str | None = None,
        runtime_transforms: Callable | None = None,
        preprocess_transform: Callable | None = None,
        access_token: str | None = None,
        overwrite_existing_shards: bool = False,
        allowed_keys: set[str] | None = None,
        streaming: bool = False,
        **sharded_storage_kwargs,
    ) -> None:
        # prepare the dataset directory
        if data_dir is not None:
            self._data_dir = self._validate_data_dir(data_dir)
            self._storage_dir = self._setup_storage_dir()

        # prepare the cached splits in deltalake storage
        self._prepare_cached_splits(
            split=split,
            access_token=access_token,
            streaming=streaming,
            allowed_keys=allowed_keys,
        )

        shard_storage_type = sharded_storage_kwargs.get("shard_storage_type", None)
        if shard_storage_type is not None:
            self._prepare_sharded_splits(
                shard_storage_type=shard_storage_type,
                preprocess_transform=preprocess_transform,
                overwrite_existing=overwrite_existing_shards,
                allowed_keys=allowed_keys,
                **sharded_storage_kwargs,
            )

        if runtime_transforms is not None:
            for split_iterator in self._split_iterators.values():
                split_iterator.output_transform = runtime_transforms

    def upload_to_hub(
        self,
        name: str | None = None,
        branch: str | None = None,
        is_public: bool = False,
    ) -> None:
        try:
            from atria_hub.api.datasets import DataInstanceType
            from atria_hub.hub import AtriaHub  # type: ignore[import-not-found]

            if name is None:
                name = self._dataset_name
            if branch is None:
                branch = self._config_name

            logger.info(
                f"Uploading dataset {self.__class__.__name__} to hub with name {name} and config {branch}."
            )

            def data_model_to_instance_type(
                data_model: type[T_BaseDataInstance],
            ) -> DataInstanceType:
                if data_model == DocumentInstance:
                    return DataInstanceType.DOCUMENT_INSTANCE
                elif data_model == ImageInstance:
                    return DataInstanceType.IMAGE_INSTANCE
                else:
                    raise ValueError(f"Unsupported data model: {data_model}")

            hub = AtriaHub()
            dataset = hub.datasets.get_or_create(
                name=name,
                description=self.metadata.description,
                data_instance_type=data_model_to_instance_type(self.data_model),
                is_public=is_public,
            )
            hub.datasets.upload_files(
                dataset=dataset,
                branch=branch,
                dataset_files=self.get_dataset_files_from_dir(),
            )
        except ImportError:
            raise ImportError(
                "The 'atria_hub' package is required to load datasets from the hub. "
                "Please install it using 'uv add https://github.com/saifullah3396/atria_hub'."
            )

    def prepare_downloads(self, data_dir: str, access_token: str | None = None) -> None:
        """
        Prepares the dataset by downloading and extracting files from the hub.

        Args:
            data_dir (str): The directory where the dataset files are stored.

        Returns:
            None
        """

        if not self._downloads_prepared:
            from atria_hub.hub import AtriaHub  # type: ignore[import-not-found]

            hub = AtriaHub()
            dataset = hub.datasets.get(name=self._dataset_name)
            logger.info(
                f"Loading dataset {self._dataset_name} from hub with branch {self._config_name} into storage directory {data_dir}."
            )

            # Download the dataset files from the hub
            hub.datasets.download_files(
                dataset=dataset, branch=self._config_name, destination_path=data_dir
            )
            self._downloads_prepared = True

    def _prepare_cached_splits(  # type: ignore[override]
        self,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
        allowed_keys: set[str] | None = None,
        streaming: bool = False,
    ) -> None:
        from atria_datasets.core.storage.deltalake_storage_manager import (
            DeltalakeStorageManager,
        )

        if split is None:
            splits = list(DatasetSplitType)
        else:
            splits = [split]

        logger.info(f"Caching dataset to storage dir: {self._storage_dir}")
        deltalake_storage_manager = DeltalakeStorageManager(
            storage_dir=str(self._storage_dir)
        )

        # prepare the downloads
        if not streaming:
            if not deltalake_storage_manager.dataset_exists():
                self.prepare_downloads(
                    data_dir=str(self._storage_dir), access_token=access_token
                )
            else:
                logger.info(
                    f"Dataset already exists in storage: {str(self._storage_dir)}. "
                    f"Skipping download."
                )

        for split in splits:
            split_exists = deltalake_storage_manager.split_exists(split=split)
            if split_exists:
                logger.info(
                    f"Loading dataset split {split.value} from cached storage: "
                    f"{deltalake_storage_manager.split_dir(split=split)}"
                )

                # read split from storage
                self._split_iterators[split] = deltalake_storage_manager.read_split(
                    split=split,
                    data_model=self.data_model,
                    allowed_keys=allowed_keys,
                    streaming_mode=streaming,
                )

    @classmethod
    def load_from_hub(
        cls,
        name: str,
        branch: str,
        split: DatasetSplitType | None = None,
        preprocess_transform: Callable | None = None,
        access_token: str | None = None,
        overwrite_existing_shards: bool = False,
        allowed_keys: set[str] | None = None,
        build_kwargs: dict[str, Any] | None = None,
        shard_storage_type: FileStorageType | None = None,
        sharded_storage_kwargs: dict[str, Any] | None = None,
        streaming: bool = False,
    ):
        try:
            import atria_hub  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "The 'atria_hub' package is required to load datasets from the hub. "
                "Please install it using 'uv add https://github.com/saifullah3396/atria_hub'."
            )

        dataset = cls(dataset_name=name, config_name=branch, **(build_kwargs or {}))

        # Build the split for the dataset
        dataset.build_split(
            split=split,
            preprocess_transform=preprocess_transform,
            access_token=access_token,
            overwrite_existing_shards=overwrite_existing_shards,
            allowed_keys=allowed_keys,
            streaming=streaming,
            shard_storage_type=shard_storage_type,
            **(sharded_storage_kwargs if sharded_storage_kwargs else {}),
        )

        return cast(AtriaDataset[T_BaseDataInstance], dataset)

    def _split_configs(self, data_dir: str) -> list[SplitConfig]:
        raise RuntimeError(
            "The `_split_configs` is undefined for AtriaHubDataset as it directly loads splits from the hub."
        )

    def _split_iterator(
        self, split: DatasetSplitType, **kwargs: Any
    ) -> Sequence | Generator[Any, None, None]:
        raise RuntimeError(
            "The `_split_iterator` is undefined for AtriaHubDataset as it directly loads splits from the hub."
        )


class AtriaImageDataset(AtriaHubDataset[ImageInstance]):
    """
    AtriaImageDataset is a specialized dataset class for handling image datasets.
    It inherits from AtriaDataset and provides additional functionality specific to image data.
    """

    __data_model__ = ImageInstance


class AtriaDocumentDataset(AtriaHubDataset[DocumentInstance]):
    """
    AtriaDocumentDataset is a specialized dataset class for handling document datasets.
    It inherits from AtriaDataset and provides additional functionality specific to document data.
    """

    __data_model__ = DocumentInstance
