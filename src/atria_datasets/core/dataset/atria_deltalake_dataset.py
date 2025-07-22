"""
Atria Hub Dataset Module

This module provides dataset classes for loading and managing datasets from the Atria Hub.
It includes support for both streaming and local dataset access, with specialized classes
for different data types (images and documents).

Classes:
    - AtriaHubDataset: Base class for datasets loaded from the Atria Hub
    - AtriaHubImageDataset: Specialized dataset for image data
    - AtriaHubDocumentDataset: Specialized dataset for document data

Dependencies:
    - atria_hub: Required for hub connectivity and dataset operations
    - omegaconf: For configuration management
    - pathlib: For file path operations
    - atria_core: For core utilities and types

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from atria_core.constants import _DEFAULT_ATRIA_DATASETS_CACHE_DIR
from atria_core.logger import get_logger
from atria_core.types import (
    DatasetSplitType,
    DocumentInstance,
    ImageInstance,
    SplitConfig,
)
from atria_core.types.datasets.metadata import DatasetMetadata
from omegaconf import DictConfig, OmegaConf

from atria_datasets.core.dataset.atria_dataset import AtriaDataset
from atria_datasets.core.dataset.exceptions import SplitNotFoundError
from atria_datasets.core.dataset.split_iterator import SplitIterator
from atria_datasets.core.storage.utilities import FileStorageType
from atria_datasets.core.typing.common import T_BaseDataInstance

if TYPE_CHECKING:
    from atria_hub.api.datasets import (
        Dataset as DatasetInfo,  # type: ignore[import-not-found]
    )
    from atria_hub.hub import AtriaHub  # type: ignore[import-not-found]

logger = get_logger(__name__)


class AtriaDeltalakeDataset(AtriaDataset[T_BaseDataInstance]):
    def __init__(
        self,
        username: str,
        dataset_name: str,
        branch: str | None = None,
        streaming: bool = False,
    ) -> None:
        """
        Initialize the AtriaHubDataset.

        Args:
            username: The username of the dataset owner on Atria Hub
            dataset_name: The name of the dataset
            branch: The branch/version to use. If None, uses the default branch
            streaming: Whether to stream data directly from hub (True) or download locally (False)

        Raises:
            ImportError: If atria_hub package is not installed
            ValueError: If dataset configuration is invalid
        """
        # Core dataset identifiers
        self._username = username
        self._dataset_name = dataset_name
        self._branch = branch
        self._streaming = streaming

        # Derived configuration
        self._config_name = branch.split("-")[0] if branch else None

        # Initialize hub connection and dataset info
        self._hub = self._initialize_hub()
        self._dataset_info = self._initialize_dataset_info()
        self._config = self._initialize_config()

        # Setup storage paths
        self._data_dir = self._setup_data_dir()
        self._storage_dir = self._setup_storage_dir()

        # Internal state management
        self._split_iterators: dict[DatasetSplitType, SplitIterator] = {}
        self._downloads_prepared: bool = False
        self._sharded_splits_prepared: bool = False

    # ==================== Properties ====================

    @property
    def config(self) -> DictConfig:
        """
        Get the dataset configuration.

        Returns:
            DictConfig: The dataset configuration loaded from the hub
        """
        return self._config

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        """
        Get the data model class for this dataset.

        Returns:
            type[T_BaseDataInstance]: The data model class (DocumentInstance or ImageInstance)

        Raises:
            ValueError: If the data instance type is not supported
        """
        from atria_hub.api.datasets import DataInstanceType

        if self._dataset_info.data_instance_type == DataInstanceType.DOCUMENT_INSTANCE:
            return DocumentInstance
        elif self._dataset_info.data_instance_type == DataInstanceType.IMAGE_INSTANCE:
            return ImageInstance
        else:
            raise ValueError(
                f"Unsupported data instance type: {self._dataset_info.data_instance_type}. "
                "Supported types are 'document' and 'image'."
            )

    @property
    def username(self) -> str:
        """Get the dataset owner's username."""
        return self._username

    @property
    def dataset_name(self) -> str:
        """Get the dataset name."""
        return self._dataset_name

    @property
    def branch(self) -> str | None:
        """Get the dataset branch."""
        return self._branch

    @property
    def streaming(self) -> bool:
        """Check if dataset is in streaming mode."""
        return self._streaming

    # ==================== Initialization Methods ====================

    def _initialize_hub(self) -> "AtriaHub":
        """
        Initialize the Atria Hub connection.

        Returns:
            AtriaHub: Initialized hub connection

        Raises:
            ImportError: If atria_hub package is not available
            Exception: If hub initialization fails
        """
        try:
            from atria_hub.hub import AtriaHub  # type: ignore[import-not-found]

            return AtriaHub().initialize()
        except ImportError:
            raise ImportError(
                "The 'atria_hub' package is required to load datasets from the hub. "
                "Please install it using 'uv add https://github.com/saifullah3396/atria_hub'."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Atria Hub: {e}")
            raise e

    def _initialize_dataset_info(self) -> "DatasetInfo":
        """
        Retrieve dataset information from the hub.

        Returns:
            DatasetInfo: Dataset metadata and information
        """
        from atria_hub.api.datasets import Dataset as DatasetInfo

        self._dataset_info: DatasetInfo = self._hub.datasets.get_by_name(
            username=self._username, name=self._dataset_name
        )
        return self._dataset_info

    def _initialize_config(self) -> DictConfig:
        """
        Load and initialize the dataset configuration.

        Returns:
            DictConfig: The dataset configuration from the hub
        """
        if self._branch is None:
            self._branch = self._dataset_info.default_branch
            self._config_name = self._branch.split("-")[0]

        config = self._hub.datasets.get_config(
            self._dataset_info.repo_id, branch=self._branch
        )
        return OmegaConf.create(config)

    def _setup_data_dir(self) -> Path:
        """
        Set up the local data directory path.

        Returns:
            Path: Path to the local data directory
        """
        return _DEFAULT_ATRIA_DATASETS_CACHE_DIR / self._username / self._dataset_name

    def _setup_storage_dir(self) -> Path:
        """
        Set up the storage directory path.

        Returns:
            Path: Path to the storage directory
        """
        return self._data_dir

    # ==================== Path Management ====================

    def split_dir(self, split: DatasetSplitType) -> Path:
        """
        Get the directory path for a specific dataset split.

        Args:
            split: The dataset split type

        Returns:
            Path: Path to the split directory
        """
        return Path(self._storage_dir) / f"delta/{split.value}"

    # ==================== Dataset Existence Checks ====================

    def dataset_exists(self) -> bool:
        """
        Check if the dataset exists locally.

        Returns:
            bool: True if dataset exists locally, False otherwise
        """
        return (Path(self._storage_dir) / "delta/").exists()

    def split_exists(self, split: DatasetSplitType) -> bool:
        """
        Check if a specific split exists locally.

        Args:
            split: The dataset split to check

        Returns:
            bool: True if split exists locally, False otherwise
        """
        return self.split_dir(split).exists()

    # ==================== Dataset Preparation ====================

    def prepare_downloads(self, data_dir: str, access_token: str | None = None) -> None:
        """
        Download dataset files from the hub to local storage.

        Args:
            data_dir: Directory where dataset files should be stored
            access_token: Optional access token for private datasets

        Note:
            This method is only effective when streaming=False. For streaming datasets,
            no downloads are performed.
        """
        if not self._streaming and not self._downloads_prepared:
            if self.dataset_exists():
                logger.info(
                    f"Dataset {self._dataset_name} already exists in {self._storage_dir}. "
                    "Skipping download."
                )
                return

            logger.info(
                f"Downloading dataset {self._dataset_name} to {self._storage_dir}"
            )
            self._hub.datasets.download_files(
                dataset_repo_id=self._dataset_info.repo_id,
                branch=self._branch,
                destination_path=str(self._storage_dir),
            )
            self._downloads_prepared = True

    # ==================== Class Methods ====================

    @classmethod
    def _validate_dataset_name(cls, name: str) -> tuple[str, str, str | None]:
        """
        Validate and parse dataset name format.

        Args:
            name: Dataset name in format 'username/dataset_name' or 'username/dataset_name/branch'

        Returns:
            tuple: (username, dataset_name, branch) where branch can be None

        Raises:
            ValueError: If dataset name format is invalid
        """
        if "/" not in name:
            raise ValueError(
                f"Invalid dataset name format: {name}. "
                "Expected format is 'username/dataset_name' or 'username/dataset_name/branch'."
            )

        parts = name.split("/")
        if len(parts) == 2:
            return parts[0], parts[1], None
        elif len(parts) == 3:
            return parts[0], parts[1], parts[2]
        else:
            raise ValueError(
                f"Invalid dataset name format: {name}. "
                "Expected format is 'username/dataset_name' or 'username/dataset_name/branch'."
            )

    @classmethod
    def load_from_hub(
        cls,
        name: str,
        split: DatasetSplitType | None = None,
        preprocess_transform: Callable | None = None,
        access_token: str | None = None,
        overwrite_existing_shards: bool = False,
        allowed_keys: set[str] | None = None,
        shard_storage_type: FileStorageType | None = None,
        sharded_storage_kwargs: dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> "AtriaHubDataset[T_BaseDataInstance]":
        """
        Load a dataset from the Atria Hub.

        Args:
            name: Dataset name in format 'username/dataset_name' or 'username/dataset_name/branch'
            split: Specific split to load. If None, loads all available splits
            preprocess_transform: Optional preprocessing function to apply to data
            access_token: Access token for private datasets
            overwrite_existing_shards: Whether to overwrite existing local shards
            allowed_keys: Set of keys to include in the data. If None, includes all keys
            shard_storage_type: Type of storage for sharded data
            sharded_storage_kwargs: Additional arguments for sharded storage
            streaming: Whether to stream data from hub or download locally

        Returns:
            AtriaHubDataset: Configured dataset instance ready for use

        Raises:
            ValueError: If dataset name format is invalid
            ImportError: If required packages are not available

        Example:
            ```python
            # Load training split of a public dataset
            dataset = AtriaHubDataset.load_from_hub(
                name="username/my_dataset", split=DatasetSplitType.TRAIN
            )

            # Load with streaming enabled
            streaming_dataset = AtriaHubDataset.load_from_hub(
                name="username/my_dataset/v1.0", streaming=True
            )
            ```
        """
        username, dataset_name, branch = cls._validate_dataset_name(name)
        dataset = cls(
            username=username,
            dataset_name=dataset_name,
            branch=branch,
            streaming=streaming,
        )

        # Build the split for the dataset
        dataset.build(
            split=split,
            preprocess_transform=preprocess_transform,
            access_token=access_token,
            overwrite_existing_shards=overwrite_existing_shards,
            allowed_keys=allowed_keys,
            shard_storage_type=shard_storage_type,
            enable_cached_splits=False,
            **(sharded_storage_kwargs if sharded_storage_kwargs else {}),
        )

        return cast(AtriaHubDataset[T_BaseDataInstance], dataset)

    # ==================== Abstract Method Implementations ====================

    def get_max_split_samples(self, split: DatasetSplitType) -> int | None:
        """
        Get the maximum number of samples in a split.

        Args:
            split: The dataset split

        Returns:
            int | None: Maximum number of samples, or None if unlimited
        """
        return None

    def _metadata(self) -> "DatasetMetadata":
        """
        Get metadata for the dataset.

        This method should be implemented by subclasses to provide specific metadata
        for the dataset, such as description, version, and other relevant information.

        Returns:
            DatasetMetadata: Metadata object containing dataset information
        """

        metadata = self._hub.datasets.get_metadata(
            dataset_repo_id=self._dataset_info.repo_id, branch=self._branch
        )
        return DatasetMetadata(**metadata)

    def _split_configs(self, data_dir: str) -> list[SplitConfig]:
        """
        Get configuration for all available splits.

        Args:
            data_dir: Data directory path

        Returns:
            list[SplitConfig]: List of split configurations
        """
        return [
            SplitConfig(split=DatasetSplitType(split))
            for split in self._hub.datasets.get_splits(
                self._dataset_info.repo_id, self._branch
            )
        ]

    def _split_iterator(
        self, split: DatasetSplitType, allowed_keys: set[str] | None = None
    ) -> Sequence | Generator[Any, None, None]:
        """
        Create an iterator for a specific dataset split.

        Args:
            split: The dataset split to iterate over
            allowed_keys: Set of keys to include in the data

        Returns:
            Sequence | Generator: Iterator over the split data

        Raises:
            SplitNotFoundError: If the split doesn't exist locally in non-streaming mode
        """
        from atria_datasets.core.storage.deltalake_reader import DeltalakeReader
        from atria_datasets.core.storage.deltalake_streamer import (
            DeltalakeOnlineStreamReader,
        )

        if not self._streaming and not self.split_exists(split):
            raise SplitNotFoundError(
                f"Split {split.value} does not exist in the dataset {self._dataset_name}. "
                "Please ensure the dataset is downloaded or use streaming=True."
            )

        # Always include essential keys
        if allowed_keys is not None:
            allowed_keys = allowed_keys.copy()
            allowed_keys.update({"index", "sample_id"})

        if self._streaming:
            storage_options = self._hub.get_storage_options()
            path = self._hub.datasets.dataset_table_path(
                dataset_repo_id=self._dataset_info.repo_id,
                branch=self._branch,
                split=split.value,
            )
            logger.info(f"Streaming dataset split {split.value} from {path}")
            return DeltalakeOnlineStreamReader(
                path=path,
                data_model=self.data_model,
                allowed_keys=allowed_keys,
                storage_options=storage_options,
            )
        else:
            logger.debug(f"Reading dataset split {split.value} from local storage")
            return DeltalakeReader(  # type: ignore
                path=str(self.split_dir(split=split)),
                data_model=self.data_model,
                allowed_keys=allowed_keys,
            )


class AtriaHubImageDataset(AtriaHubDataset[ImageInstance]):
    """
    Specialized dataset class for handling image datasets from the Atria Hub.

    This class inherits from AtriaHubDataset and is specifically typed for ImageInstance
    data models, providing type safety and specialized functionality for image data.

    Example:
        ```python
        image_dataset = AtriaHubImageDataset.load_from_hub(
            name="username/image_dataset", split=DatasetSplitType.TRAIN
        )

        for image_instance in image_dataset:
            # image_instance is guaranteed to be of type ImageInstance
            process_image(image_instance.image)
        ```
    """

    __data_model__ = ImageInstance


class AtriaHubDocumentDataset(AtriaHubDataset[DocumentInstance]):
    """
    Specialized dataset class for handling document datasets from the Atria Hub.

    This class inherits from AtriaHubDataset and is specifically typed for DocumentInstance
    data models, providing type safety and specialized functionality for document data.

    Example:
        ```python
        doc_dataset = AtriaHubDocumentDataset.load_from_hub(
            name="username/document_dataset", split=DatasetSplitType.TRAIN
        )

        for doc_instance in doc_dataset:
            # doc_instance is guaranteed to be of type DocumentInstance
            process_document(doc_instance.text)
        ```
    """

    __data_model__ = DocumentInstance
