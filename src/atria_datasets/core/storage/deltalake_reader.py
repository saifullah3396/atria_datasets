import logging
from collections.abc import Sequence

import pandas as pd
import pyarrow.dataset as ds
from deltalake import DeltaTable

from atria_datasets.core.dataset.atria_dataset import DatasetLoadingMode
from atria_datasets.core.typing.common import T_BaseDataInstance

logger = logging.getLogger(__name__)


def transform_file_path(value: str, storage_dir: str, config_name: str) -> str:
    return f"{storage_dir}/{config_name}/raw/{value}"


class DeltalakeReader(Sequence[T_BaseDataInstance]):
    def __init__(
        self,
        table_path: str,
        data_model: type[T_BaseDataInstance],
        allowed_keys: set[str] | None = None,
    ):
        self.path = table_path
        self.data_model = data_model
        self.allowed_keys = allowed_keys
        self._length = 0

    @classmethod
    def from_mode(
        cls,
        mode: DatasetLoadingMode,
        table_path: str,
        data_model: type[T_BaseDataInstance],
        allowed_keys: set[str] | None = None,
        storage_dir: str | None = None,
        config_name: str | None = None,
        storage_options: dict | None = None,
        presign_expiry: int = 3600,
    ) -> "DeltalakeReader":
        kwargs = {
            "table_path": table_path,
            "data_model": data_model,
            "allowed_keys": allowed_keys,
            "storage_dir": storage_dir,
            "config_name": config_name,
            "storage_options": storage_options,
            "presign_expiry": presign_expiry,
        }

        if mode == DatasetLoadingMode.in_memory:
            return InMemoryDeltalakeReader(**kwargs)
        elif mode == DatasetLoadingMode.local_streaming:
            return LocalDeltalakeReader(**kwargs)
        elif mode == DatasetLoadingMode.online_streaming:
            return OnlineDeltalakeReader(**kwargs)
        else:
            raise ValueError(f"Unsupported loading mode: {mode}")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> T_BaseDataInstance:
        if isinstance(index, list):
            return self.__getitems__(index)
        return self._load_and_process_rows([index])[0]

    def __getitems__(self, indices: list[int]) -> list[T_BaseDataInstance]:
        return self._load_and_process_rows(indices)

    def _process_row(self, row: dict) -> T_BaseDataInstance:
        if self.allowed_keys:
            row = {k: v for k, v in row.items() if k in self.allowed_keys}
        return self.data_model.from_row(row)

    def _load_and_process_rows(self, indices: list[int]) -> list[T_BaseDataInstance]:
        raise NotImplementedError

    def dataframe(self) -> pd.DataFrame:
        raise NotImplementedError


class InMemoryDeltalakeReader(DeltalakeReader):
    def __init__(self, *args, **kwargs):
        self.storage_dir = kwargs.pop("storage_dir", None)
        self.config_name = kwargs.pop("config_name", None)
        self.storage_options = kwargs.pop("storage_options", None)
        super().__init__(*args, **kwargs)
        self._df = DeltaTable(
            self.path, storage_options=self.storage_options
        ).to_pandas()
        if self.allowed_keys:
            self.allowed_keys = [
                col
                for col in self._df.columns
                if col.startswith(tuple(self.allowed_keys))
            ]
        self._length = len(self._df)

    def _load_and_process_rows(self, indices: list[int]) -> list[T_BaseDataInstance]:
        rows = self._df.iloc[indices]
        if self.allowed_keys:
            rows = rows[self.allowed_keys]
        row_dicts = rows.to_dict(orient="records")
        for row in row_dicts:
            for key, value in row.items():
                if (
                    isinstance(value, str)
                    and "file_path" in key
                    and self.storage_dir
                    and self.config_name
                ):
                    row[key] = transform_file_path(
                        value, self.storage_dir, self.config_name
                    )
        return [self._process_row(row) for row in row_dicts]

    def dataframe(self) -> pd.DataFrame:
        return self._df


class LocalDeltalakeReader(DeltalakeReader):
    def __init__(self, *args, **kwargs):
        from pyarrow.fs import S3FileSystem

        self.storage_dir = kwargs.pop("storage_dir", None)
        self.config_name = kwargs.pop("config_name", None)
        self.storage_options = kwargs.pop("storage_options", None)
        super().__init__(*args, **kwargs)
        delta = DeltaTable(self.path, storage_options=self.storage_options)
        file_uris = delta.file_uris()
        if file_uris[0].startswith("lakefs://"):
            filesystem = S3FileSystem(
                endpoint_override=self.storage_options["AWS_ENDPOINT"],
                access_key=self.storage_options["AWS_ACCESS_KEY_ID"],
                secret_key=self.storage_options["AWS_SECRET_ACCESS_KEY"],
            )
        file_uris = [file_uri.replace("lakefs://", "") for file_uri in file_uris]
        self._dataset = ds.dataset(file_uris, filesystem=filesystem)
        self._length = self._dataset.count_rows()

    def _load_and_process_rows(self, indices: list[int]) -> list[T_BaseDataInstance]:
        row_dicts = self._dataset.take(indices).to_pylist()
        for row in row_dicts:
            for key, value in row.items():
                if (
                    isinstance(value, str)
                    and "file_path" in key
                    and self.storage_dir
                    and self.config_name
                ):
                    row[key] = transform_file_path(
                        value, self.storage_dir, self.config_name
                    )
        return [self._process_row(row) for row in row_dicts]

    def dataframe(self) -> pd.DataFrame:
        return self._dataset.to_table().to_pandas()


class OnlineDeltalakeReader(LocalDeltalakeReader):
    def __init__(self, *args, **kwargs):
        self.presign_expiry = kwargs.pop("presign_expiry", 3600)
        super().__init__(*args, **kwargs)
        self._initialize_s3_client()
        self._extract_s3_path_parts()

    def _initialize_s3_client(self):
        import boto3
        from botocore.config import Config

        self._s3_client = boto3.client(
            "s3",
            endpoint_url=self.storage_options.get("AWS_ENDPOINT"),
            aws_access_key_id=self.storage_options.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=self.storage_options.get("AWS_SECRET_ACCESS_KEY"),
            region_name="stub",
            config=Config(signature_version="s3v4"),
        )

    def _extract_s3_path_parts(self):
        parts = self.path.split("/")
        self.repo_name = parts[2]
        self.branch_name = parts[3]
        self.config_name = parts[4]

    def _generate_presigned_url(self, value: str) -> str:
        try:
            return self._s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.repo_name,
                    "Key": f"{self.branch_name}/{self.config_name}/{value}",
                },
                ExpiresIn=self.presign_expiry,
            )
        except Exception as e:
            logger.warning(f"Failed to generate presigned URL for {value}: {e}")
            return value

    def _load_and_process_rows(self, indices: list[int]) -> list[T_BaseDataInstance]:
        row_dicts = self._dataset.take(indices).to_pylist()
        for row in row_dicts:
            for key, value in row.items():
                if isinstance(value, str) and "file_path" in key:
                    row[key] = self._generate_presigned_url(value)
        return [self._process_row(row) for row in row_dicts]
