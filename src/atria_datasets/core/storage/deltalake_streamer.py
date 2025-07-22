import logging
from collections.abc import Sequence
from typing import Any

from deltalake import DeltaTable
from pyarrow.dataset import Dataset as ArrowDataset

from atria_datasets.core.typing.common import T_BaseDataInstance

logger = logging.getLogger(__name__)


# can only work with no worker for now
class DeltalakeOnlineStreamReader(Sequence[T_BaseDataInstance]):
    def __init__(
        self,
        path: str,
        data_model: type[T_BaseDataInstance],
        allowed_keys: set[str] | None = None,
        storage_options: dict | None = None,
        presign_expiry: int = 3600,  # seconds
    ):
        self.path = path
        self.repo_name = self.path.split("/")[2]
        self.branch_name = self.path.split("/")[3]
        self.data_model = data_model
        self.allowed_keys = allowed_keys
        self.storage_options = storage_options or {}
        self.presign_expiry = presign_expiry
        self._length = (
            DeltaTable(path, storage_options=self.storage_options)
            .to_pyarrow_dataset()
            .count_rows()
        )
        self._s3_client: Any | None = None
        self._pa_dataset: ArrowDataset | None = None

    def _initialize_dataset(self) -> ArrowDataset:
        if self._pa_dataset is None:
            self._pa_dataset = DeltaTable(
                self.path, storage_options=self.storage_options
            ).to_pyarrow_dataset()

    def _initialize_s3_client(self):
        if self._s3_client is None:
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

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> T_BaseDataInstance:  # type: ignore
        if isinstance(index, list):
            return self.__getitems__(index)
        self._initialize_dataset()
        self._initialize_s3_client()
        rows = self._pa_dataset.take([index]).to_pylist()  # type: ignore
        return self._process_rows(rows)[0]

    def __getitems__(self, indices: list[int]) -> list[T_BaseDataInstance]:
        self._initialize_dataset()
        self._initialize_s3_client()

        rows = self._pa_dataset.take(indices).to_pylist()  # type: ignore
        return self._process_rows(rows)

    def _process_rows(self, rows: list[dict]) -> list[T_BaseDataInstance]:
        processed_rows = []
        for row in rows:
            if self.allowed_keys is not None:
                row = {k: v for k, v in row.items() if k in self.allowed_keys}
            row = self._convert_paths_to_presigned_urls(row)
            instance = self.data_model.from_row(row)
            processed_rows.append(instance)
        return processed_rows

    def _convert_paths_to_presigned_urls(self, row: dict) -> dict:
        for key, value in row.items():
            if isinstance(value, str) and "file_path" in key:
                try:
                    presigned_url = self._s3_client.generate_presigned_url(  # type: ignore
                        "get_object",
                        Params={
                            "Bucket": self.repo_name,
                            "Key": f"{self.branch_name}/" + str(value),
                        },
                        ExpiresIn=self.presign_expiry,
                    )
                    row[key] = presigned_url
                except Exception as e:
                    logger.warning(f"Failed to generate presigned URL for {value}: {e}")
        return row
