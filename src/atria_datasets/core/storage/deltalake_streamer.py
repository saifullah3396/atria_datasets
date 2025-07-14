import logging
from collections.abc import Sequence

import boto3
from deltalake import DeltaTable

from atria_datasets.core.typing.common import T_BaseDataInstance

logger = logging.getLogger(__name__)


class DeltaStreamer(Sequence[T_BaseDataInstance]):
    def __init__(
        self,
        path: str,
        data_model: type[T_BaseDataInstance],
        allowed_keys: set[str] | None = None,
        storage_options: dict | None = None,
        presign_expiry: int = 3600,  # seconds
    ):
        self.path = path
        self.data_model = data_model
        self.allowed_keys = allowed_keys
        self.storage_options = storage_options or {}
        self.presign_expiry = presign_expiry

        # Get table length without loading all IDs
        delta_table = DeltaTable(path, storage_options=self.storage_options)
        self._length = delta_table.to_pyarrow_dataset().count_rows()

        # Worker-local DeltaTable instance (lazy)
        self._delta_table = None

    def _get_delta_table(self) -> DeltaTable:
        if self._delta_table is None:
            self._delta_table = DeltaTable(
                self.path, storage_options=self.storage_options
            )
        return self._delta_table

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self, idx: int | slice
    ) -> T_BaseDataInstance | list[T_BaseDataInstance]:
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(self._length)))
            return self.get_batch(indices)
        else:
            if idx < 0:
                idx = self._length + idx
            if idx < 0 or idx >= self._length:
                raise IndexError(
                    f"Index {idx} out of bounds for dataset with length {self._length}"
                )
            return self.get_batch([idx])[0]

    def get_batch(self, indices: list[int]) -> list[T_BaseDataInstance]:
        """Efficiently load a batch of samples by their indices."""
        if not indices:
            return []

        # Validate indices
        for idx in indices:
            if idx < 0 or idx >= self._length:
                raise IndexError(
                    f"Index {idx} out of bounds for dataset with length {self._length}"
                )

        delta_table = self._get_delta_table()

        # Get columns to fetch
        columns = list(self.allowed_keys) if self.allowed_keys else None

        # Convert to Arrow dataset and take specific indices
        dataset = delta_table.to_pyarrow_dataset(columns=columns)
        table = dataset.take(indices)

        # Convert to list of dicts
        rows = table.to_pylist()

        # Lazily create boto3 client if needed
        if self.boto3_client is None:
            self.boto3_client = boto3.client("s3")

        # Process each row
        result = []
        for row_dict in rows:
            # Convert S3 paths to presigned URLs
            row_dict = self._convert_paths_to_presigned_urls(row_dict)
            # Create data instance
            result.append(self.data_model.from_row(row_dict))

        return result

    def _convert_paths_to_presigned_urls(self, row: dict) -> dict:
        for key, value in row.items():
            if (
                isinstance(value, str)
                and key.startswith("file_path")
                and value.startswith("s3://")
            ):
                try:
                    bucket, key_path = self._parse_s3_path(value)
                    presigned_url = self.boto3_client.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": bucket, "Key": key_path},
                        ExpiresIn=self.presign_expiry,
                    )
                    row[key] = presigned_url
                except Exception as e:
                    logger.warning(f"Failed to generate presigned URL for {value}: {e}")
                    # Keep original path if presigning fails
        return row

    @staticmethod
    def _parse_s3_path(s3_uri: str) -> tuple[str, str]:
        s3_uri = s3_uri.replace("s3://", "")
        parts = s3_uri.split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""
