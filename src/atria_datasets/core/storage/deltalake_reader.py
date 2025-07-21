from collections.abc import Sequence

import pandas as pd
import pyarrow.dataset as ds
from deltalake import DeltaTable

from atria_datasets.core.typing.common import T_BaseDataInstance


class DeltalakeReader(Sequence[T_BaseDataInstance]):
    def __init__(
        self,
        path: str,
        data_model: T_BaseDataInstance,
        allowed_keys: set[str] | None = None,
        storage_options: dict | None = None,
    ):
        self._path = path
        self._storage_options = storage_options
        self._data_model = data_model
        self._allowed_keys = allowed_keys
        if self._allowed_keys is not None:
            self._allowed_keys = [
                col
                for col in self.df.columns
                if col.startswith(tuple(self._allowed_keys))
            ]
        self._pa_table = ds.dataset(  # we get the latest version of the Delta table and load it as a PyArrow dataset
            DeltaTable(self._path, storage_options=self._storage_options).file_uris()
        )
        self.length = self._pa_table.count_rows()

    def dataframe(self) -> pd.DataFrame:
        return self._pa_table.to_pandas()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> T_BaseDataInstance:  # type: ignore
        if isinstance(index, list):
            return self.__getitems__(index)
        rows = self._pa_table.take([index]).to_pylist()  # type: ignore
        return self._process_rows(rows)[0]

    def __getitems__(self, indices: list[int]) -> list[T_BaseDataInstance]:
        rows = self._pa_table.take(indices).to_pylist()  # type: ignore
        return self._process_rows(rows)

    def _process_rows(self, rows: list[dict]) -> list[T_BaseDataInstance]:
        processed_rows = []
        for row in rows:
            if self._allowed_keys is not None:
                row = {k: v for k, v in row.items() if k in self._allowed_keys}
            instance = self._data_model.from_row(row)
            processed_rows.append(instance)
        return processed_rows
