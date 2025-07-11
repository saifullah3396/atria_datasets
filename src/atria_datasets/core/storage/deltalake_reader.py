from collections.abc import Sequence

import pandas as pd

from atria_datasets.core.typing.common import T_BaseDataInstance


class DeltalakeReader(Sequence[T_BaseDataInstance]):
    def __init__(
        self,
        path: str,
        data_model: T_BaseDataInstance,
        allowed_keys: set[str] | None = None,
    ):
        import deltalake

        self.delta_table_path = path
        self.data_model = data_model
        self.allowed_keys = allowed_keys
        self.df = deltalake.DeltaTable(self.delta_table_path).to_pandas()
        if self.allowed_keys is not None:
            self.allowed_keys = {
                col
                for col in self.df.columns
                if col.startswith(tuple(self.allowed_keys))
            }

    def dataframe(self) -> pd.DataFrame:
        """
        Displays the dataset split information in a rich format.
        """
        return self.df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> T_BaseDataInstance:  # type: ignore[override]
        if idx < 0 or idx >= len(self.df):
            raise IndexError(
                f"Index {idx} out of bounds for dataset with {len(self.df)} rows."
            )
        if self.allowed_keys is not None:
            row = self.df.iloc[idx][self.allowed_keys]
        else:
            row = self.df.iloc[idx]
        return self.data_model.from_row(row)
