from typing import Any

import pandas as pd
from deltalake import DeltaTable


class InferenceAgent:
    def __init__(self, storage_options: dict[str, str]):
        self.storage_options = storage_options
        self.current_dataset_path = None
        self.delta_table = None
        self.df = None

    def _build_dataset_path(self, repo_name: str, branch_name: str) -> str:
        return f"s3://{repo_name}/{branch_name}/delta/train"

    def _needs_dataset_update(self, repo_name: str, branch_name: str) -> bool:
        new_path = self._build_dataset_path(repo_name, branch_name)
        return self.current_dataset_path != new_path

    def _load_dataset(self, repo_name: str, branch_name: str):
        path = self._build_dataset_path(repo_name, branch_name)
        self.delta_table = DeltaTable(path, storage_options=self.storage_options)
        self.df = self.delta_table.to_pandas(columns=["index", "sample_id"])
        self.current_dataset_path = path
        print(f"Dataset loaded from: {path}")

    def _apply_inference(self, subset: pd.DataFrame) -> list:
        # Dummy predictions - replace with your actual inference logic
        return list(range(len(subset)))

    def run(self, request: dict[str, Any]) -> pd.DataFrame:
        repo_name = request.get("repo_name")
        branch_name = request.get("branch_name")
        sample_size = request.get("sample_size", 10)

        if not repo_name or not branch_name:
            raise ValueError("Request must contain repo_name and branch_name")

        # Check if dataset needs to be updated
        if self._needs_dataset_update(repo_name, branch_name):
            self._load_dataset(repo_name, branch_name)

        # Sample data
        sample_indices = self.df.sample(n=sample_size).index.tolist()
        subset = self.df.loc[sample_indices]

        # Apply inference
        preds = self._apply_inference(subset)

        # Prepare results
        results = subset[["index", "sample_id"]].copy()
        results["prediction"] = preds

        return results


# Usage example:
storage_options = {
    "AWS_ACCESS_KEY_ID": "AKIAJ2NNAE7KY6KJDFVQ",
    "AWS_SECRET_ACCESS_KEY": "OjJSp5cwWOH92Vem2h2yRo/KAOvqHedzI+Uovfiv",
    "AWS_ENDPOINT": "http://127.0.0.1:8090",
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

agent = InferenceAgent(storage_options)

request = {
    "repo_name": "testuser12345-cifar10",
    "branch_name": "main",
    "sample_size": 10,
}

results = agent.run(request)
print(results)
