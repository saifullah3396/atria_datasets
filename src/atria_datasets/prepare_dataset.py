import fire
from atria_core.logger import get_logger

from atria_datasets.core.dataset.atria_dataset import DatasetLoadingMode

logger = get_logger(__name__)


def main(
    name: str,
    data_dir: str | None = None,
    access_token: str | None = None,
    overwrite_existing_cached: bool = False,
    overwrite_existing_shards: bool = False,
    num_processes: int = 8,
    upload_to_hub: bool = False,
    overwrite_in_hub: bool = True,
):
    from atria_datasets import AtriaDataset

    dataset = AtriaDataset.load_from_registry(
        name=name,
        data_dir=data_dir,
        overwrite_existing_cached=overwrite_existing_cached,
        overwrite_existing_shards=overwrite_existing_shards,
        access_token=access_token,
        num_processes=num_processes,
        dataset_load_mode=DatasetLoadingMode.in_memory,
        # build_kwargs={
        #     "max_train_samples": 10,
        #     "max_validation_samples": 10,
        #     "max_test_samples": 10,
        # },
    )
    logger.info(f"Loaded dataset:\n{dataset}")
    for sample in dataset.train:
        sample.load()
        break

    if upload_to_hub:
        dataset.upload_to_hub(overwrite_existing=overwrite_in_hub)


if __name__ == "__main__":
    fire.Fire(main)
