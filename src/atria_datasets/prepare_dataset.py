import fire
import torch
from atria_core.logger import get_logger

logger = get_logger(__name__)

torch.set_printoptions(profile="short", threshold=10)


def main(
    name,
    max_samples: int | None = None,
    access_token: str | None = None,
    overwrite_existing_cached: bool = False,
    overwrite_existing_shards: bool = False,
):
    from atria_datasets import AtriaDataset

    if max_samples is not None:
        logger.info(f"Loading dataset: {name} with max_samples={max_samples}")
    else:
        logger.info(f"Loading dataset: {name}")
    dataset = AtriaDataset.load_from_registry(
        name=name,
        split=None,
        overwrite_existing_cached=overwrite_existing_cached,
        overwrite_existing_shards=overwrite_existing_shards,
        access_token=access_token,
        build_kwargs={
            "max_train_samples": max_samples,
            "max_validation_samples": max_samples,
            "max_test_samples": max_samples,
        },
    )
    logger.info(f"Loaded dataset: {dataset}")
    for sample in dataset.train:
        logger.info(sample)
        break
    # dataset.upload_to_hub()


if __name__ == "__main__":
    fire.Fire(main)
