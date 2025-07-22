import fire
import torch
from atria_core.logger import get_logger

logger = get_logger(__name__)

torch.set_printoptions(profile="short", threshold=10)


def main(
    name,
    access_token: str | None = None,
    overwrite_existing_cached: bool = False,
    overwrite_existing_shards: bool = False,
):
    from atria_datasets import AtriaDataset

    logger.info(f"Loading dataset: {name}")
    dataset = AtriaDataset.load_from_registry(
        name=name,
        overwrite_existing_cached=overwrite_existing_cached,
        overwrite_existing_shards=overwrite_existing_shards,
        access_token=access_token,
    )
    logger.info(f"Loaded dataset: {dataset}")
    for sample in dataset.train:
        logger.info(sample)
        break

    if "/" in name:
        name = name.split("/")[0]
    # dataset.upload_to_hub(name=name)


if __name__ == "__main__":
    fire.Fire(main)
