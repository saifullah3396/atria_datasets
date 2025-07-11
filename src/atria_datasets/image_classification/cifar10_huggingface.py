from typing import Any

from atria_core.types import (
    AtriaHuggingfaceDatasetConfig,
    GroundTruth,
    Image,
    ImageInstance,
    Label,
)

from atria_datasets import DATASET, AtriaHuggingfaceDataset


@DATASET.register("huggingface_cifar10")
class HuggingfaceCifar10(AtriaHuggingfaceDataset[ImageInstance]):
    _REGISTRY_CONFIGS = [
        AtriaHuggingfaceDatasetConfig(
            config_name="plain_text", hf_repo="uoft-cs/cifar10"
        )
    ]
    _DATA_MODEL = ImageInstance

    def _input_transform(self, sample: dict[str, Any]) -> ImageInstance:
        label = Label(
            value=sample["label"],
            name=self.metadata.dataset_labels.classification[sample["label"]],
        )
        return ImageInstance(
            image=Image(content=sample["img"]), gt=GroundTruth(classification=label)
        )
