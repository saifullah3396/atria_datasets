from typing import Any

from atria_core.types import (
    AtriaHuggingfaceDatasetConfig,
    GroundTruth,
    Image,
    ImageInstance,
    Label,
)

from atria_datasets import DATASET, AtriaHuggingfaceImageDataset


@DATASET.register("huggingface_cifar10")
class HuggingfaceCifar10(AtriaHuggingfaceImageDataset):
    _REGISTRY_CONFIGS = {
        "plain_text": AtriaHuggingfaceDatasetConfig(hf_repo="uoft-cs/cifar10")
    }

    def _input_transform(self, sample: dict[str, Any]) -> ImageInstance:
        label = Label(
            value=sample["label"],
            name=self.metadata.dataset_labels.classification[sample["label"]],
        )
        return ImageInstance(
            image=Image(content=sample["img"]), gt=GroundTruth(classification=label)
        )
