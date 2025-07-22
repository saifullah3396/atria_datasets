from typing import Any

from atria_core.types import ClassificationGT, GroundTruth, Image, ImageInstance, Label

from atria_datasets import DATASET, AtriaHuggingfaceImageDataset


@DATASET.register("huggingface_cifar10")
class HuggingfaceCifar10(AtriaHuggingfaceImageDataset):
    _REGISTRY_CONFIGS = {
        "plain_text": {"hf_repo": "uoft-cs/cifar10", "hf_config_name": "plain_text"}
    }

    def _input_transform(self, sample: dict[str, Any]) -> ImageInstance:
        return ImageInstance(
            image=Image(content=sample["img"]),
            gt=GroundTruth(
                classification=ClassificationGT(
                    label=Label(
                        value=sample["label"],
                        name=self.metadata.dataset_labels.classification[
                            sample["label"]
                        ],
                    )
                )
            ),
        )
