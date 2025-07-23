from typing import Any

from atria_core.types import ClassificationGT, GroundTruth, Image, ImageInstance, Label

from atria_datasets import DATASET, AtriaHuggingfaceImageDataset


@DATASET.register("huggingface_cifar10")
class HuggingfaceCifar10(AtriaHuggingfaceImageDataset):
    _REGISTRY_CONFIGS = {
        "plain_text": {"hf_repo": "uoft-cs/cifar10", "hf_config_name": "plain_text"},
        "plain_text_1k": {
            "hf_repo": "uoft-cs/cifar10",
            "hf_config_name": "plain_text",
            "max_train_samples": 1000,
            "max_test_samples": 1000,
            "max_validation_samples": 1000,
        },
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
