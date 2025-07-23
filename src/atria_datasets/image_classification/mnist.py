from typing import Any

from atria_core.types import ClassificationGT, GroundTruth, Image, ImageInstance, Label

from atria_datasets import DATASET, AtriaHuggingfaceImageDataset


@DATASET.register("mnist")
class MNIST(AtriaHuggingfaceImageDataset):
    _REGISTRY_CONFIGS = {
        "mnist": {"hf_repo": "ylecun/mnist", "hf_config_name": "mnist"},
        "mnist_1k": {
            "hf_repo": "ylecun/mnist",
            "hf_config_name": "mnist",
            "max_train_samples": 1000,
            "max_test_samples": 1000,
            "max_validation_samples": 1000,
        },
    }

    def _input_transform(self, sample: dict[str, Any]) -> ImageInstance:
        return ImageInstance(
            image=Image(content=sample["image"]),
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
