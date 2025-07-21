from typing import Any

from atria_core.types import (
    AtriaHuggingfaceDatasetConfig,
    ClassificationGT,
    GroundTruth,
    Image,
    ImageInstance,
    Label,
)

from atria_datasets import DATASET, AtriaHuggingfaceImageDataset


@DATASET.register("mnist")
class MNIST(AtriaHuggingfaceImageDataset):
    _REGISTRY_CONFIGS = {"mnist": AtriaHuggingfaceDatasetConfig(hf_repo="ylecun/mnist")}

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
