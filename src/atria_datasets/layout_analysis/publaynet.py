from typing import Any

from atria_core.types import (
    AnnotatedObject,
    BoundingBox,
    BoundingBoxMode,
    DatasetLabels,
    DatasetMetadata,
    DocumentInstance,
    GroundTruth,
    Image,
    Label,
    LayoutAnalysisGT,
)

from atria_datasets import DATASET, AtriaHuggingfaceDocumentDataset

_CLASSES = ["text", "title", "list", "table", "figure"]


@DATASET.register("publaynet")
class PubLayNet(AtriaHuggingfaceDocumentDataset):
    _REGISTRY_CONFIGS = {
        "default": {"hf_repo": "jordanparker6/publaynet", "hf_config_name": "default"}
    }

    def _metadata(self) -> DatasetMetadata:
        metadata = super()._metadata()
        metadata.dataset_labels = DatasetLabels(layout=_CLASSES)
        return metadata

    def _input_transform(self, sample: dict[str, Any]) -> DocumentInstance:
        annotated_objects = []
        image = Image(content=sample["image"])
        for ann in sample["annotations"]:
            if ann.get("ignore", False):
                continue

            bbox = BoundingBox(value=ann["bbox"], mode=BoundingBoxMode.XYWH)
            if not bbox.is_valid:
                continue

            if ann["area"] <= 0 or bbox.width < 1 or bbox.height < 1:
                continue

            category_idx = ann["category_id"] - 1
            if category_idx < 0 or category_idx > len(_CLASSES):
                continue

            annotated_objects.append(
                AnnotatedObject(
                    label=Label(value=category_idx, name=_CLASSES[category_idx]),
                    bbox=BoundingBox(
                        value=ann["bbox"], mode=BoundingBoxMode.XYWH
                    ).switch_mode(),
                    segmentation=ann["segmentation"],
                    iscrowd=ann["iscrowd"],
                )
            )
        return DocumentInstance(
            sample_id=sample["image_id"],
            image=image,
            gt=GroundTruth(
                layout=LayoutAnalysisGT(annotated_objects=annotated_objects)
            ),
        )
