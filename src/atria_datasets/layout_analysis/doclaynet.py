from typing import Any

from atria_core.types import (
    AnnotatedObject,
    AtriaHuggingfaceDatasetConfig,
    BoundingBox,
    BoundingBoxMode,
    ClassificationGT,
    DatasetLabels,
    DatasetMetadata,
    DocumentInstance,
    GroundTruth,
    Image,
    Label,
    LayoutAnalysisGT,
)

from atria_datasets import DATASET
from atria_datasets.core.dataset.atria_huggingface_dataset import (
    AtriaHuggingfaceDocumentDataset,
)

_DOC_CLASSES = [
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
]

_LAYOUT_CLASSES = [
    "Caption ",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title ",
]


@DATASET.register("doclaynet")
class DocLayNet(AtriaHuggingfaceDocumentDataset):
    _REGISTRY_CONFIGS = {
        "2022.08": AtriaHuggingfaceDatasetConfig(hf_repo="ds4sd/DocLayNet")
    }

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            dataset_labels=DatasetLabels(
                classification=_DOC_CLASSES, layout=_LAYOUT_CLASSES
            ),
            description="DocLayNet is a large-scale dataset for document layout analysis.",
        )

    def _data_model(self):
        return DocumentInstance

    def _data_model_transform(self, sample: dict[str, Any]) -> DocumentInstance:
        annotated_objects = []
        image = Image(content=sample["image"])
        for ann in sample["objects"]:
            if ann.get("ignore", False):
                continue

            bbox = BoundingBox(value=ann["bbox"], mode=BoundingBoxMode.XYWH)
            if not bbox.is_valid:
                continue

            if ann["area"] <= 0 or bbox.width < 1 or bbox.height < 1:
                continue

            category_idx = ann["category_id"] - 1
            if category_idx < 0 or category_idx > len(_LAYOUT_CLASSES):
                continue

            annotated_objects.append(
                AnnotatedObject(
                    label=Label(value=category_idx, name=_LAYOUT_CLASSES[category_idx]),
                    bbox=bbox.switch_mode(),
                    segmentation=ann["segmentation"],
                    iscrowd=ann["iscrowd"],
                )
            )
        return DocumentInstance(
            sample_id=sample["image_id"],
            image=image,
            gt=GroundTruth(
                classification=ClassificationGT(
                    label=Label(
                        value=_DOC_CLASSES.index(sample["doc_category"]),
                        name=sample["doc_category"],
                    )
                ),
                layout=LayoutAnalysisGT(annotated_objects=annotated_objects),
            ),
        )
