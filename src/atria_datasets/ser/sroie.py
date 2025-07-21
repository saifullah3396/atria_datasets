import json
import os
from pathlib import Path

from atria_core.types import (
    OCR,
    SERGT,
    AtriaDatasetConfig,
    BoundingBoxList,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    GroundTruth,
    Image,
    Label,
    LabelList,
    SplitConfig,
)

from atria_datasets import DATASET
from atria_datasets.core.dataset.atria_dataset import AtriaDocumentDataset

from .utilities import _normalize_bbox

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """SROIE Receipts Dataset"""

_HOMEPAGE = "https://rrc.cvc.uab.es/?ch=13"

_LICENSE = "Apache-2.0 license"

_CLASSES = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-DATE",
    "I-DATE",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-TOTAL",
    "I-TOTAL",
]

_DATA_URLS = {
    "sroie": (
        "https://drive.google.com/file/d/1ZyxAw1d-9UvhgNLGRvsJK4gBCMf0VpGD/view?usp=sharing",
        ".zip",
    )
}


class SROIEConfig(AtriaDatasetConfig):
    pass


@DATASET.register("sroie")
class SROIE(AtriaDocumentDataset):
    _REGISTRY_CONFIGS = {"default": SROIEConfig(data_urls=_DATA_URLS)}

    def _data_model(self) -> DocumentInstance:
        return DocumentInstance

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            citation=_CITATION,
            homepage=_HOMEPAGE,
            description=_DESCRIPTION,
            license=_LICENSE,
            dataset_labels=DatasetLabels(ser=_CLASSES),
        )

    def _split_configs(self, data_dir: str):
        data_dir = Path(data_dir)
        return [
            SplitConfig(
                split=DatasetSplitType.train,
                gen_kwargs={"split_dir": data_dir / "sroie/sroie/train"},
            ),
            SplitConfig(
                split=DatasetSplitType.test,
                gen_kwargs={"split_dir": data_dir / "sroie/sroie/test"},
            ),
        ]

    def _load_ground_truth(
        self, annotation_path: str, image_size: tuple[int, int]
    ) -> OCR:
        with open(annotation_path, encoding="utf8") as f:
            sample = json.load(f)
        words = []
        word_bboxes = []
        word_labels = []
        for word, box, label in zip(
            sample["words"], sample["bbox"], sample["labels"], strict=True
        ):
            word = word.strip()
            if len(word) == 0:
                continue
            words.append(word)
            word_bboxes.append(_normalize_bbox(box, image_size))
            word_labels.append(label)
        return GroundTruth(
            ser=SERGT(
                words=words,
                word_bboxes=BoundingBoxList(value=word_bboxes),
                word_labels=LabelList.from_list(
                    [
                        Label(value=_CLASSES.index(label), name=label)
                        for label in word_labels
                    ]
                ),
            )
        )

    def _split_iterator(self, split: DatasetSplitType, split_dir: Path):
        ann_dir = split_dir / "tagged"
        image_dir = split_dir / "images"
        for _, filename in enumerate(sorted(os.listdir(image_dir))):
            image = Image(file_path=image_dir / Path(filename).name)
            ground_truth = self._load_ground_truth(
                annotation_path=ann_dir / Path(filename).with_suffix(".json"),
                image_size=image.size,
            )
            yield DocumentInstance(
                sample_id=Path(filename).name, image=image, gt=ground_truth
            )
