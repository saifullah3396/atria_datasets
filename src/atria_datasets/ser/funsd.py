import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

from atria_core.logger.logger import get_logger
from atria_core.types import (
    OCR,
    SERGT,
    AtriaDatasetConfig,
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
from atria_core.types.factory import BoundingBoxList

from atria_datasets import DATASET
from atria_datasets.core.dataset.atria_dataset import AtriaDocumentDataset

from .utilities import (
    _get_line_bboxes,
    _normalize_bbox,
    _sorted_indices_in_reading_order,
)

logger = get_logger(__name__)

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """FUNSD Dataset"""

_HOMEPAGE = "https://guillaumejaume.github.io/FUNSD/"
_DATA_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"

_LICENSE = "Apache-2.0 license"

_CLASSES = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]


class FUNSDConfig(AtriaDatasetConfig):
    apply_reading_order_correction: bool = True


@DATASET.register("funsd")
class FUNSD(AtriaDocumentDataset):
    _REGISTRY_CONFIGS = {"default": FUNSDConfig(data_urls=_DATA_URL)}

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
                gen_kwargs={"split_dir": data_dir / "dataset/dataset/training_data"},
            ),
            SplitConfig(
                split=DatasetSplitType.test,
                gen_kwargs={"split_dir": data_dir / "dataset/dataset/testing_data"},
            ),
        ]

    def _load_ground_truth(
        self, annotation_path: str, image_size: tuple[int, int]
    ) -> OCR:
        words = []
        word_bboxes = []
        word_segment_level_bboxes = []
        word_labels = []
        with open(annotation_path, encoding="utf8") as f:
            annotation = json.load(f)
        for item in annotation["form"]:
            cur_line_bboxes = []
            words_item, label_item = item["words"], item["label"]
            words_item = [w for w in words_item if w["text"].strip() != ""]
            if len(words_item) == 0:
                continue

            if label_item == "other":
                for w in words_item:
                    words.append(w["text"])
                    word_labels.append("O")
                    cur_line_bboxes.append(_normalize_bbox(w["box"], image_size))
            else:
                words.append(words_item[0]["text"])
                word_labels.append("B-" + label_item.upper())
                cur_line_bboxes.append(
                    _normalize_bbox(words_item[0]["box"], image_size)
                )
                for w in words_item[1:]:
                    words.append(w["text"])
                    word_labels.append("I-" + label_item.upper())
                    cur_line_bboxes.append(_normalize_bbox(w["box"], image_size))

            # add per word box
            word_bboxes.extend(cur_line_bboxes)

            # add segment level box
            cur_line_bboxes = _get_line_bboxes(cur_line_bboxes)
            word_segment_level_bboxes.extend(cur_line_bboxes)

        # sort the word reading order
        if self.apply_reading_order_correction:
            sorted_indces = _sorted_indices_in_reading_order(word_bboxes)
            words = [words[i] for i in sorted_indces]
            word_bboxes = [word_bboxes[i] for i in sorted_indces]
            word_labels = [word_labels[i] for i in sorted_indces]
            word_segment_level_bboxes = [
                word_segment_level_bboxes[i] for i in sorted_indces
            ]

        return GroundTruth(
            ser=SERGT(
                words=words,
                word_bboxes=BoundingBoxList(value=word_bboxes),
                word_labels=LabelList.from_list(
                    [
                        Label(value=_CLASSES.index(word_label), name=word_label)
                        for word_label in word_labels
                    ]
                ),
                segment_level_bboxes=BoundingBoxList(value=word_segment_level_bboxes),
            )
        )

    def _split_iterator(
        self, split: DatasetSplitType, split_dir: str
    ) -> Generator[tuple[str, dict[str, Any]], None, None]:
        ann_dir = split_dir / "annotations"
        image_dir = split_dir / "images"

        for _, filename in enumerate(sorted(os.listdir(image_dir))):
            image = Image(file_path=image_dir / Path(filename).name)
            ground_truth = self._load_ground_truth(
                annotation_path=ann_dir / Path(filename).with_suffix(".json"),
                image_size=(image.source_width, image.source_height),
            )
            yield DocumentInstance(
                sample_id=Path(filename).name, image=image, gt=ground_truth
            )
