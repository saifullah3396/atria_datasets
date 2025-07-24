from collections.abc import Generator
from pathlib import Path

from atria_core.types import (
    AnnotatedObject,
    AnnotatedObjectList,
    BoundingBox,
    BoundingBoxMode,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    GroundTruth,
    Image,
    Label,
    LayoutAnalysisGT,
)

from atria_datasets import DATASET, AtriaDocumentDataset
from atria_datasets.core.dataset.atria_dataset import AtriaDatasetConfig

from .utilities import _load_coco_json

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{icdar2019,
    title={ICDAR2019 Competition on Table Detection and Recognition},
    author={Zhang, Yifan and Wang, Yuhui and Zhang, Yifan and Liu, Zhen and Li, Yujie and Wang, Xiaoyang},
    booktitle={2019 International Conference on Document Analysis and Recognition (ICDAR)},
    pages={1--6},
    year={2019},
    organization={IEEE}
}
"""

# You can copy an official description
_DESCRIPTION = """\
The ICDAR 2019 Table Detection Challenge (cTDaR) is a competition that focuses on the detection of tables in document images.
"""

_HOMEPAGE = "https://github.com/cndplab-founder/ICDAR2019_cTDaR/tree/master/samples"

_LICENSE = "Apache-2.0 license"

_CLASSES = ["table"]

_URLS = [
    # Add actual download URLs here when available
]


class SplitIterator:
    def __init__(
        self, split: DatasetSplitType, data_dir: str, config: AtriaDatasetConfig
    ):
        self.split = split
        self.data_dir = Path(data_dir)
        self.config = config

        if not (self.data_dir / self.config.config_name / split.value).exists():
            raise FileNotFoundError(
                f"Split {split.value} does not exist in {self.data_dir / self.config.config_name}"
            )

        self.image_dir = self.data_dir / self.config.config_name / split.value
        self.ann_file = self.data_dir / self.config.config_name / f"{split.value}.json"

    def __iter__(self) -> Generator[DocumentInstance, None, None]:
        samples_list, category_names = _load_coco_json(
            json_file=str(self.ann_file), image_root=str(self.image_dir)
        )
        assert category_names == _CLASSES, (
            "Category names do not match between dataset config and json file."
            f"Required: {_CLASSES}, found: {category_names}"
        )
        for sample in samples_list:
            annotated_objects = [
                AnnotatedObject(
                    label=Label(
                        value=ann["category_id"], name=_CLASSES[ann["category_id"]]
                    ),
                    bbox=BoundingBox(
                        value=ann["bbox"], mode=BoundingBoxMode.XYWH
                    ).switch_mode(),
                    segmentation=ann["segmentation"],
                    iscrowd=bool(ann["iscrowd"]),
                )
                for ann in sample["annotations"]
            ]
            yield DocumentInstance(
                sample_id=Path(sample["file_name"]).name,
                image=Image(file_path=sample["file_name"]),
                gt=GroundTruth(
                    layout=LayoutAnalysisGT(
                        annotated_objects=AnnotatedObjectList.from_list(
                            annotated_objects
                        )
                    )
                ),
            )

    def __len__(self) -> int:
        samples_list, _ = _load_coco_json(
            json_file=str(self.ann_file), image_root=str(self.image_dir)
        )
        return len(samples_list)


@DATASET.register(
    "icdar2019",
    configs=[
        AtriaDatasetConfig(config_name="trackA_modern"),
        AtriaDatasetConfig(config_name="trackA_archival"),
    ],
)
class Icdar2019(AtriaDocumentDataset):
    def _download_urls(self) -> list[str]:
        return _URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            dataset_labels=DatasetLabels(layout=_CLASSES),
        )

    def _available_splits(self) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Generator[DocumentInstance, None, None]:
        return SplitIterator(split=split, data_dir=data_dir, config=self.config)
