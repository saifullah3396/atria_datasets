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
The ICDAR 2019 Table Detection Challenge (cTDaR) is a competition that focuses on the detection of tables in document images. The challenge is part of the International Conference on Document Analysis and Recognition (ICDAR) 2019, which is a prestigious event in the field of document analysis and recognition. The cTDaR challenge aims to advance the state-of-the-art in table detection and provide a benchmark for evaluating the performance of different algorithms and approaches.
"""

_HOMEPAGE = "https://github.com/cndplab-founder/ICDAR2019_cTDaR/tree/master/samples"

_LICENSE = "Apache-2.0 license"

_CLASSES = ["table"]

_URLS = [
    # Add actual download URLs here when available
]


@DATASET.register("icdar2019")
class Icdar2019(AtriaDocumentDataset):
    _REGISTRY_CONFIGS = {
        "trackA_modern": {"type": "modern"},
        "trackA_archival": {"type": "archival"},
    }

    def __init__(
        self,
        max_train_samples: int | None = None,
        max_validation_samples: int | None = None,
        max_test_samples: int | None = None,
        type: str = "modern",
    ):
        super().__init__(
            max_train_samples=max_train_samples,
            max_validation_samples=max_validation_samples,
            max_test_samples=max_test_samples,
        )
        self.type = type

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
        class SplitIterator:
            def __init__(
                self, split: DatasetSplitType, data_dir: str, type: str = "modern"
            ):
                self.split = split
                self.data_dir = Path(data_dir)

                # Construct config name from task and type
                self.config_name = f"trackA_{type}"

                if not (self.data_dir / self.config_name / split.value).exists():
                    raise FileNotFoundError(
                        f"Split {split.value} does not exist in {self.data_dir / self.config_name}"
                    )

                self.image_dir = self.data_dir / self.config_name / split.value
                self.ann_file = self.data_dir / self.config_name / f"{split.value}.json"

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
                                value=ann["category_id"],
                                name=_CLASSES[ann["category_id"]],
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

        return SplitIterator(split=split, data_dir=data_dir, type=self.type)
