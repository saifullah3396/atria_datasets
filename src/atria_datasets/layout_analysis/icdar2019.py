from pathlib import Path

from atria_core.types import (
    AnnotatedObject,
    AtriaDatasetConfig,
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
    SplitConfig,
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


@DATASET.register("icdar2019")
class Icdar2019(AtriaDocumentDataset):
    _REGISTRY_CONFIGS = {
        "trackA_modern": AtriaDatasetConfig(
            # this dataset is obtained following the preprocessing steps in
            # https://github.com/microsoft/unilm/tree/master/dit/object_detection
        ),
        "trackA_archival": AtriaDatasetConfig(
            # this dataset is obtained following the preprocessing steps in
            # https://github.com/microsoft/unilm/tree/master/dit/object_detection
        ),
    }

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            dataset_labels=DatasetLabels(layout=_CLASSES),
        )

    def _data_model(self):
        return DocumentInstance

    def _split_configs(self, data_dir: str) -> list[SplitConfig]:
        data_dir = Path(data_dir)

        split_configs = []
        for split in [DatasetSplitType.train, DatasetSplitType.test]:
            if not (data_dir / self.config_name / split).exists():
                raise FileNotFoundError(
                    f"Split {split.value} does not exist in {data_dir / self.config_name}"
                )
            split_configs.append(
                SplitConfig(
                    split=split,
                    gen_kwargs={
                        "image_dir": data_dir / self.config_name / split.value,
                        "ann_file": data_dir / self.config_name / f"{split.value}.json",
                    },
                )
            )
        return split_configs

    def _split_iterator(self, split: DatasetSplitType, image_dir: str, ann_file: str):
        samples_list, category_names = _load_coco_json(
            json_file=str(ann_file), image_root=image_dir
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
                    iscrowd=ann["iscrowd"],
                )
                for ann in sample["annotations"]
            ]
            yield DocumentInstance(
                sample_id=Path(sample["file_name"]).name,
                image=Image(file_path=sample["file_name"]),
                gt=GroundTruth(
                    layout=LayoutAnalysisGT(annotated_objects=annotated_objects)
                ),
            )
