import json
from pathlib import Path

from atria_core.types import (
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

from .utilities import _normalize_bbox, _sorted_indices_in_reading_order

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """WildReceipts Dataset"""

_HOMEPAGE = ""

_LICENSE = "Apache-2.0 license"

_DATA_URLS = "https://download.openmmlab.com/mmocr/data/wildreceipt.tar"

_CLASSES = [
    "B-Store_name_value",
    "B-Store_name_key",
    "B-Store_addr_value",
    "B-Store_addr_key",
    "B-Tel_value",
    "B-Tel_key",
    "B-Date_value",
    "B-Date_key",
    "B-Time_value",
    "B-Time_key",
    "B-Prod_item_value",
    "B-Prod_item_key",
    "B-Prod_quantity_value",
    "B-Prod_quantity_key",
    "B-Prod_price_value",
    "B-Prod_price_key",
    "B-Subtotal_value",
    "B-Subtotal_key",
    "B-Tax_value",
    "B-Tax_key",
    "B-Tips_value",
    "B-Tips_key",
    "B-Total_value",
    "B-Total_key",
    "O",
]


class WildReceiptsConfig(AtriaDatasetConfig):
    apply_reading_order_correction: bool = True


@DATASET.register("wild_receipts")
class WildReceipts(AtriaDocumentDataset):
    _REGISTRY_CONFIGS = {"default": WildReceiptsConfig(data_urls=_DATA_URLS)}

    def _data_model(self) -> DocumentInstance:
        return DocumentInstance

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            citation=_CITATION,
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            dataset_labels=DatasetLabels(ser=_CLASSES),
        )

    def _split_configs(self, data_dir: str):
        data_dir = Path(data_dir)
        return [
            SplitConfig(
                split=DatasetSplitType.train,
                gen_kwargs={
                    "split_file_path": data_dir
                    / "wildreceipt/wildreceipt"
                    / "train.txt"
                },
            ),
            SplitConfig(
                split=DatasetSplitType.test,
                gen_kwargs={
                    "split_file_path": data_dir / "wildreceipt/wildreceipt" / "test.txt"
                },
            ),
        ]

    def _read_label_map(self, filepath: str) -> dict[int, str]:
        label_map = {}
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx, label = line.split(maxsplit=1)
                label_map[int(idx)] = label.strip()
        return label_map

    def _load_ground_truth(
        self, split_file_path: str, sample_info: str, image_size: tuple
    ):
        words = []
        labels = []
        bboxes = []
        id2labels = self._read_label_map(split_file_path.parent / "class_list.txt")
        for i in sample_info["annotations"]:
            label = id2labels[i["label"]]
            if (
                label == "Ignore" or i["text"] == ""
            ):  # label 0 is attached to ignore so we skip it
                continue
            if label in ["Others"]:
                label = "O"
            else:
                label = "B-" + label
            labels.append(label)
            words.append(i["text"])
            bboxes.append(
                _normalize_bbox(
                    [i["box"][6], i["box"][7], i["box"][2], i["box"][3]], image_size
                )
            )

        # sort the word reading order
        if self.apply_reading_order_correction:
            sorted_indces = _sorted_indices_in_reading_order(bboxes)
            words = [words[i] for i in sorted_indces]
            word_labels = [labels[i] for i in sorted_indces]
            word_bboxes = [bboxes[i] for i in sorted_indces]

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

    def _split_iterator(self, split: DatasetSplitType, split_file_path: Path):
        item_list = []
        with open(split_file_path) as f:
            for line in f:
                item_list.append(line.rstrip("\n\r"))
        for idx, filename in enumerate(item_list):
            sample_info = json.loads(filename)
            image = Image(file_path=split_file_path.parent / sample_info["file_name"])
            ground_truth = self._load_ground_truth(
                split_file_path=split_file_path,
                sample_info=sample_info,
                image_size=image.size,
            )
            yield DocumentInstance(
                sample_id=Path(filename).name, image=image, gt=ground_truth
            )
