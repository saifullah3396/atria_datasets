import os
from pathlib import Path

from atria_core.logger.logger import get_logger
from atria_core.types import (
    AtriaDatasetConfig,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    GroundTruth,
    Image,
    LayoutAnalysisGT,
    SplitConfig,
)

from atria_datasets import DATASET, AtriaDocumentDataset

from .utilities import read_pascal_voc, read_words_json

logger = get_logger(__name__)

_CITATION = """\
@software{smock2021tabletransformer,
    author = {Smock, Brandon and Pesala, Rohith},
    month = {06},
    title = {{Table Transformer}},
    url = {https://github.com/microsoft/table-transformer},
    version = {1.0.0},
    year = {2021}
}
"""


_DESCRIPTION = """\
ICDAR-2013: Towards comprehensive table extraction from unstructured documents.
"""


_HOMEPAGE = "https://github.com/microsoft/table-transformer/"


_LICENSE = "https://github.com/microsoft/table-transformer/blob/main/LICENSE"

_URLS = [
    "https://huggingface.co/datasets/bsmock/FinTabNet.c/resolve/main/FinTabNet.c-PDF_Annotations.tar.gz",
    "https://huggingface.co/datasets/bsmock/FinTabNet.c/resolve/main/FinTabNet.c-Structure.tar.gz",
]
_CLASSES = [
    "table",
    "table column",
    "table row",
    "table column header",
    "table projected row header",
    "table spanning cell",
]


@DATASET.register("fintabnet")
class FinTabNet(AtriaDocumentDataset):
    _REGISTRY_CONFIGS = {"structure": AtriaDatasetConfig(data_urls=_URLS)}

    def _data_model(self) -> DocumentInstance:
        return DocumentInstance

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            dataset_labels=DatasetLabels(layout=_CLASSES),
        )

    def _split_configs(self, data_dir: str):
        base_path = Path(data_dir) / "FinTabNet.c-Structure" / "FinTabNet.c-Structure"

        splits = []
        for split in [
            DatasetSplitType.test,
            DatasetSplitType.validation,
            DatasetSplitType.train,
        ]:
            if split == DatasetSplitType.test:
                split_path = "test"
            elif split == DatasetSplitType.validation:
                split_path = "val"
            elif split == DatasetSplitType.train:
                split_path = "train"
            splits.append(
                SplitConfig(
                    split=split,
                    gen_kwargs={
                        "xmls_dir": base_path / split_path,
                        "images_dir": base_path / "images",
                        "words_dir": base_path / "words",
                    },
                )
            )
        return splits

    def _split_iterator(
        self, split: DatasetSplitType, xmls_dir: Path, images_dir: Path, words_dir: Path
    ):
        xml_filenames = [elem for elem in os.listdir(xmls_dir) if elem.endswith(".xml")]
        for filename in xml_filenames:
            xml_filepath = xmls_dir / filename
            image_file_path = images_dir / filename.replace(".xml", ".jpg")
            word_file_path = words_dir / filename.replace(".xml", "_words.json")
            annotated_objects = read_pascal_voc(xml_filepath, labels=_CLASSES)
            words, word_bboxes = read_words_json(word_file_path)
            yield DocumentInstance(
                sample_id=Path(image_file_path).name,
                image=Image(file_path=image_file_path),
                gt=GroundTruth(
                    layout=LayoutAnalysisGT(
                        annotated_objects=annotated_objects,
                        words=words,
                        word_bboxes=word_bboxes,
                    )
                ),
            )
