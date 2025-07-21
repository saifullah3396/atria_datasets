import os
from pathlib import Path

from atria_core.logger.logger import get_logger
from atria_core.types import (
    AnnotatedObjectList,
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

from .utilities import read_pascal_voc

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
PubTables-1M: Towards comprehensive table extraction from unstructured documents.
"""


_HOMEPAGE = "https://github.com/microsoft/table-transformer/"


_LICENSE = "https://github.com/microsoft/table-transformer/blob/main/LICENSE"

_DETECTION_URLS = [
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Annotations_Test.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Annotations_Train.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Annotations_Val.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Filelists.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Images_Test.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Images_Train_Part1.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Images_Train_Part2.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Images_Val.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Page_Words.tar.gz",
    # "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-PDF_Annotations.tar.gz",
]
_DETECTION_LABELS = ["table", "table rotated"]
_STRUCTURE_URLS = [
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Annotations_Test.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Annotations_Train.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Annotations_Val.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Filelists.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Images_Test.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Images_Train.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Images_Val.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Table_Words.tar.gz",
    # "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-PDF_Annotations.tar.gz",
]
_STRUCTURE_LABELS = [
    "table",
    "table column",
    "table row",
    "table column header",
    "table projected row header",
    "table spanning cell",
]


def extract_archive(path):
    import tarfile

    root_path = path.parent
    folder_name = path.name.replace(".tar.gz", "")

    def extract_nonexisting(archive):
        for member in archive.members:
            name = member.name
            if not (root_path / folder_name / name).exists():
                archive.extract(name, path=root_path / folder_name)

    with tarfile.open(path) as archive:
        extract_nonexisting(archive)


def folder_iterator(folder):
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            yield os.path.join(subdir, file)


@DATASET.register("pubtables1m")
class PubTables1M(AtriaDocumentDataset):
    _REGISTRY_CONFIGS = {
        "detection": AtriaDatasetConfig(data_urls=_DETECTION_URLS),
        "structure": AtriaDatasetConfig(data_urls=_STRUCTURE_URLS),
    }

    def _data_model(self) -> DocumentInstance:
        return DocumentInstance

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            dataset_labels=DatasetLabels(
                layout=(
                    _STRUCTURE_LABELS
                    if self.config_name == "structure"
                    else _DETECTION_LABELS
                )
            ),
        )

    def _split_configs(self, data_dir: str):
        # extract archives
        anns_paths = []
        for split in ["Test", "Train", "Val"]:
            anns_paths += list(
                folder_iterator(
                    data_dir
                    / f"PubTables-1M-{self.config_name.upper()}_Annotations_{split}"
                )
            )
        anns_paths = {f"{split.lower()}/{k}" for k in anns_paths}

        # get all images paths
        images_paths = []
        for split in ["Test", "Train", "Val"]:
            images_paths += list(
                folder_iterator(
                    data_dir / f"PubTables-1M-{self.config_name.upper()}_Images_{split}"
                )
            )
        images_paths = {f"images/{k}" for k in anns_paths}

        # get all words paths
        words_paths = list(
            folder_iterator(
                data_dir / f"PubTables-1M-{self.config_name.upper()}_Page_Words"
            )
        )
        words_paths = {f"words/{k}" for k in words_paths}

        return [
            SplitConfig(
                name=DatasetSplitType.test,
                gen_kwargs={
                    "xml_filelist": data_dir
                    / f"{self.config_name.upper()}_Filelists"
                    / "test_filelist.txt",
                    "images_filelist": data_dir
                    / f"{self.config_name.upper()}_Filelists"
                    / "images_filelist.txt",
                    "images_paths": images_paths,
                    "anns_paths": anns_paths,
                    "words_paths": words_paths,
                },
            ),
            SplitConfig(
                name=DatasetSplitType.train,
                gen_kwargs={
                    "xml_filelist": data_dir
                    / f"{self.config_name.upper()}_Filelists"
                    / "train_filelist.txt",
                    "images_filelist": data_dir
                    / f"{self.config_name.upper()}_Filelists"
                    / "images_filelist.txt",
                    "images_paths": images_paths,
                    "anns_paths": anns_paths,
                    "words_paths": words_paths,
                },
            ),
            SplitConfig(
                name=DatasetSplitType.validation,
                gen_kwargs={
                    "xml_filelist": data_dir
                    / f"{self.config_name.upper()}_Filelists"
                    / "val_filelist.txt",
                    "images_filelist": data_dir
                    / f"{self.config_name.upper()}_Filelists"
                    / "images_filelist.txt",
                    "images_paths": images_paths,
                    "anns_paths": anns_paths,
                    "words_paths": words_paths,
                },
            ),
        ]

    def _generate_examples(
        self,
        xml_filelist: Path,
        images_filelist: Path,
        images_path: Path,
        anns_path: Path,
        words_path: Path,
    ):
        # read annotations
        with open(xml_filelist) as file:
            lines = file.readlines()
            lines = [l.split("/")[-1] for l in lines]
        xml_file_names = {
            f.strip().replace(".xml", "") for f in lines if f.strip().endswith(".xml")
        }
        with open(images_filelist) as file:
            lines = file.readlines()
        image_file_paths = {
            f.strip().replace(self.config.image_extension, "")
            for f in lines
            if f.strip().endswith(self.config.image_extension)
        }
        file_paths = sorted(xml_file_names.intersection(image_file_paths))
        logger.info(f"Generating {len(file_paths)} samples...")
        for idx, sample_file_path in enumerate(file_paths):
            ann_path = anns_path[sample_file_path + ".xml"]
            annotated_objects = read_pascal_voc(ann_path, labels=self.config.labels)
            image_file_path = images_path[
                sample_file_path + self.config.image_extension
            ]
            yield DocumentInstance(
                sample_id=Path(image_file_path).name,
                image=Image(file_path=image_file_path),
                gt=GroundTruth(
                    layout=LayoutAnalysisGT(
                        annotated_objects=AnnotatedObjectList.from_list(
                            annotated_objects
                        )
                    )
                ),
            )
