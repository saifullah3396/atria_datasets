"""RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset"""

from pathlib import Path

from atria_core.types import (
    OCR,
    AtriaDatasetConfig,
    ClassificationGT,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    GroundTruth,
    Image,
    Label,
    OCRType,
    SplitConfig,
)

from atria_datasets import DATASET, AtriaDocumentDataset

_CITATION = """\
@inproceedings{harley2015icdar,
    title = {Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval},
    author = {Adam W Harley and Alex Ufkes and Konstantinos G Derpanis},
    booktitle = {International Conference on Document Analysis and Recognition ({ICDAR})}},
    year = {2015}
}
"""


_DESCRIPTION = """\
The RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset consists of 400,000 grayscale images in 16 classes, with 25,000 images per class. There are 320,000 training images, 40,000 validation images, and 40,000 test images.
"""


_HOMEPAGE = "https://www.cs.cmu.edu/~aharley/rvl-cdip/"


_LICENSE = "https://www.industrydocuments.ucsf.edu/help/copyright/"

_IMAGE_DATA_NAME = "rvl-cdip"
_OCR_DATA_NAME = "rvl-cdip-ocr"

_URLS = {
    f"{_IMAGE_DATA_NAME}.tar.gz": f"https://huggingface.co/datasets/rvl_cdip/resolve/main/data/{_IMAGE_DATA_NAME}.tar.gz",
    f"{_OCR_DATA_NAME}.tar.gz": f"https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/{_OCR_DATA_NAME}.tar.gz",
}

_METADATA_URLS = {  # for main let us always have tobacco3482 overlap removed from the dataset
    "main": {
        "labels/main/train.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/tobacco3482_excluded/train.txt",
        "labels/main/test.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/tobacco3482_excluded/test.txt",
        "labels/main/val.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/tobacco3482_excluded/val.txt",
    },
    "tobacco3482_included": {
        "labels/tobacco3482_included/train.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/default/train.txt",
        "labels/tobacco3482_included/test.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/default/test.txt",
        "labels/tobacco3482_included/val.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/default/val.txt",
    },
}

_CLASSES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]


@DATASET.register("rvlcdip")
class RvlCdip(AtriaDocumentDataset):
    """Ryerson Vision Lab Complex Document Information Processing dataset."""

    _REGISTRY_CONFIGS = {
        "main": AtriaDatasetConfig(data_urls={**_METADATA_URLS["main"], **_URLS})
    }

    def _data_model(self) -> DocumentInstance:
        return DocumentInstance

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            citation=_CITATION,
            license=_LICENSE,
            homepage=_HOMEPAGE,
            dataset_labels=DatasetLabels(classification=_CLASSES),
        )

    def _split_configs(self, data_dir: str):
        data_dir = Path(data_dir)
        image_data_dir = data_dir / _IMAGE_DATA_NAME / "images"
        ocr_data_dir = data_dir / _OCR_DATA_NAME / "images"
        return [
            SplitConfig(
                split=DatasetSplitType.train,
                gen_kwargs={
                    "image_data_dir": image_data_dir,
                    "ocr_data_dir": ocr_data_dir,
                    "split_file_paths": data_dir / "labels/main/train.txt",
                },
            ),
            SplitConfig(
                split=DatasetSplitType.test,
                gen_kwargs={
                    "image_data_dir": image_data_dir,
                    "ocr_data_dir": ocr_data_dir,
                    "split_file_paths": data_dir / "labels/main/test.txt",
                },
            ),
            SplitConfig(
                split=DatasetSplitType.validation,
                gen_kwargs={
                    "image_data_dir": image_data_dir,
                    "ocr_data_dir": ocr_data_dir,
                    "split_file_paths": data_dir / "labels/main/val.txt",
                },
            ),
        ]

    def _split_iterator(
        self,
        split: DatasetSplitType,
        image_data_dir: str,
        ocr_data_dir: str,
        split_file_paths: str,
    ):
        with open(split_file_paths) as f:
            split_file_paths = f.read().splitlines()

        for _, image_file_path_with_label in enumerate(split_file_paths):
            image_file_path, label = image_file_path_with_label.split(" ")
            yield DocumentInstance(
                sample_id=Path(image_file_path).name,
                image=Image(file_path=Path(image_data_dir) / image_file_path),
                ocr=OCR(
                    file_path=Path(ocr_data_dir)
                    / image_file_path.replace(".tif", ".hocr.lstm"),
                    ocr_type=OCRType.tesseract,
                ),
                gt=GroundTruth(
                    classification=ClassificationGT(
                        label=Label(value=int(label), name=_CLASSES[int(label)])
                    )
                ),
            )
