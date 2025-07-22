"""RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset"""

from collections.abc import Generator, Iterable
from pathlib import Path

from atria_core.types import (
    OCR,
    ClassificationGT,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    GroundTruth,
    Image,
    Label,
    OCRType,
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
    "default": {
        "labels/default/train.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/tobacco3482_excluded/train.txt",
        "labels/default/test.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/tobacco3482_excluded/test.txt",
        "labels/default/val.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/tobacco3482_excluded/val.txt",
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

    __extract_downloads__ = False
    _REGISTRY_CONFIGS = {
        "image": {"load_ocr": False},
        "image_with_ocr": {"load_ocr": True},
    }

    def __init__(
        self,
        max_train_samples: int | None = None,  # these get passed to the config
        max_validation_samples: int | None = None,  # these get passed to the config
        max_test_samples: int | None = None,  # these get passed to the config
        type: str = "default",  # type of dataset to load, e.g., "default" or "tobacco3482_included"
        load_ocr: bool = False,
    ):
        super().__init__(
            max_train_samples=max_train_samples,
            max_validation_samples=max_validation_samples,
            max_test_samples=max_test_samples,
        )
        self.type = type
        self.load_ocr = load_ocr

    def _download_urls(self) -> list[str]:
        if self.type in _METADATA_URLS:
            return list(_METADATA_URLS[self.type].values()) + list(_URLS.values())
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            citation=_CITATION,
            license=_LICENSE,
            homepage=_HOMEPAGE,
            dataset_labels=DatasetLabels(classification=_CLASSES),
        )

    def _available_splits(self):
        return [
            DatasetSplitType.train,
            DatasetSplitType.test,
            DatasetSplitType.validation,
        ]

    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Iterable[tuple[Path, Path, int]]:
        class SplitIterator(Iterable[tuple[Path, Path, int]]):
            def __init__(self, split: DatasetSplitType, data_dir: str):
                if split == DatasetSplitType.train:
                    split_file_paths = Path(data_dir) / "labels/main/train.txt"
                elif split == DatasetSplitType.test:
                    split_file_paths = Path(data_dir) / "labels/main/test.txt"
                elif split == DatasetSplitType.validation:
                    split_file_paths = Path(data_dir) / "labels/main/val.txt"
                with open(split_file_paths) as f:
                    self.split_file_paths = f.read().splitlines()
                self.image_data_dir = data_dir / _IMAGE_DATA_NAME / "images"
                self.ocr_data_dir = data_dir / _OCR_DATA_NAME / "images"

            def __iter__(self) -> Generator[tuple[Path, Path, int], None, None]:
                for image_file_path_with_label in self.split_file_paths:
                    image_file_path, label = image_file_path_with_label.split(" ")
                    ocr_file_path = Path(self.ocr_data_dir) / image_file_path.replace(
                        ".tif", ".hocr.lstm"
                    )
                    image_file_path = Path(self.image_data_dir) / image_file_path
                    yield image_file_path, ocr_file_path, label

            def __len__(self) -> int:
                return len(self.split_file_paths)

        return SplitIterator(split=split, data_dir=Path(data_dir))

    def _input_transform(self, sample: tuple[Path, Path, int]) -> DocumentInstance:
        image_file_path, ocr_file_path, label = sample

        return DocumentInstance(
            sample_id=Path(image_file_path).name,
            image=Image(file_path=image_file_path),
            ocr=OCR(file_path=ocr_file_path, ocr_type=OCRType.tesseract),
            gt=GroundTruth(
                classification=ClassificationGT(
                    label=Label(value=int(label), name=_CLASSES[int(label)])
                )
            ),
        )
