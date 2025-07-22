from collections.abc import Iterable
from pathlib import Path
from random import shuffle

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
@article{Kumar2014StructuralSF,
    title={Structural similarity for document image classification and retrieval},
    author={Jayant Kumar and Peng Ye and David S. Doermann},
    journal={Pattern Recognit. Lett.},
    year={2014},
    volume={43},
    pages={119-126}
}
"""

_DESCRIPTION = """\
The Tobacco3482 dataset consists of 3842 grayscale images in 10 classes. In this version, the dataset is plit into 2782 training images, and 700 test images.
"""

_HOMEPAGE = "https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg"
_LICENSE = "https://www.industrydocuments.ucsf.edu/help/copyright/"
_IMAGE_DATA_NAME = "tobacco3482"
_OCR_DATA_NAME = "tobacco3482_ocr"
_DATA_URLS = [
    f"https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/{_IMAGE_DATA_NAME}.tar.gz",
    f"https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/{_OCR_DATA_NAME}.tar.gz",
    "https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/train.txt",
    "https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/test.txt",
]
_CLASSES = [
    "Letter",
    "Resume",
    "Scientific",
    "ADVE",
    "Email",
    "Report",
    "News",
    "Memo",
    "Form",
    "Note",
]


@DATASET.register("tobacco3482")
class Tobacco3482(AtriaDocumentDataset):
    _REGISTRY_CONFIGS = {
        "image_only": {"load_ocr": False},
        "image_with_ocr": {"load_ocr": True},
    }

    def __init__(
        self,
        max_train_samples: int | None = None,  # these get passed to the config
        max_validation_samples: int | None = None,  # these get passed to the config
        max_test_samples: int | None = None,  # these get passed to the config
        load_ocr: bool = False,
    ):
        super().__init__(
            max_train_samples=max_train_samples,
            max_validation_samples=max_validation_samples,
            max_test_samples=max_test_samples,
        )
        self.load_ocr = load_ocr

    def _download_urls(self) -> list[str]:
        return _DATA_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            citation=_CITATION,
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            dataset_labels=DatasetLabels(classification=_CLASSES),
        )

    def _available_splits(self):
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Iterable[tuple[Path, Path, int]]:
        class SplitIterator:
            def __init__(self, split: DatasetSplitType, data_dir: str):
                if split == DatasetSplitType.train:
                    split_file_paths = Path(data_dir) / "train.txt"
                elif split == DatasetSplitType.test:
                    split_file_paths = Path(data_dir) / "test.txt"
                with open(split_file_paths) as f:
                    self.split_file_paths = f.read().splitlines()
                    shuffle(self.split_file_paths)
                self.image_data_dir = data_dir / _IMAGE_DATA_NAME
                self.ocr_data_dir = data_dir / _OCR_DATA_NAME

            def __iter__(self):
                for image_file_path in self.split_file_paths:
                    label_index = _CLASSES.index(Path(image_file_path).parent.name)
                    ocr_file_path = Path(self.ocr_data_dir) / image_file_path.replace(
                        ".jpg", ".hocr"
                    )
                    image_file_path = Path(self.image_data_dir) / image_file_path
                    yield image_file_path, ocr_file_path, label_index

            def __len__(self) -> int:
                return len(self.split_file_paths)

        return SplitIterator(split=split, data_dir=Path(data_dir))

    def _input_transform(self, sample: tuple[Path, Path, int]) -> DocumentInstance:
        image_file_path, ocr_file_path, label_index = sample
        return DocumentInstance(
            sample_id=Path(image_file_path).name,
            image=Image(file_path=image_file_path),
            ocr=OCR(file_path=ocr_file_path, type=OCRType.tesseract)
            if self.load_ocr
            else None,
            gt=GroundTruth(
                classification=ClassificationGT(
                    label=Label(name=_CLASSES[label_index], value=label_index)
                )
            ),
        )
