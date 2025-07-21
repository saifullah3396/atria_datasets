from pathlib import Path
from random import shuffle

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
    _REGISTRY_CONFIGS = {"main": AtriaDatasetConfig(data_urls=_DATA_URLS)}

    def _data_model(self) -> DocumentInstance:
        return DocumentInstance

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            citation=_CITATION,
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            dataset_labels=DatasetLabels(classification=_CLASSES),
        )

    def _split_configs(self, data_dir: str):
        data_dir = Path(data_dir)
        return [
            SplitConfig(
                split=DatasetSplitType.train,
                gen_kwargs={
                    "image_data_dir": data_dir / _IMAGE_DATA_NAME,
                    "ocr_data_dir": data_dir / _OCR_DATA_NAME,
                    "split_file_paths": data_dir / "train.txt",
                },
            ),
            SplitConfig(
                split=DatasetSplitType.test,
                gen_kwargs={
                    "image_data_dir": data_dir / _IMAGE_DATA_NAME,
                    "ocr_data_dir": data_dir / _OCR_DATA_NAME,
                    "split_file_paths": data_dir / "test.txt",
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
            shuffle(split_file_paths)

        for _, image_file_path in enumerate(split_file_paths):
            label_index = _CLASSES.index(Path(image_file_path).parent.name)
            ocr_file_path = Path(ocr_data_dir) / image_file_path.replace(
                ".jpg", ".hocr"
            )
            image_file_path = Path(image_data_dir) / image_file_path
            yield label_index, image_file_path, ocr_file_path

    def _input_transform(self, sample: tuple[int, str, str]) -> DocumentInstance:
        label_index, image_file_path, ocr_file_path = sample
        return DocumentInstance(
            sample_id=Path(image_file_path).name,
            image=Image(file_path=image_file_path),
            ocr=OCR(file_path=ocr_file_path, type=OCRType.tesseract),
            gt=GroundTruth(
                classification=ClassificationGT(
                    label=Label(name=_CLASSES[label_index], value=label_index)
                )
            ),
        )
