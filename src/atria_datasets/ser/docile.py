import base64
import io
from pathlib import Path

import PIL
import pyarrow
import pyarrow_hotfix
from atria_core.logger.logger import get_logger
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
from docile.dataset import KILE_FIELDTYPES, LIR_FIELDTYPES, Dataset

from atria_datasets import DATASET
from atria_datasets.core.constants import _DEFAULT_DOWNLOAD_PATH
from atria_datasets.core.dataset.atria_dataset import AtriaDocumentDataset
from atria_datasets.core.download_manager.download_manager import DownloadManager

from .docile_utils.preprocessor import (
    generate_unique_entities,
    load_docile_dataset,
    prepare_docile_dataset,
)

pyarrow_hotfix.uninstall()
pyarrow.PyExtensionType.set_auto_load(True)

logger = get_logger(__name__)


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """Docile Dataset"""

_HOMEPAGE = "https://github.com/rossumai/docile"

_LICENSE = "Apache-2.0 license"


_ALL_LABELS = generate_unique_entities()
_KILE_LABELS = (
    ["O"] + [f"B-{x}" for x in KILE_FIELDTYPES] + [f"I-{x}" for x in KILE_FIELDTYPES]
)
_LIR_LABELS = (
    ["O"] + [f"B-{x}" for x in LIR_FIELDTYPES] + [f"I-{x}" for x in LIR_FIELDTYPES]
)

_DATA_URLS = [
    "https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/{access_token}/annotated-trainval.zip",
    "https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/{access_token}/test.zip",
]

_SYNTHETIC_DATA_URLS = [
    "https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/{access_token}/synthetic.zip"
]


def filter_empty_words(row):
    filtered_indices = [idx for idx, x in enumerate(row["words"]) if x != ""]
    for k in ["words", "word_bboxes", "word_bboxes_segment_level", "word_labels"]:
        if k not in row.keys():
            continue

        if k == "words":
            row[k] = [row[k][idx].strip() for idx in filtered_indices]
        else:
            row[k] = [row[k][idx] for idx in filtered_indices]
    return row


class DocileConfig(AtriaDatasetConfig):
    synthetic: bool = False
    overlap_threshold: float = 0.5
    image_shape: tuple = (1024, 1024)


@DATASET.register("docile")
class Docile(AtriaDocumentDataset):
    _REGISTRY_CONFIGS = {
        "kile": DocileConfig(synthetic=False, data_urls=_DATA_URLS),
        "lir": DocileConfig(synthetic=False, data_urls=_DATA_URLS),
        "kile_synthetic": DocileConfig(synthetic=True, data_urls=_SYNTHETIC_DATA_URLS),
        "lir_synthetic": DocileConfig(synthetic=True, data_urls=_SYNTHETIC_DATA_URLS),
    }

    def _data_model(self) -> DocumentInstance:
        return DocumentInstance

    def _metadata(self) -> DatasetMetadata:
        if self.config_name == "kile":
            labels = _KILE_LABELS
        elif self.config_name == "lir":
            labels = _KILE_LABELS
        return DatasetMetadata(
            citation=_CITATION,
            homepage=_HOMEPAGE,
            description=_DESCRIPTION,
            license=_LICENSE,
            dataset_labels=DatasetLabels(ser=labels),
        )

    def prepare_downloads(self, data_dir: str, access_token: str | None = None) -> None:
        if access_token is None:
            logger.warning(
                "access_token must be passed to download this dataset. "
                "See `https://github.com/rossumai/docile` for instructions to get the access token"
            )
            return

        download_dir = Path(data_dir) / _DEFAULT_DOWNLOAD_PATH
        download_dir.mkdir(parents=True, exist_ok=True)
        download_manager = DownloadManager(data_dir=data_dir, download_dir=download_dir)

        if self._data_urls is not None:
            self._downloaded_files = download_manager.download_and_extract(
                [url.format(access_token=access_token) for url in self._data_urls]
            )

    def _split_configs(self, data_dir: str):
        return [
            SplitConfig(
                split=DatasetSplitType.train,
                gen_kwargs={"data_dir": data_dir, "split_dir": "annotated-trainval"},
            ),
            SplitConfig(
                split=DatasetSplitType.validation,
                gen_kwargs={"data_dir": data_dir, "split_dir": "annotated-trainval"},
            ),
        ]

    def label_names(self):
        if self.config_name == "kile":
            return _KILE_LABELS
        elif self.config_name == "lir":
            return _LIR_LABELS
        else:
            return _ALL_LABELS

    def _remap_labels_to_task_labels(self, labels):
        import numpy as np

        all_labels = np.array(_ALL_LABELS)
        if self.config_name == "kile":
            label_map = _KILE_LABELS
        elif self.config_name == "lir":
            label_map = _LIR_LABELS

        labels_to_idx = dict(zip(label_map, range(len(label_map))))
        remapped_labels = []
        for label in labels:
            # each label is a boolean map to multiple unique entities in _ALL_LABELS
            # here we only take those labels that are present in the label_map (KILE OR LIR or other)
            sample_label = [x for x in all_labels[label] if x in label_map]
            if len(sample_label) > 0:  # now we take the label index from the label_map
                remapped_labels.append(labels_to_idx[sample_label[0]])
            else:
                remapped_labels.append(labels_to_idx[label_map[0]])
        return remapped_labels

    def _prepare_dataset(self, split: DatasetSplitType, data_dir: str, split_dir: str):
        if not hasattr(self, "_dataset"):
            data_dir = Path(data_dir)
            split_name = "val" if split == DatasetSplitType.validation else "train"
            docile_dataset = Dataset(
                split_name, data_dir / split_dir, load_annotations=False, load_ocr=False
            )
            preprocessed_name = f"{docile_dataset.split_name}_multilabel_preprocessed_withImgs_{self.image_shape[0]}x{self.image_shape[1]}.json"
            if not (data_dir / split_dir / split_dir / preprocessed_name).exists():
                prepare_docile_dataset(
                    docile_dataset,
                    self.overlap_threshold,
                    data_dir / split_dir,
                    image_shape=self.image_shape,
                )
            self._dataset = load_docile_dataset(
                docile_dataset, data_dir / split_dir, image_shape=self.image_shape
            ).as_pandas_dataset()
            self._label_names = self.label_names()
        return self._dataset, self._label_names

    def _split_iterator(self, split: DatasetSplitType, data_dir: str, split_dir: str):
        dataset, label_names = self._prepare_dataset(split, data_dir, split_dir)
        for _, row in dataset.iterrows():
            row["ner_tags"] = self._remap_labels_to_task_labels(row["ner_tags"])
            row["tokens"] = list(row["tokens"])
            yield DocumentInstance(
                sample_id=str(row["id"]),
                image=Image(
                    content=PIL.Image.open(io.BytesIO(base64.b64decode(row["img"])))
                ),
                gt=GroundTruth(
                    ser=SERGT(
                        words=row["tokens"],
                        word_bboxes=BoundingBoxList(value=row["bboxes"]),
                        word_labels=LabelList.from_list(
                            [
                                Label(value=label, name=label_names[label])
                                for label in row["ner_tags"]
                            ]
                        ),
                    )
                ),
            )
