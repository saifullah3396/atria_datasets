# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DockBank dataset"""

import json
from collections.abc import Generator, Iterable
from pathlib import Path

from atria_core.logger.logger import get_logger
from atria_core.types import (
    SERGT,
    BoundingBoxList,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    GroundTruth,
    Image,
    Label,
    LabelList,
)

from atria_datasets import DATASET
from atria_datasets.core.dataset.atria_dataset import AtriaDocumentDataset

logger = get_logger(__name__)

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@misc{li2020docbank,
    title={DocBank: A Benchmark Dataset for Document Layout Analysis},
    author={Minghao Li and Yiheng Xu and Lei Cui and Shaohan Huang and Furu Wei and Zhoujun Li and Ming Zhou},
    year={2020},
    eprint={2006.01038},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

# You can copy an official description
_DESCRIPTION = """\
DocBank is a new large-scale dataset that is constructed using a weak supervision approach.
It enables models to integrate both the textual and layout information for downstream tasks.
The current DocBank dataset totally includes 500K document pages, where 400K for training, 50K for validation and 50K for testing.
"""

_HOMEPAGE = "https://doc-analysis.github.io/docbank-page/index.html"

_LICENSE = "Apache-2.0 license"

_DATA_URLS = [
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.001",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.002",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.003",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.004",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.005",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.006",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.007",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.008",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.009",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.010",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_txt.zip",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/MSCOCO_Format_Annotation.zip",
]

_CLASSES = [
    "abstract",
    "author",
    "caption",
    "equation",
    "figure",
    "footer",
    "list",
    "paragraph",
    "reference",
    "section",
    "table",
    "title",
    "date",
]


@DATASET.register("docbank")
class DocBankLER(AtriaDocumentDataset):
    _REGISTRY_CONFIGS = {
        "1k": {
            "max_train_samples": 1000,
            "max_validation_samples": 1000,
            "max_test_samples": 1000,
        }
    }

    def __init__(self, max_words_per_sample: int = 4000, **kwargs):
        self._max_words_per_sample = max_words_per_sample
        super().__init__(**kwargs)

    def _download_urls(self) -> list[str]:
        return _DATA_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            citation=_CITATION,
            license=_LICENSE,
            homepage=_HOMEPAGE,
            dataset_labels=DatasetLabels(layout=_CLASSES),
        )

    def _available_splits(self) -> list[DatasetSplitType]:
        return [
            DatasetSplitType.train,
            DatasetSplitType.validation,
            DatasetSplitType.test,
        ]

    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Iterable[DocumentInstance]:
        class SplitIterator:
            def __init__(self, split: DatasetSplitType, data_dir: str):
                self.split = split
                self.data_dir = Path(data_dir)
                self.max_words_per_sample = getattr(self, "max_words_per_sample", 4000)

                # Set up paths based on split
                if split == DatasetSplitType.train:
                    self.split_filepath = (
                        self.data_dir / "MSCOCO_Format_Annotation/500K_train.json"
                    )
                elif split == DatasetSplitType.test:
                    self.split_filepath = (
                        self.data_dir / "MSCOCO_Format_Annotation/500K_test.json"
                    )
                elif split == DatasetSplitType.validation:
                    self.split_filepath = (
                        self.data_dir / "MSCOCO_Format_Annotation/500K_valid.json"
                    )

                self.image_base_dir = self.data_dir / "DocBank_500K_ori_img/"
                self.annotation_base_dir = self.data_dir / "DocBank_500K_txt/"

            def _load_ground_truth(
                self, text_file: Path, image_file_path: Path
            ) -> GroundTruth:
                words = []
                word_bboxes = []
                word_labels = []

                with open(text_file, encoding="utf8") as fp:
                    for line in fp.readlines():
                        tts = line.split("\t")
                        if not len(tts) == 10:
                            logger.warning(f"Incomplete line in file {text_file}")
                            continue

                        word = tts[0]
                        bbox = list(map(int, tts[1:5]))
                        structure = tts[9]

                        if len(word) == 0:
                            continue
                        x1, y1, x2, y2 = bbox
                        area = (x2 - x1) * (y2 - y1)
                        if (
                            word == "##LTLine##" or word == "##LTFigure##" and area < 10
                        ):  # remove ltline and ltfigures with very small noisy features
                            continue

                        words.append(word)
                        word_bboxes.append(
                            bbox
                        )  # boxes are already normalized 0 to 1000
                        word_labels.append(structure.strip())

                return GroundTruth(
                    ser=SERGT(
                        words=words,
                        word_bboxes=BoundingBoxList(value=word_bboxes),
                        word_labels=LabelList.from_list(
                            [
                                Label(value=_CLASSES.index(word_label), name=word_label)
                                for word_label in word_labels
                            ]
                        ),
                    )
                )

            def __iter__(self) -> Generator[DocumentInstance, None, None]:
                with open(self.split_filepath) as f:
                    split_data = json.load(f)
                    for idx in range(len(split_data["images"])):
                        image_file_path = split_data["images"][idx]["file_name"]
                        text_file = self.annotation_base_dir / (
                            image_file_path.replace("_ori.jpg", "") + ".txt"
                        )

                        ground_truth = self._load_ground_truth(
                            text_file, self.image_base_dir / image_file_path
                        )

                        if (
                            len(ground_truth.ser.words) > 0
                            and len(ground_truth.ser.words) < self.max_words_per_sample
                        ):
                            yield DocumentInstance(
                                sample_id=Path(image_file_path).name,
                                image=Image(
                                    file_path=self.image_base_dir / image_file_path
                                ),
                                gt=ground_truth,
                            )

            def __len__(self) -> int:
                with open(self.split_filepath) as f:
                    split_data = json.load(f)
                    return len(split_data["images"])

        return SplitIterator(split=split, data_dir=data_dir)
