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
import os
from pathlib import Path

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


class DocBankConfig(AtriaDatasetConfig):
    max_words_per_sample: int = 4000


@DATASET.register("docbank")
class DocBankLER(AtriaDocumentDataset):
    _REGISTRY_CONFIGS = {"default": DocBankConfig(data_urls=_DATA_URLS)}

    def _data_model(self) -> DocumentInstance:
        return DocumentInstance

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            citation=_CITATION,
            license=_LICENSE,
            homepage=_HOMEPAGE,
            dataset_labels=DatasetLabels(layout=_CLASSES),
        )

    def _split_configs(self, data_dir: str):
        data_dir = Path(data_dir)
        return [
            SplitConfig(
                split=DatasetSplitType.train,
                gen_kwargs={
                    "image_base_dir": data_dir / "DocBank_500K_ori_img/",
                    "annotation_base_dir": data_dir / "DocBank_500K_txt/",
                    "split_filepath": data_dir / "500K_train.json",
                },
            ),
            SplitConfig(
                split=DatasetSplitType.test,
                gen_kwargs={
                    "image_base_dir": data_dir / "DocBank_500K_ori_img/",
                    "annotation_base_dir": data_dir / "DocBank_500K_txt/",
                    "split_filepath": data_dir / "500K_test.json",
                },
            ),
            SplitConfig(
                split=DatasetSplitType.validation,
                gen_kwargs={
                    "image_base_dir": data_dir / "DocBank_500K_ori_img/",
                    "annotation_base_dir": data_dir / "DocBank_500K_txt/",
                    "split_filepath": data_dir / "500K_valid.json",
                },
            ),
        ]

    def _load_ground_truth(self, text_file: Path, image_file_path: Path):
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
                word_bboxes.append(bbox)  # boxes are already normalized 0 to 1000
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

    def _split_iterator(
        self,
        split: DatasetSplitType,
        image_base_dir: str,
        annotation_base_dir: str,
        split_filepath: str,
    ):
        image_base_dir = Path(image_base_dir)
        annotation_base_dir = Path(annotation_base_dir)
        with open(split_filepath) as f:
            split_data = json.load(f)
            for idx in range(len(split_data["images"])):
                image_file_path = split_data["images"][idx]["file_name"]
                text_file = os.path.join(
                    annotation_base_dir,
                    image_file_path.replace("_ori.jpg", "") + ".txt",
                )
                ground_truth = self._load_ground_truth(
                    text_file, image_base_dir / image_file_path
                )
                if (
                    len(ground_truth.ser.words) > 0
                    and len(ground_truth.ser.words) < self.max_words_per_sample
                ):
                    yield DocumentInstance(
                        sample_id=Path(image_file_path).name,
                        image=Image(file_path=image_file_path),
                        gt=ground_truth,
                    )
