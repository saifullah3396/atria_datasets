import json
import pickle
from pathlib import Path

import pandas as pd
import tqdm
from atria_core.logger.logger import get_logger
from atria_core.types import (
    AtriaDatasetConfig,
    BoundingBoxList,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    GroundTruth,
    Image,
    QuestionAnswerPair,
    SplitConfig,
    VisualQuestionAnswerGT,
)
from datasets import load_from_disk

from atria_datasets import DATASET, AtriaDocumentDataset

from .utilities import (
    _get_line_bboxes,
    _normalize_bbox,
    anls_metric_str,
    extract_start_end_index_v1,
    extract_start_end_index_v2,
    extract_start_end_index_v3,
)

logger = get_logger(__name__)

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """DocVQA Receipts Dataset"""

_HOMEPAGE = "https://rrc.cvc.uab.es/?ch=13"

_LICENSE = "Apache-2.0 license"


def find_answers_in_words(words, answers, extraction_method="v1"):
    if extraction_method == "v1":
        return extract_start_end_index_v1(answers, words)
    elif extraction_method == "v2":
        return extract_start_end_index_v2(answers, words)
    elif extraction_method == "v1_v2":
        processed_answers, all_not_found = extract_start_end_index_v1(answers, words)
        if all_not_found:
            processed_answers, _ = extract_start_end_index_v2(answers, words)
        return processed_answers, all_not_found
    elif extraction_method == "v2_v1":
        processed_answers, all_not_found = extract_start_end_index_v2(answers, words)
        if all_not_found:
            processed_answers, _ = extract_start_end_index_v1(answers, words)
        return processed_answers, all_not_found
    elif extraction_method == "v3":
        processed_answers, all_not_found = extract_start_end_index_v3(answers, words)
        return processed_answers, all_not_found
    else:
        raise ValueError(f"Extraction method {extraction_method} not supported")


class DocVQAConfig(AtriaDatasetConfig):
    answers_extraction_method: str = "v1"


@DATASET.register("docvqa")
class DocVQA(AtriaDocumentDataset):
    _REGISTRY_CONFIGS = {"default": DocVQAConfig(), "with_msr_ocr": DocVQAConfig()}

    def _data_model(self) -> DocumentInstance:
        return DocumentInstance

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            dataset_labels=DatasetLabels(),
        )

    def _split_configs(self, data_dir: str):
        data_dir = Path(data_dir)
        assert data_dir.exists(), (
            f"Data directory {data_dir} does not exist. You must download the dataset first from homepage: {_HOMEPAGE}"
        )
        return [
            SplitConfig(
                split=DatasetSplitType.validation,
                gen_kwargs={
                    "filepath": data_dir / "spdocvqa_qas" / "val_v1.0_withQT.json",
                    "data_dir": data_dir,
                },
            ),
            SplitConfig(
                split=DatasetSplitType.train,
                gen_kwargs={
                    "filepath": data_dir / "spdocvqa_qas" / "train_v1.0_withQT.json",
                    "data_dir": data_dir,
                },
            ),
            SplitConfig(
                split=DatasetSplitType.test,
                gen_kwargs={
                    "filepath": data_dir / "spdocvqa_qas" / "test_v1.0.json",
                    "data_dir": data_dir,
                },
            ),
        ]

    def _read_dataset_from_filepath(self, filepath: str):
        with open(filepath) as f:
            return pd.DataFrame(json.load(f)["data"])

    def _load_recognition_result_from_sample(self, data_dir: str, sample: dict):
        ocr_dir = Path(data_dir) / "spdocvqa_ocr"
        with open(
            ocr_dir
            / f"{sample['ucsf_document_id']}_{sample['ucsf_document_page_no']}.json"
        ) as f:
            ocr = json.load(f)

        if len(ocr["recognitionResults"]) > 1:
            raise ValueError(
                "More than one recognition result found in OCR file. This is not supported."
            )

        return ocr["recognitionResults"][0]

    def _extract_content_from_recognition_result(self, recognition_result: dict):
        image_width = recognition_result["width"]
        image_height = recognition_result["height"]
        image_size = (image_width, image_height)
        words = []
        word_bboxes = []
        segment_level_bboxes = []
        for line in recognition_result["lines"]:
            cur_line_bboxes = []
            for word_and_box in line["words"]:
                word = word_and_box["text"].strip()
                if word.startswith("http") or word == "":
                    continue

                x1, y1, x2, y2, x3, y3, x4, y4 = word_and_box["boundingBox"]
                bbox_x1 = min([x1, x2, x3, x4])
                bbox_x2 = max([x1, x2, x3, x4])
                bbox_y1 = min([y1, y2, y3, y4])
                bbox_y2 = max([y1, y2, y3, y4])
                words.append(word.lower())
                cur_line_bboxes.append(
                    _normalize_bbox([bbox_x1, bbox_y1, bbox_x2, bbox_y2], image_size)
                )

            if len(cur_line_bboxes) > 0:
                # add word box
                word_bboxes.extend(cur_line_bboxes)

                # add line box
                cur_line_bboxes = _get_line_bboxes(cur_line_bboxes)
                segment_level_bboxes.extend(cur_line_bboxes)
        assert len(segment_level_bboxes) == len(word_bboxes) == len(words)
        return words, word_bboxes, segment_level_bboxes

    def _extract_answers(self, split: DatasetSplitType, sample: dict, words: list):
        if split in [DatasetSplitType.train, DatasetSplitType.validation]:
            answers = (
                list({x.lower() for x in sample["answers"]})
                if "answers" in sample
                else []
            )
            processed_answers, _ = find_answers_in_words(
                words, answers, self.config.answers_extraction_method
            )
            answer_start_indices = [
                ans["answer_start_index"] for ans in processed_answers
            ]
            answer_end_indices = [ans["answer_end_index"] for ans in processed_answers]
            gold_answers = [ans["gold_answer"] for ans in processed_answers]
        else:
            answer_start_indices = [-1]
            answer_end_indices = [-1]
            gold_answers = [""]
        return answer_start_indices, answer_end_indices, gold_answers

    def _read_msr_data(self, split: DatasetSplitType, data_dir: str):
        if split in [DatasetSplitType.train, DatasetSplitType.validation]:
            filename = (
                "train_msr_ocr" if split == DatasetSplitType.train else "val_msr_ocr"
            )
            msr_file = Path(data_dir) / "msr" / filename
            return load_from_disk(str(msr_file))
        else:
            msr_file = Path(data_dir) / "msr" / "test_v1.0_msr.json"
            with open(msr_file, encoding="utf-8") as read_file:
                return json.load(read_file)

    def _extract_content_from_msr_data(self, sample: dict):
        words = [word.lower() for word in sample["words"]]
        word_bboxes = sample["boxes"] if "boxes" in sample else sample["layout"]
        segment_level_bboxes = word_bboxes
        return (
            words,
            word_bboxes,
            segment_level_bboxes,
        )  # segment_level_bboxes same as word_bboxes as we don't have the layout info

    def _load_dataset(self, data_dir: str, split: DatasetSplitType, filepath: str):
        preprocessed_dataset = []
        all_gold_answers = []
        all_extracted_answers = []
        total_answers_found = 0
        dataset = self._read_dataset_from_filepath(filepath)
        for _, sample in tqdm.tqdm(
            dataset.iterrows(), desc=f"Preprocessing dataset split [{split.value}]"
        ):
            # load ocr
            recognition_result = self._load_recognition_result_from_sample(
                data_dir, sample
            )

            # extract words and boxes
            words, word_bboxes, segment_level_bboxes = (
                self._extract_content_from_recognition_result(recognition_result)
            )

            # extract answers
            answer_start_indices, answer_end_indices, current_gold_answers = (
                self._extract_answers(split, sample, words)
            )

            # find extracted and gold answers
            current_extracted_answers = []
            for start_word_id in answer_start_indices:
                if start_word_id != -1:
                    total_answers_found += 1
                    break

            for start_word_id, end_word_id in zip(
                answer_start_indices, answer_end_indices, strict=True
            ):
                if start_word_id != -1:
                    current_extracted_answers.append(
                        " ".join(words[start_word_id : end_word_id + 1])
                    )
                    break

            if len(current_extracted_answers) > 0:
                all_extracted_answers.append(current_extracted_answers)
                all_gold_answers.append(current_gold_answers)

            sample = {
                # image
                "image_file_path": Path(data_dir)
                / sample["image"].replace("documents", "spdocvqa_images"),
                # text
                "words": words,
                "word_bboxes": word_bboxes,
                "segment_level_bboxes": segment_level_bboxes,
                # question/answer
                "question_id": sample["questionId"],
                "question": sample["question"].lower(),
                "gold_answers": current_gold_answers,
                "answer_start_indices": answer_start_indices,
                "answer_end_indices": answer_end_indices,
            }
            preprocessed_dataset.append(sample)
        return (
            preprocessed_dataset,
            all_gold_answers,
            all_extracted_answers,
            total_answers_found,
        )

    def _load_dataset_with_msr(
        self, split: DatasetSplitType, data_dir: str, filepath: str
    ):
        dataset = self._read_dataset_from_filepath(filepath)
        msr_data = self._read_msr_data(split, data_dir)
        preprocessed_dataset = []
        all_gold_answers = []
        all_extracted_answers = []
        total_answers_found = 0
        for idx, sample in tqdm.tqdm(
            dataset.iterrows(), desc=f"Preprocessing dataset split [{split.value}]"
        ):
            assert sample["questionId"] == msr_data[idx]["questionId"]

            # extract words and boxes
            words, word_bboxes, segment_level_bboxes = (
                self._extract_content_from_msr_data(msr_data[idx])
            )

            # extract answers
            answer_start_indices, answer_end_indices, current_gold_answers = (
                self._extract_answers(split, sample, words)
            )

            # find extracted and gold answers
            current_extracted_answers = []
            for start_word_id in answer_start_indices:
                if start_word_id != -1:
                    total_answers_found += 1
                    break

            for start_word_id, end_word_id in zip(
                answer_start_indices, answer_end_indices, strict=True
            ):
                if start_word_id != -1:
                    current_extracted_answers.append(
                        " ".join(words[start_word_id : end_word_id + 1])
                    )
                    break

            if len(current_extracted_answers) > 0:
                all_extracted_answers.append(current_extracted_answers)
                all_gold_answers.append(current_gold_answers)

            sample = {
                # image
                "image_file_path": Path(data_dir)
                / sample["image"].replace("documents", "spdocvqa_images"),
                # text
                "words": words,
                "word_bboxes": word_bboxes,
                "segment_level_bboxes": segment_level_bboxes,
                # question/answer
                "question_id": sample["questionId"],
                "question": sample["question"].lower(),
                "gold_answers": current_gold_answers,
                "answer_start_indices": answer_start_indices,
                "answer_end_indices": answer_end_indices,
            }
            preprocessed_dataset.append(sample)
        return (
            preprocessed_dataset,
            all_gold_answers,
            all_extracted_answers,
            total_answers_found,
        )

    def _load_and_preprocess_dataset(
        self, split: DatasetSplitType, data_dir: str, filepath: str
    ) -> list[dict]:
        preprocessed_dataset_path = (
            Path(data_dir) / f"{split.value}_preprocessed_dataset.pkl"
        )
        if not hasattr(self, "_preprocess_dataset"):
            if preprocessed_dataset_path.exists():
                with open(preprocessed_dataset_path, "rb") as pickle_file:
                    self._preprocessed_dataset = pickle.load(pickle_file)
                return self._preprocessed_dataset
            if self.config_name == "with_msr_ocr":
                (
                    self._preprocessed_dataset,
                    all_gold_answers,
                    all_extracted_answers,
                    total_answers_found,
                ) = self._load_dataset_with_msr(split, data_dir, filepath)
            else:
                (
                    self._preprocessed_dataset,
                    all_gold_answers,
                    all_extracted_answers,
                    total_answers_found,
                ) = self._load_dataset(split, data_dir, filepath)
            if split in [DatasetSplitType.train, DatasetSplitType.validation]:
                _, anls = anls_metric_str(
                    predictions=all_extracted_answers, gold_labels=all_gold_answers
                )
                total_questions_in_dataset = len(self._preprocessed_dataset)
                logger.info(f"Preprocessed {filepath} dataset statistics:")
                logger.info(f"Extracted answers: {all_extracted_answers[:100]}")
                logger.info(f"Extracted gold answers: {all_gold_answers[:100]}")
                logger.info(f"Ground truth ANLS: {anls}")
                logger.info(f"Total questions in dataset: {total_questions_in_dataset}")
                logger.info(f"Total answers found: {total_answers_found}")
                logger.info(
                    f"Total answers not found: {total_questions_in_dataset - total_answers_found}"
                )
            with open(preprocessed_dataset_path, "wb") as pickle_file:
                pickle.dump(self._preprocessed_dataset, pickle_file)
            logger.info(f"Preprocessed dataset saved to {preprocessed_dataset_path}")
        return self._preprocessed_dataset

    def _split_iterator(self, split: DatasetSplitType, data_dir: str, filepath: str):
        preprocessed_dataset = self._load_and_preprocess_dataset(
            split, data_dir, filepath
        )
        for sample in preprocessed_dataset:
            yield DocumentInstance(
                sample_id=Path(sample["image_file_path"]).name,
                image=Image(file_path=sample["image_file_path"]),
                gt=GroundTruth(
                    vqa=VisualQuestionAnswerGT(
                        qa_pair=QuestionAnswerPair(
                            id=sample["question_id"],
                            question_text=sample["question"],
                            answer_start=sample["answer_start_indices"],
                            answer_end=sample["answer_end_indices"],
                            answer_text=sample["gold_answers"],
                        ),
                        words=sample["words"],
                        word_bboxes=BoundingBoxList(value=sample["word_bboxes"]),
                        segment_level_bboxes=BoundingBoxList(
                            value=sample["segment_level_bboxes"]
                        ),
                    )
                ),
            )
