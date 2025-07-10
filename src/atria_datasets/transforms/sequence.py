import inspect
import os
from collections.abc import Mapping
from typing import Any

import torch
from atria_core.constants import _DEFAULT_ATRIA_MODELS_CACHE_DIR
from atria_core.logger.logger import get_logger
from atria_core.types import DocumentInstance, QuestionAnswerPair, TaskType
from atria_core.types.tensor_types import (
    TensorTokenizedQuestionAnswerPair,
    TokenizedDocumentInstance,
)
from atria_registry import DATA_TRANSFORM
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from atria_datasets.core.transforms.base import DataTransform

logger = get_logger(__name__)


@DATA_TRANSFORM.register("tokenizer_object_sanitizer")
class TokenizerObjectSanitizer(DataTransform):
    """
    This class is used to sanitize the tokenized sequences to match the original sequence elements.
    For example, if you have labels for words, you can map them to the tokens.
    [0, 1, 2] -> ['hello', 'world', '!']
    This is mapped to tokens like so
    [<padding_element>, 0, <padding_element>, 1,  <padding_element>, 2, <padding_element>, ...] ->
    [<cls-token>, '#hel', '#lo', '#wor', '#ld' '!', <pad-token>, ...]
    """

    def __init__(self, apply_value_to_first_subword: bool = True):
        super().__init__(input_path=None)
        self._apply_value_to_first_subword = apply_value_to_first_subword

    def _sanitize_sequence(
        self, sequence: torch.Tensor, word_ids: torch.Tensor, padding_value: Any = -100
    ):
        sanitized = torch.ones(sequence.shape) * padding_value
        last_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id == -100:
                continue
            if (
                last_word_id is not None
                and last_word_id == word_id
                and self._apply_value_to_first_subword
            ):
                continue
            sanitized[idx] = sequence[word_id]
            last_word_id = word_id
        return sanitized

    def _apply_transform(
        self, tokenized_object_instance: TokenizedDocumentInstance
    ) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        tokenized_object_instance.token_ids = self._sanitize_sequence(
            tokenized_object_instance.token_ids,
            tokenized_object_instance.word_ids,
            padding_value=-100,
        )
        tokenized_object_instance.token_bboxes = self._sanitize_sequence(
            tokenized_object_instance.token_bboxes,
            tokenized_object_instance.word_ids,
            padding_value=[-100, -100, -100, -100],
        )
        tokenized_object_instance.token_labels = self._sanitize_sequence(
            tokenized_object_instance.token_labels,
            tokenized_object_instance.word_ids,
            padding_value=-100,
        )
        return tokenized_object_instance


@DATA_TRANSFORM.register("document_instance_tokenizer")
class DocumentInstanceTokenizer(DataTransform):
    _REGISTRY_CONFIGS = {
        "sequence_classification": {"task_type": TaskType.sequence_classification},
        "semantic_entity_recognition": {
            "task_type": TaskType.semantic_entity_recognition
        },
        "question_answering": {
            "task_type": TaskType.question_answering,
            "call_kwargs": {"stride": 128, "truncation": "only_second"},
        },
        "visual_question_answering": {
            "task_type": TaskType.visual_question_answering,
            "call_kwargs": {"stride": 128, "truncation": "only_second"},
        },
    }

    def __init__(
        self,
        tokenizer_name: str,
        task_type: TaskType,
        init_kwargs: dict | None = None,
        call_kwargs: dict | None = None,
        overflow_sampling: str = "return_all",
        max_overflow_samples: int = 10,
        processor_input_key_map: dict | None = None,
        use_ssl: bool = False,
        do_normalize: bool = True,
        do_resize: bool = True,
        resize_height: int = 224,
        resize_width: int = 224,
        image_mean: list[float] | None = IMAGENET_DEFAULT_MEAN,
        image_std: list[float] | None = IMAGENET_DEFAULT_STD,
    ):
        super().__init__()

        self._tokenizer_name = tokenizer_name
        self._task_type = task_type
        self._init_kwargs = init_kwargs if init_kwargs is not None else {}
        self._call_kwargs = call_kwargs if call_kwargs is not None else {}
        self._overflow_sampling = overflow_sampling
        self._max_overflow_samples = max_overflow_samples
        self._processor_input_key_map = (
            processor_input_key_map
            if processor_input_key_map is not None
            else {"text": "text"}
        )
        self._use_ssl = use_ssl
        self._do_normalize = do_normalize
        self._do_resize = do_resize
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._image_mean = image_mean
        self._image_std = image_std

        # how to return overflowing samples?
        assert self._overflow_sampling in [
            "return_all",
            "return_random_n",
            "no_overflow",
            "return_first_n",
        ], f"Overflow sampling strategy {self._overflow_sampling} is not supported."

        # initialize the tokenizer
        default_init_kwargs = {
            "cache_dir": os.path.join(_DEFAULT_ATRIA_MODELS_CACHE_DIR, "transformers"),
            "local_files_only": False,
            "apply_ocr": False,
            "image_mean": IMAGENET_DEFAULT_MEAN,
            "image_std": IMAGENET_DEFAULT_STD,
            "add_prefix_space": True,
            "do_lower_case": True,
            "do_normalize": False,  # normalize means imagenet normalization.
            "do_resize": False,  # resize means resizing to 224x224.
            "do_rescale": False,  # rescale means division by 1/255.
        }

        # setup the default call kwargs
        default_call_kwargs = {
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
            "max_length": 512,
            "stride": 0,
            "pad_to_multiple_of": 8,
            "is_split_into_words": True,
            "return_overflowing_tokens": self._overflow_sampling
            != "no_overflow",  # set some arguments that we need to stay fixed for our case
            "return_token_type_ids": None,
            "return_attention_mask": None,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_length": False,
            "return_tensors": "pt",
            "verbose": True,
        }
        self._init_kwargs = {**default_init_kwargs, **self._init_kwargs}
        self._call_kwargs = {**default_call_kwargs, **self._call_kwargs}

        # initialize the default input mapping
        if len(self._processor_input_key_map) == 0:
            logger.info(
                f"Default input mapping created: {self._processor_input_key_map}"
            )

        # lazy initialization of the processor to avoid significant transformers loading times
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._tokenizer_name, **self._init_kwargs
        )
        possible_args = inspect.signature(self._processor.__call__).parameters
        for key in list(self._call_kwargs.keys()):
            if key not in possible_args:
                logger.warning(
                    f"Invalid keyword argument '{key}' found in call_kwargs for {self.__class__.__name__}. Skipping it."
                )
                self._call_kwargs.pop(key)

    def _prepare_tokenizer_inputs(
        self, document_instance: DocumentInstance
    ) -> Mapping[str, Any]:
        ground_truth = document_instance.gt
        document_instance.image.convert_to_rgb().to_tensor()
        if self._do_resize:
            document_instance.image.resize(
                width=self._resize_width, height=self._resize_height
            )

        if self._do_normalize:
            document_instance.image.normalize(
                mean=self._image_mean, std=self._image_std
            )

        if self._task_type == TaskType.sequence_classification:
            assert ground_truth.ocr is not None, (
                "Ground truth of type 'ocr' is required for the task type 'sequence_classification'."
                " Available ground truth types: {}"
            ).format(
                [
                    key
                    for key, value in document_instance.gt.__dict__.items()
                    if value is not None
                ]
            )
            boxes = [
                bbox.value if bbox.value is not None else []
                for bbox in ground_truth.ocr.word_bboxes
            ]
            return {
                "text": ground_truth.ocr.words,
                "boxes": boxes,
                "images": document_instance.image.content,
            }
        elif self._task_type == TaskType.semantic_entity_recognition:
            assert ground_truth.ser is not None, (
                "Ground truth of type 'ser' is required for the task type 'semantic_entity_recognition'."
                " Available ground truth types: {}"
            ).format(
                [
                    key
                    for key, value in document_instance.gt.__dict__.items()
                    if value is not None
                ]
            )
            if self._use_ssl and ground_truth.ser.segment_level_bboxes is not None:
                boxes = [
                    bbox.value if bbox.value is not None else []
                    for bbox in ground_truth.ser.segment_level_bboxes
                ]
            else:
                boxes = [
                    bbox.value if bbox.value is not None else []
                    for bbox in ground_truth.ser.word_bboxes
                ]
            return {
                "text": ground_truth.ser.words,
                "boxes": boxes,
                "word_labels": [label.value for label in ground_truth.ser.word_labels],
                "images": document_instance.image.content,
            }
        elif self._task_type == TaskType.question_answering:
            assert ground_truth.qa is not None, (
                "Ground truth of type 'qa' is required for the task type 'visual_question_answering'."
                " Available ground truth types: {}"
            ).format(
                [
                    key
                    for key, value in document_instance.gt.__dict__.items()
                    if value is not None
                ]
            )
            return {
                "text": ground_truth.qa.qa_pair.question_text,
                "text_pair": ground_truth.qa.words,
            }
        elif self._task_type == TaskType.visual_question_answering:
            assert ground_truth.vqa is not None, (
                "Ground truth of type 'vqa' is required for the task type 'visual_question_answering'."
                " Available ground truth types: {}"
            ).format(
                [
                    key
                    for key, value in document_instance.gt.__dict__.items()
                    if value is not None
                ]
            )
            if self._use_ssl and ground_truth.vqa.segment_level_bboxes is not None:
                boxes = [
                    bbox.value if bbox.value is not None else []
                    for bbox in ground_truth.vqa.segment_level_bboxes
                ]
            else:
                boxes = [
                    bbox.value if bbox.value is not None else []
                    for bbox in ground_truth.vqa.word_bboxes
                ]
            return {
                "text": ground_truth.vqa.qa_pair.question_text,
                "text_pair": ground_truth.vqa.words,
                "boxes": boxes,
                "images": document_instance.image.content,
            }
        else:
            raise ValueError(
                f"Task type {self._task_type} is not supported. "
                "Supported task types are: {}".format(
                    [
                        TaskType.sequence_classification,
                        TaskType.semantic_entity_recognition,
                        TaskType.question_answering,
                        TaskType.visual_question_answering,
                    ]
                )
            )

    def _get_subword_start_end(self, word_start, word_end, word_ids, sequence_ids):
        start_of_context = -1
        for i in range(len(sequence_ids)):
            if sequence_ids[i] == 1:
                start_of_context = i
                break
        num_question_tokens = start_of_context
        assert start_of_context != -1, "Could not find the start of the context"
        subword_start = -1
        subword_end = -1
        for i in range(start_of_context, len(word_ids)):
            if word_start == word_ids[i] and subword_start == -1:
                subword_start = i
            if word_end == word_ids[i]:
                subword_end = i
        return subword_start, subword_end, num_question_tokens

    def _generate_qa_token_ids(
        self,
        qa_pair: QuestionAnswerPair,
        word_ids: torch.Tensor,
        sequence_ids: torch.Tensor,
        sequence_length: int = 512,
    ) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        tokenized_answer_starts, tokenized_answer_ends = [], []
        for word_ids_per_overflow, sequence_ids_per_overflow in zip(
            word_ids, sequence_ids, strict=True
        ):
            tokenized_answer_start, tokenized_answer_end = None, None
            if qa_pair.answer_start == -1:
                tokenized_answer_start = 0
                tokenized_answer_end = 0
            else:
                (tokenized_answer_start, tokenized_answer_end, _) = (
                    self._get_subword_start_end(
                        qa_pair.answer_start,
                        qa_pair.answer_end,
                        word_ids_per_overflow,
                        sequence_ids_per_overflow,
                    )
                )
                if tokenized_answer_start == -1:
                    tokenized_answer_start = 0
                    tokenized_answer_end = 0
                if tokenized_answer_end == -1:
                    tokenized_answer_end = sequence_length - 1
                assert tokenized_answer_end >= tokenized_answer_start, (
                    "End token index is less than start token index. "
                    "Something is wrong in the conversion from answer word indices to answer token indices."
                )
            tokenized_answer_starts.append(tokenized_answer_start)
            tokenized_answer_ends.append(tokenized_answer_end)
        return TensorTokenizedQuestionAnswerPair(
            **qa_pair.__dict__,
            tokenized_answer_starts=torch.tensor(
                tokenized_answer_starts, dtype=torch.long, device=word_ids.device
            ),
            tokenized_answer_ends=torch.tensor(
                tokenized_answer_ends, dtype=torch.long, device=word_ids.device
            ),
        )

    def _apply_transforms(
        self, document_instance: DocumentInstance
    ) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        assert not document_instance._is_batched, (
            f"Document instance {document_instance.id} is batched. "
            "This tokenizer only supports single document instances."
        )

        tokenized_samples_batch = self._processor(
            **self._prepare_tokenizer_inputs(document_instance), **self._call_kwargs
        )
        tokenized_samples_batch["word_ids"] = torch.tensor(
            [
                [-100 if x is None else x for x in tokenized_samples_batch.word_ids(i)]
                for i in range(len(tokenized_samples_batch["input_ids"]))
            ]
        )
        tokenized_samples_batch["sequence_ids"] = torch.tensor(
            [
                [
                    -100 if x is None else x
                    for x in tokenized_samples_batch.sequence_ids(i)
                ]
                for i in range(len(tokenized_samples_batch["input_ids"]))
            ]
        )

        # rename some fields
        tokenized_samples_batch["token_ids"] = tokenized_samples_batch.pop("input_ids")
        tokenized_samples_batch["token_bboxes"] = tokenized_samples_batch.pop(
            "bbox", None
        )
        tokenized_samples_batch["token_type_ids"] = tokenized_samples_batch.pop(
            "token_type_ids", None
        )
        tokenized_samples_batch["token_labels"] = tokenized_samples_batch.pop(
            "labels", None
        )
        if "pixel_values" in tokenized_samples_batch:
            document_instance.image.content = tokenized_samples_batch.pop(
                "pixel_values"
            )[0]

        # remove the ground truth of ser
        document_instance.gt.ser = None

        additional_kwargs = {}
        if self._task_type == TaskType.visual_question_answering:
            words = document_instance.gt.vqa.words
            qa_pair = document_instance.gt.vqa.qa_pair
            additional_kwargs = {"words": words, "qa_pair": qa_pair}

        document_instance = TokenizedDocumentInstance(
            # copy the document instance attributes
            image=document_instance.image,
            label=(
                document_instance.gt.classification.label
                if document_instance.gt.classification is not None
                else None
            ),
            # copy the tokenized samples batch attributes
            **tokenized_samples_batch,
            **additional_kwargs,
        )
        document_instance._tokenizer = self._processor.tokenizer

        if document_instance.qa_pair is not None:
            document_instance.qa_pair = self._generate_qa_token_ids(
                document_instance.qa_pair,
                document_instance.word_ids,
                document_instance.sequence_ids,
                sequence_length=document_instance.token_ids.shape[-1],
            )

        return document_instance

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(tokenizer_name={self._tokenizer_name}, "
            f"task_type={self._task_type}, init_kwargs={self._init_kwargs}, "
            f"call_kwargs={self._call_kwargs}, overflow_sampling={self._overflow_sampling}, "
            f"max_overflow_samples={self._max_overflow_samples}, "
            f"processor_input_key_map={self._processor_input_key_map})"
        )
