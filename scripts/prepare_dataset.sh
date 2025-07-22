#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DOCSETS_DIR=$SCRIPT_DIR/../
ATRIA_DIR=$SCRIPT_DIR/../../atria
ATRIA_CORE_DIR=$SCRIPT_DIR/../../atria_core
PYTHONPATH=$DOCSETS_DIR/src:$ATRIA_DIR/src:$ATRIA_CORE_DIR/src:$PYTHONPATH
DATASET_NAME=$1

declare -a available_datasets=(
    "cifar10"
    "huggingface_cifar10/plain_text"
    "tobacco3482/image_with_ocr"
    "rvlcdip/image_with_ocr_1k"
    "mnist/mnist"
    "cord"
    "funsd"
    "sroie"
    "wild_receipts"
    "docile/kile"
    "docbank/1k" 
    "fintabnet/1k" 
    "icdar2019/trackA_modern"
    "icdar2013" 
    # "doclaynet/2022.08" # too big failing downloads
    "docvqa/with_msr_ocr" 
    # "publaynet/default" # too big failing downloads
    # "pubtables1m/detection_1k" # too big failing downloads
    # "pubtables1m/structure_1k" # too big failing downloads
)

prepare_dataset() {
    local target_dataset_name="$1"
    local additional_args="${@:2}"

    for dataset in "${available_datasets[@]}"; do
        IFS=' ' read -r dataset_name <<<"$dataset"

        if [[ "$dataset_name" == "$target_dataset_name" ]]; then
            PYTHONPATH=$PYTHONPATH python -m atria_datasets.prepare_dataset \
                $dataset_name \
                $additional_args
            return 0
        fi
    done

    echo "Dataset '$target_dataset_name' not found."
    return 1
}

if [[ -z "$DATASET_NAME" ]]; then
    for dataset in "${available_datasets[@]}"; do
        IFS=' ' read -r dataset_name <<<"$dataset"
        prepare_dataset "$dataset_name" "${@:2}"
    done
    exit 0
else
    prepare_dataset "$DATASET_NAME" "${@:2}"
fi
