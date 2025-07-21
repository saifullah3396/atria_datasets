#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DOCSETS_DIR=$SCRIPT_DIR/../
ATRIA_DIR=$SCRIPT_DIR/../../atria
ATRIA_CORE_DIR=$SCRIPT_DIR/../../atria_core
PYTHONPATH=$DOCSETS_DIR/src:$ATRIA_DIR/src:$ATRIA_CORE_DIR/src:$PYTHONPATH
DATASET_NAME=$1

declare -a available_datasets=(
    # "tobacco3482/main"
    # "rvlcdip/main"
    # "mnist/mnist"
    # "cord/cordv2"
    # "funsd/default"
    # "sroie/default"
    # "wild_receipts/default"
    # "docile/kile"
    # "docbank/default"
    # "docvqa/with_msr_ocr"
    # "publaynet/main"
    # "doclaynet/main"
    "icdar2019/trackA_modern" 
    # "icdar2013/main"
    # "fintabnet/main"
    # "pubtables1m/detection"
    # "pubtables1m/structure"
)

prepare_dataset() {
    local target_dataset_name="$1"
    local additional_args="${@:2}"

    for dataset in "${available_datasets[@]}"; do
        IFS=' ' read -r dataset_name <<<"$dataset"

        if [[ "$dataset_name" == "$target_dataset_name" ]]; then
            set -x
            PYTHONPATH=$PYTHONPATH python -m atria_datasets.prepare_dataset \
                $dataset_name \
                --max_samples=100 \
                $additional_args
            set +x
            return 0
        fi
    done

    echo "Dataset '$target_dataset_name' not found."
    return 1
}

if [ -z "$DATASET_NAME" ]; then
    echo "No dataset name provided. Please provide a dataset name or use 'all'."
    exit 1
elif [[ "$DATASET_NAME" == "all" ]]; then
    echo "Running prepare_dataset on all datasets..."
    for dataset in "${available_datasets[@]}"; do
        IFS=' ' read -r dataset_name <<<"$dataset"
        prepare_dataset "$dataset_name" "${@:2}"
    done
    exit 0
else
    prepare_dataset "$DATASET_NAME" "${@:2}"
fi
