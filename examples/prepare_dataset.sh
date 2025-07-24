#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

declare -a small_datasets=(
    # "cifar10/1k"
    # "huggingface_cifar10/plain_text_1k"
    # "tobacco3482/image_with_ocr"
    # "rvlcdip/image_with_ocr_1k"
    # "mnist/mnist_1k"
    # "cord/default"
    # "funsd/default"
    # "sroie/default"
    # "wild_receipts/default"
    # "docile/kile"
    "docbank/1k"  # too big, failing downloads
    # "fintabnet/1k" 
    # "icdar2019/trackA_modern"
    # "icdar2013/default"
    "docvqa/default" 
)


declare -a big_datasets=(
    "rvlcdip/image_with_ocr"
    "docbank/default"  
    "doclaynet/2022.08" # too big, failing downloads
    "publaynet/default" # too big, failing downloads
    "pubtables1m/detection_1k" # too big, failing downloads
    "pubtables1m/structure_1k" # too big, failing downloads
)


if [[ "$1" == "small_datasets" ]]; then
    for dataset_entry in "${available_datasets[@]}"; do
        IFS="/" read -r name config <<< "$dataset_entry"
        echo "Processing dataset: $name with config: $config"
        uv run python -m atria_datasets.prepare_dataset $name $config ${@:2}
    done
elif [[ "$1" == "big_datasets" ]]; then
    for dataset_entry in "${big_datasets[@]}"; do
        IFS="/" read -r name config <<< "$dataset_entry"
        echo "Processing dataset: $name with config: $config"
        uv run python -m atria_datasets.prepare_dataset $name $config ${@:2}
    done
else
    dataset_entry="$1"
    IFS="/" read -r name config <<< "$dataset_entry"
    echo "Processing dataset: $name with config: $config"
    uv run python -m atria_datasets.prepare_dataset $name $config ${@:2}
fi