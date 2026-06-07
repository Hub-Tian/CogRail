#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
set -euo pipefail

BASE_MODEL_DIR="/data/Projects/ZhangQY/Model"
TRAIN_SCRIPT="$(dirname "$0")/train_Type_I.sh"

run_train() {
    local model_name="$1"
    local model_path="$2"
    local template="$3"

    echo "===== Start training ${model_name} ====="
    MODEL_PATH="${model_path}" \
    TEMPLATE="${template}" \
    OUTPUT_DIR="saves/${model_name}/lora/MRSI_sft_I" \
        bash "${TRAIN_SCRIPT}"
    echo "===== Finished training ${model_name} ====="
}

run_train "Llama-3.2-Vision" "${BASE_MODEL_DIR}/Llama-3.2-Vision/Llama-3.2-11B-Vision-Instruct" "mllama"
run_train "llava-1.6" "${BASE_MODEL_DIR}/llava-1.6" "llava_next"
run_train "Qwen2.5-VL" "${BASE_MODEL_DIR}/Qwen2.5-VL" "qwen2_vl"
run_train "Qwen2-VL" "${BASE_MODEL_DIR}/Qwen2-VL" "qwen2_vl"
run_train "Yi-VL-6B" "${BASE_MODEL_DIR}/Yi-VL-6B" "yi_vl"
