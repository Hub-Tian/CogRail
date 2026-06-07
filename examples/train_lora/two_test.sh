#!/bin/bash

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/data/Projects/ZhangQY/Model}"

MODEL_NAME="Qwen2.5-VL"
MODEL_PATH="${BASE_MODEL_DIR}/Qwen2.5-VL"
TEMPLATE="qwen2_vl"

SAVE_ROOT="${SAVE_ROOT:-saves/rebuttal_MRSI_predictions}"
RUN_TAG="${RUN_TAG:-Qwen2.5_VL_4weights_2datasets_$(date +%Y%m%d_%H%M%S)}"
PREDICT_ROOT="${PREDICT_ROOT:-${SAVE_ROOT}/${RUN_TAG}}"

CUTOFF_LEN="${CUTOFF_LEN:-3200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
PRECISION="${PRECISION:-fp16}"

# 4 个训练得到的 LoRA 权重后缀
ADAPTER_LIST=(
    "without_move_ans"
    "without_move_ans_I"
    "without_pos_ans"
    "without_pos_ans_I"
)

COMMON_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    COMMON_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

PRECISION_ARGS=()
case "${PRECISION}" in
    bf16)
        PRECISION_ARGS+=(--bf16)
        ;;
    fp16)
        PRECISION_ARGS+=(--fp16)
        ;;
    fp32)
        ;;
    *)
        echo "Unsupported PRECISION=${PRECISION}. Use bf16, fp16, or fp32." >&2
        exit 1
        ;;
esac

get_eval_dataset() {
    local adapter_suffix="$1"

    if [[ "${adapter_suffix}" == *_I ]]; then
        echo "Cog-MRSI_RailThreat_Type_I_test"
    else
        echo "Cog-MRSI_RailThreat_Type_II_test"
    fi
}

get_data_file() {
    local adapter_suffix="$1"

    if [[ "${adapter_suffix}" == *_I ]]; then
        echo "/data/Projects/SongZH/Cog-MRSI/data_after_deal_masked_obstacle/Type_I/test/QAs/Cog-MRSI_RailThreat_Type_I_test.json"
    else
        echo "/data/Projects/SongZH/Cog-MRSI/data_after_deal_masked_obstacle/Type_II/test/QAs/Cog-MRSI_RailThreat_Type_II_test.json"
    fi
}

run_predict() {
    local adapter_suffix="$1"

    local eval_dataset
    local data_file
    eval_dataset="$(get_eval_dataset "${adapter_suffix}")"
    data_file="$(get_data_file "${adapter_suffix}")"

    local adapter_path="saves/${MODEL_NAME}/lora/MRSI_sft_${adapter_suffix}"
    local output_dir="${PREDICT_ROOT}/${adapter_suffix}_${eval_dataset}"
    local jsonl_file="${output_dir}/generated_predictions.jsonl"
    local json_file="${output_dir}/generated_predictions.json"

    if [[ ! -d "${adapter_path}" ]]; then
        echo "Missing adapter directory: ${adapter_path}" >&2
        exit 1
    fi

    if [[ ! -f "${data_file}" ]]; then
        echo "Missing source data file: ${data_file}" >&2
        exit 1
    fi

    echo "======================================"
    echo "Start prediction"
    echo "Model:        ${MODEL_NAME}"
    echo "Base model:   ${MODEL_PATH}"
    echo "Adapter:      ${adapter_path}"
    echo "Eval dataset: ${eval_dataset}"
    echo "Data file:    ${data_file}"
    echo "Output dir:   ${output_dir}"
    echo "======================================"

    llamafactory-cli train \
        --model_name_or_path "${MODEL_PATH}" \
        --adapter_name_or_path "${adapter_path}" \
        --trust_remote_code \
        --stage sft \
        --do_predict \
        --finetuning_type lora \
        --eval_dataset "${eval_dataset}" \
        --template "${TEMPLATE}" \
        --cutoff_len "${CUTOFF_LEN}" \
        --overwrite_cache \
        --preprocessing_num_workers 16 \
        --dataloader_num_workers 4 \
        --output_dir "${output_dir}" \
        --overwrite_output_dir \
        --report_to none \
        --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
        --predict_with_generate \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        "${PRECISION_ARGS[@]}" \
        --ddp_timeout 180000000 \
        "${COMMON_ARGS[@]}"

    python - "${jsonl_file}" "${json_file}" "${data_file}" "${MODEL_NAME}" "${MODEL_PATH}" "${adapter_path}" "${eval_dataset}" "${adapter_suffix}" <<'PY'
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
data_path = Path(sys.argv[3])
model_name = sys.argv[4]
model_path = sys.argv[5]
adapter_path = sys.argv[6]
eval_dataset = sys.argv[7]
adapter_suffix = sys.argv[8]

if not jsonl_path.is_file():
    raise SystemExit(f"Prediction file not found: {jsonl_path}")

with jsonl_path.open("r", encoding="utf-8") as f:
    predictions = [json.loads(line) for line in f if line.strip()]

with data_path.open("r", encoding="utf-8") as f:
    samples = json.load(f)

if len(predictions) > len(samples):
    raise SystemExit(
        f"More predictions than source samples: {len(predictions)} > {len(samples)}"
    )

records = []
for idx, pred in enumerate(predictions):
    sample = samples[idx]
    records.append(
        {
            "id": idx,
            "model_name": model_name,
            "model_name_or_path": model_path,
            "adapter_name_or_path": adapter_path,
            "adapter_suffix": adapter_suffix,
            "eval_dataset": eval_dataset,
            "images": sample.get("images", []),
            "system": sample.get("system", ""),
            "conversations": sample.get("conversations", []),
            "prompt": pred.get("prompt", ""),
            "predict": pred.get("predict", ""),
            "label": pred.get("label", ""),
        }
    )

json_path.parent.mkdir(parents=True, exist_ok=True)
with json_path.open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Saved {len(records)} records to {json_path}")
PY

    echo "===== Finished adapter ${adapter_suffix}; JSON: ${json_file} ====="
}

for adapter_suffix in "${ADAPTER_LIST[@]}"; do
    run_predict "${adapter_suffix}"
done

echo "All predictions finished."
echo "Prediction root: ${PREDICT_ROOT}"