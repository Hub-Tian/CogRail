#!/bin/bash

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/data/Projects/ZhangQY/Model}"
DATASET="${DATASET:-Cog-MRSI_RailThreat_masked_Type_I_test}"
SAVE_DATASET="${SAVE_DATASET:-Cog-MRSI_RailThreat_masked_Type_I_MRSI_sft_$(date +%Y%m%d_%H%M%S)}"
DATA_FILE="${DATA_FILE:-/data/Projects/SongZH/Cog-MRSI/data_after_deal_masked_obstacle/Type_I/test/QAs/Cog-MRSI_RailThreat_Type_I_test.json}"
PREDICT_ROOT="${PREDICT_ROOT:-saves/rebuttal_MRSI_predictions/${SAVE_DATASET}}"
CUTOFF_LEN="${CUTOFF_LEN:-3200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
PRECISION="${PRECISION:-fp16}"

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

run_predict() {
    local model_name="$1"
    local model_path="$2"
    local template="$3"
    local adapter_path="saves/${model_name}/lora/MRSI_sft_I"
    local output_dir="${PREDICT_ROOT}/${model_name}"
    local jsonl_file="${output_dir}/generated_predictions.jsonl"
    local json_file="${output_dir}/generated_predictions.json"

    if [[ ! -d "${adapter_path}" ]]; then
        echo "Missing adapter directory: ${adapter_path}" >&2
        exit 1
    fi

    echo "===== Start prediction ${model_name} ====="
    llamafactory-cli train \
        --model_name_or_path "${model_path}" \
        --adapter_name_or_path "${adapter_path}" \
        --trust_remote_code \
        --stage sft \
        --do_predict \
        --finetuning_type lora \
        --eval_dataset "${DATASET}" \
        --template "${template}" \
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

    python - "${jsonl_file}" "${json_file}" "${DATA_FILE}" "${model_name}" "${model_path}" "${adapter_path}" <<'PY'
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
data_path = Path(sys.argv[3])
model_name = sys.argv[4]
model_path = sys.argv[5]
adapter_path = sys.argv[6]

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
    records.append(
        {
            "id": idx,
            "model_name": model_name,
            "model_name_or_path": model_path,
            "adapter_name_or_path": adapter_path,
            "images": samples[idx].get("images", []),
            "system": samples[idx].get("system", ""),
            "conversations": samples[idx].get("conversations", []),
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
    echo "===== Finished prediction ${model_name}; JSON: ${json_file} ====="
}

run_predict "Llama-3.2-Vision" "${BASE_MODEL_DIR}/Llama-3.2-Vision/Llama-3.2-11B-Vision-Instruct" "mllama"
run_predict "llava-1.6" "${BASE_MODEL_DIR}/llava-1.6" "llava_next"
run_predict "Qwen2.5-VL" "${BASE_MODEL_DIR}/Qwen2.5-VL" "qwen2_vl"
run_predict "Qwen2-VL" "${BASE_MODEL_DIR}/Qwen2-VL" "qwen2_vl"
run_predict "Yi-VL-6B" "${BASE_MODEL_DIR}/Yi-VL-6B" "yi_vl"
