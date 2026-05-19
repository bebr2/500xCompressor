#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash run_sra_500x_baseline.sh MODEL_PATH LORA_PATH [DATASET] [METHOD] [OUTPUT_NAME] [extra runner args...]"
  echo "Example: bash run_sra_500x_baseline.sh /mnt/hdfs/wangchangyue/LLM/Qwen3-8B /mnt/hdfs/wangchangyue/500xCompressor/output/500xCompressor_finetune-Qwen3-8B/checkpoint_best/pytorch_model.bin champ golden_skill"
  exit 2
fi

MODEL_PATH="$1"
LORA_PATH="$2"
DATASET="${3:-champ}"
METHOD="${4:-golden_skill}"
OUTPUT_NAME="${5:-${METHOD}}"

if [ "$#" -ge 5 ]; then
  shift 5
else
  shift "$#"
fi

REPO_ROOT="${REPO_ROOT:-/mnt/hdfs/wangchangyue/500xCompressor}"
RERANK_ROOT="${RERANK_ROOT:-/mnt/hdfs/wangchangyue/Rerank}"
SKILLRAG_ROOT="${SKILLRAG_ROOT:-/mnt/hdfs/wangchangyue/skillrag}"
CORPUS_PATH="${CORPUS_PATH:-${SKILLRAG_ROOT}/data/bench/corpus/corpus.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RERANK_ROOT}/results}"

if [ ! -f "${CORPUS_PATH}" ] && [ -f "${RERANK_ROOT}/prepare/output/corpus.json" ]; then
  CORPUS_PATH="${RERANK_ROOT}/prepare/output/corpus.json"
fi

NUM_MEM="${NUM_MEM:-256}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TOOLQA_MAX_STEPS="${TOOLQA_MAX_STEPS:-20}"
TOOLQA_STEP_TOKENS="${TOOLQA_STEP_TOKENS:-512}"
TOP_K="${TOP_K:-1}"
FP16="${FP16:-false}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-1}"
RESUME="${RESUME:-0}"
DISABLE_PARALLEL="${DISABLE_PARALLEL:-0}"

MODEL_NAME="$(basename "${MODEL_PATH}")-500x-nm${NUM_MEM}"
RESULT_PATH="${RESULT_PATH:-${OUTPUT_ROOT}/${DATASET}/${MODEL_NAME}/${OUTPUT_NAME}.jsonl}"

ARGS=(
  "${REPO_ROOT}/codes/prediction/run_sra_baseline_inference.py"
  --baseline 500x
  --model_path "${MODEL_PATH}"
  --lora_path "${LORA_PATH}"
  --method "${METHOD}"
  --skillrag_root "${SKILLRAG_ROOT}"
  --rerank_root "${RERANK_ROOT}"
  --corpus_path "${CORPUS_PATH}"
  --num_mem "${NUM_MEM}"
  --max_length "${MAX_LENGTH}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --toolqa_max_steps "${TOOLQA_MAX_STEPS}"
  --toolqa_step_tokens "${TOOLQA_STEP_TOKENS}"
  --fp16 "${FP16}"
)

if [ "${DATASET}" = "all" ]; then
  ARGS+=(--output_root "${OUTPUT_ROOT}")
else
  ARGS+=(--dataset "${DATASET}" --result_path "${RESULT_PATH}")
fi

if [ "${LIMIT}" != "0" ]; then
  ARGS+=(--limit "${LIMIT}")
fi

if [ "${OVERWRITE}" = "1" ]; then
  ARGS+=(--overwrite)
fi

if [ "${RESUME}" = "1" ]; then
  ARGS+=(--resume)
fi

if [ "${DISABLE_PARALLEL}" = "1" ]; then
  ARGS+=(--disable_parallel)
fi

if [ -n "${TOOLQA_DATA_DIR:-}" ]; then
  ARGS+=(--toolqa_data_dir "${TOOLQA_DATA_DIR}")
fi

if [ -n "${INSTANCES_DIR:-}" ]; then
  ARGS+=(--instances_dir "${INSTANCES_DIR}")
fi

if [ -n "${RETRIEVAL_RESULTS:-}" ]; then
  ARGS+=(--retrieval_results "${RETRIEVAL_RESULTS}" --top_k "${TOP_K}")
elif [ "${METHOD}" != "naive" ] && [ "${METHOD}" != "golden_skill" ]; then
  echo "METHOD=${METHOD} requires RETRIEVAL_RESULTS=/path/to/retrieval.json"
  exit 2
fi

python "${ARGS[@]}" "$@"
