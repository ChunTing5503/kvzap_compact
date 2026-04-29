#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sweep KVzap thresholds on AIME25 using the sampled prefill/decode profiler in
# kvzap/evaluate_aime.py. KVzap uses a score threshold rather than a fixed input
# compression ratio, so the achieved compression ratio is measured from outputs.

#SBATCH -J kvzap-aime-sweep
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
KVZAP_DIR="${ROOT_DIR}/kvzap"
PYTHON_BIN=${PYTHON_BIN:-}

# --- editable run configuration ---
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-8B"}
DEVICE=${DEVICE:-"cuda:0"}
MODEL_TYPES=(${MODEL_TYPES:-"mlp"})
THRESHOLDS=(${THRESHOLDS:-"-3 -4 -5 -6"})
INCLUDE_BASELINE=${INCLUDE_BASELINE:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
NUM=${NUM:-}

cd "${ROOT_DIR}" || exit 1

if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python
fi

lscpu
nvidia-smi

COMMON_ARGS=(
  --model_name "${MODEL_NAME}"
  --device "${DEVICE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
)

if [[ -n "${NUM}" ]]; then
  COMMON_ARGS+=(--num "${NUM}")
fi

echo "Root dir: ${ROOT_DIR}"
echo "Model: ${MODEL_NAME}"
echo "Device: ${DEVICE}"
echo "Python: ${PYTHON_BIN}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Model types: ${MODEL_TYPES[*]}"
echo "Thresholds: ${THRESHOLDS[*]}"
if [[ -n "${NUM}" ]]; then
  echo "Num examples: ${NUM}"
fi

cd "${KVZAP_DIR}" || exit 1

if [[ "${INCLUDE_BASELINE}" == "1" ]]; then
  echo "Running no_press baseline once before threshold sweep"
  "${PYTHON_BIN}" -u evaluate_aime.py no_press "${COMMON_ARGS[@]}"
fi

for model_type in "${MODEL_TYPES[@]}"; do
  for threshold in "${THRESHOLDS[@]}"; do
    echo "Running AIME25 sweep: model_type=${model_type}, threshold=${threshold}"
    "${PYTHON_BIN}" -u evaluate_aime.py "${model_type}" \
      --threshold "${threshold}" \
      "${COMMON_ARGS[@]}"
  done
done

echo "AIME25 threshold sweep completed."
