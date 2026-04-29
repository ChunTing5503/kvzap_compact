#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH -J kvzap-eval
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH --mail-type=ALL

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_FILE=${CONFIG_FILE:-"${ROOT_DIR}/evaluation/configs/eval_scbench_kv_qwen3_8b_kvzap_mlp_streaming_compact.yaml"}
DEVICE=${DEVICE:-cuda:0}
THRESHOLDS=${THRESHOLDS:-}
PYTHON_BIN=${PYTHON_BIN:-}

cd "${ROOT_DIR}"

if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "DEVICE=${DEVICE}"
echo "THRESHOLDS=${THRESHOLDS:-<none>}"
echo "PYTHON_BIN=${PYTHON_BIN}"
nvidia-smi -L || true

if [[ -n "${THRESHOLDS}" ]]; then
  for threshold in ${THRESHOLDS}; do
    echo "Running threshold sweep entry: threshold=${threshold}"
    "${PYTHON_BIN}" -u evaluation/evaluate.py \
      --config_file "${CONFIG_FILE}" \
      --device "${DEVICE}" \
      --threshold "${threshold}" \
      "$@"
  done
else
  "${PYTHON_BIN}" -u evaluation/evaluate.py \
    --config_file "${CONFIG_FILE}" \
    --device "${DEVICE}" \
    "$@"
fi
