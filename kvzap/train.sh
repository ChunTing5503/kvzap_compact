#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-8B"}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT_DIR}/kvzap_runs/qwen3_8b_run1"}
PYTHON_BIN=${PYTHON_BIN:-}

cd "${ROOT_DIR}/kvzap"

if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python
fi

"${PYTHON_BIN}" -u train.py --model_name "${MODEL_NAME}" --output_dir "${OUTPUT_DIR}"
