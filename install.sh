#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT_DIR}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate your conda environment first, then rerun this script."
  echo "Example: conda activate /scratch/cl5503/conda_envs/kvpress-compact"
  exit 1
fi

python -m pip install -e ".[eval,train]"

cd "${ROOT_DIR}/_upstream/KVzip/csrc"
python build.py install
