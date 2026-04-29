# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from benchmarks.aime25.calculate_metrics import calculate_metrics as aime25_scorer
    from benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer
    from benchmarks.scbench.calculate_metrics import calculate_metrics as scbench_scorer
except ImportError:
    from evaluation.benchmarks.aime25.calculate_metrics import calculate_metrics as aime25_scorer
    from evaluation.benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer
    from evaluation.benchmarks.scbench.calculate_metrics import calculate_metrics as scbench_scorer

from kvpress import DMSPress, KVzapCompactPress, KVzapPress, KVzapStreamingCompactPress, KVzipPress

DATASET_REGISTRY = {
    "aime25": "alessiodevoto/aime25",
    "ruler": "simonjegou/ruler",
    "scbench_kv": "Jang-Hyun/SCBench-preprocessed",
}

SCORER_REGISTRY = {
    "aime25": aime25_scorer,
    "ruler": ruler_scorer,
    "scbench_kv": scbench_scorer,
}


PRESS_REGISTRY = {
    "no_press": None,
    "kvzip": KVzipPress(),
    # Official KVzap baseline: thresholded logical masking, no physical compaction.
    "kvzap_mlp": DMSPress(press=KVzapPress(model_type="mlp"), decoding=True),
    # Optional ablation: compact the prefilled cache but keep standard decoding.
    "kvzap_mlp_compact": KVzapCompactPress(model_type="mlp"),
    # Main project contribution: physical KV-cache compaction during streaming decode.
    "kvzap_mlp_streaming_compact": KVzapStreamingCompactPress(model_type="mlp"),
}
