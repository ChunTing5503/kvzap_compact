# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset


SCBENCH_DATASET_ID = "Jang-Hyun/SCBench-preprocessed"


def is_scbench_dataset(dataset_name: str) -> bool:
    return dataset_name.startswith("scbench_")


def get_scbench_max_new_tokens(dataset_name: str) -> int:
    if "_mf" in dataset_name:
        return 32
    if "summary" in dataset_name:
        return 256
    if "repoqa" in dataset_name:
        return 512
    return 96


def _normalize_scbench_answer(answer) -> str:
    if isinstance(answer, list):
        return ", ".join(str(item) for item in answer)
    if answer is None:
        return ""
    return str(answer)


def _candidate_hf_roots() -> list[Path]:
    roots: list[Path] = []

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home))

    hfh = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hfh:
        roots.append(Path(hfh).parent)

    roots.extend(
        [
            Path("/scratch/cl5503/huggingface"),
            Path("/scratch/cl5503/.cache/huggingface"),
            Path.home() / ".cache" / "huggingface",
        ]
    )

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            unique_roots.append(root)
            seen.add(root)
    return unique_roots


def _resolve_local_scbench_parquet(dataset_name: str) -> Path | None:
    relative = Path("hub") / "datasets--Jang-Hyun--SCBench-preprocessed" / "snapshots"
    parquet_name = f"{dataset_name}.parquet"
    for root in _candidate_hf_roots():
        snapshot_root = root / relative
        if not snapshot_root.exists():
            continue
        matches = sorted(snapshot_root.glob(f"*/{parquet_name}"))
        if matches:
            return matches[-1]
    return None


def _load_scbench_samples(dataset_name: str):
    local_parquet = _resolve_local_scbench_parquet(dataset_name)
    if local_parquet is not None:
        return load_dataset("parquet", data_files=str(local_parquet), split="train")

    try:
        return load_dataset(SCBENCH_DATASET_ID, data_files=f"{dataset_name}.parquet", split="train")
    except Exception as exc:
        raise ValueError(
            f"Could not load {dataset_name}. No local parquet snapshot was found, and loading via datasets also failed."
        ) from exc


def load_scbench_dataframe(dataset_name: str) -> pd.DataFrame:
    samples = _load_scbench_samples(dataset_name)
    max_new_tokens = get_scbench_max_new_tokens(dataset_name)
    rows: list[dict[str, str | int]] = []

    for sample in samples:
        prompts = list(sample["prompts"])
        answers = list(sample["ground_truth"])
        context = str(prompts[0])

        for question, answer in zip(prompts[1:], answers):
            rows.append(
                {
                    "context": context,
                    "question": str(question),
                    "answer_prefix": "",
                    "answer": _normalize_scbench_answer(answer),
                    "max_new_tokens": max_new_tokens,
                }
            )

    return pd.DataFrame(rows)
