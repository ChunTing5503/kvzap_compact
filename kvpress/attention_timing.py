# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

_ATTN_TIMING = {
    "attention_decode_ms": 0.0,
    "kv_prepare_decode_ms": 0.0,
    "pending": [],
}


def reset_attention_timing():
    _ATTN_TIMING["attention_decode_ms"] = 0.0
    _ATTN_TIMING["kv_prepare_decode_ms"] = 0.0
    _ATTN_TIMING["pending"] = []


def _flush_attention_timing():
    pending = _ATTN_TIMING["pending"]
    if not pending:
        return

    if torch.cuda.is_available():
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_idx)

    for key, start_event, end_event in pending:
        _ATTN_TIMING[key] += float(start_event.elapsed_time(end_event))
    pending.clear()


def get_attention_timing() -> dict[str, float]:
    _flush_attention_timing()
    return {
        "attention_decode_ms": float(_ATTN_TIMING["attention_decode_ms"]),
        "kv_prepare_decode_ms": float(_ATTN_TIMING["kv_prepare_decode_ms"]),
    }


def time_region(key: str, enabled: bool, fn: Callable[[], Any]) -> Any:
    if not enabled:
        return fn()

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = fn()
        end_event.record()
        _ATTN_TIMING["pending"].append((key, start_event, end_event))
        return output

    start_time = time.perf_counter()
    output = fn()
    _ATTN_TIMING[key] += (time.perf_counter() - start_time) * 1000.0
    return output
