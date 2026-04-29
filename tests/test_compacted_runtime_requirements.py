# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
import pytest

from kvpress.pipeline import KVPressTextGenerationPipeline
from kvpress.compacted_cache import get_compacted_runtime_unavailable_reasons
from kvpress.presses.kvzap_compact_press import KVzapCompactPress
from kvpress.presses.kvzap_streaming_compact_press import KVzapStreamingCompactPress


def test_pipeline_rejects_compacted_runtime_press_when_runtime_is_unavailable():
    pipeline = object.__new__(KVPressTextGenerationPipeline)
    pipeline.model = SimpleNamespace(device=torch.device("cpu"))

    with pytest.raises(RuntimeError, match="requires the compacted KV-cache runtime"):
        pipeline._create_default_cache(KVzapCompactPress())


def test_pipeline_rejects_streaming_compacted_runtime_press_when_runtime_is_unavailable():
    pipeline = object.__new__(KVPressTextGenerationPipeline)
    pipeline.model = SimpleNamespace(device=torch.device("cpu"))

    with pytest.raises(RuntimeError, match="requires the compacted KV-cache runtime"):
        pipeline._create_default_cache(KVzapStreamingCompactPress())


def test_compacted_runtime_reports_non_fp16_model_requirement():
    model = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16)

    reasons = get_compacted_runtime_unavailable_reasons(model)

    assert any("torch.float16" in reason for reason in reasons)
