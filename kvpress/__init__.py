# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from kvpress.attention_patch import patch_attention_functions
from kvpress.compacted_attention import patch_compacted_attention
from kvpress.pipeline import KVPressTextGenerationPipeline
from kvpress.presses.base_press import SUPPORTED_MODELS, BasePress
from kvpress.presses.dms_press import DMSPress
from kvpress.presses.kvzap_compact_press import KVzapCompactPress
from kvpress.presses.kvzap_press import KVzapPress
from kvpress.presses.kvzap_streaming_compact_press import KVzapStreamingCompactPress
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.presses.scorer_press import ScorerPress

# Patch the attention functions to support head-wise compression
patch_attention_functions()
patch_compacted_attention()

__all__ = [
    "BasePress",
    "ScorerPress",
    "KVPressTextGenerationPipeline",
    "KVzipPress",
    "KVzapCompactPress",
    "KVzapPress",
    "KVzapStreamingCompactPress",
    "DMSPress",
    "SUPPORTED_MODELS",
]
