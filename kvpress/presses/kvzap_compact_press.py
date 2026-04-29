# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Literal, Optional

import torch
import torch.nn as nn

from kvpress.compacted_cache import CompactedDynamicCache
from kvpress.presses.base_press import BasePress
from kvpress.presses.kvzap_press import KVzapModel, prepare_kvzap_model_for_runtime, resolve_kvzap_model_name
from kvpress.utils import extract_keys_and_values


@dataclass
class KVzapCompactPress(BasePress):
    """
    Threshold-driven physical KV-cache eviction for KVzap.

    This press uses the same per-head thresholding rule as `DMSPress(KVzapPress(...))`
    during prefilling, but instead of keeping a dense cache and masking keys during
    attention it physically compacts the KV cache into the compacted runtime.
    """

    model_type: Literal["linear", "mlp"] = "mlp"
    threshold: float = -5.0
    sliding_window_size: int = 128
    requires_compacted_cache_runtime: bool = True
    kvzap_model_name: Optional[str] = field(default=None, init=False)
    compression_ratios: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    valid_masks: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _cache: Optional[object] = field(default=None, init=False, repr=False)

    def post_init_from_model(self, model):
        kvzap_model_name = resolve_kvzap_model_name(self.model_type, model)
        if kvzap_model_name != self.kvzap_model_name:
            self.kvzap_model_name = kvzap_model_name
            self.kvzap_model = KVzapModel.from_pretrained(self.kvzap_model_name)
        self.kvzap_model = prepare_kvzap_model_for_runtime(self.kvzap_model, model)

    @property
    def compression_ratio(self):
        assert len(self.compression_ratios) > 0, "Forward pass must be run to compute the compression ratio"
        return sum(self.compression_ratios.values()) / len(self.compression_ratios)

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")

    def _score_hidden_states(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        kvzap_module = self.kvzap_model.layers[module.layer_idx]
        with torch.no_grad():
            return kvzap_module(hidden_states).transpose(1, 2)

    def _build_valid_mask(self, scores: torch.Tensor, k_len: int) -> torch.Tensor:
        valid = torch.zeros_like(scores, dtype=torch.bool)
        protected = min(self.sliding_window_size, k_len)
        evict_end = k_len - protected

        if protected > 0:
            valid[..., evict_end:] = True
        if evict_end > 0:
            valid[..., :evict_end] = scores[..., :evict_end] >= self.threshold
        return valid

    @contextmanager
    def __call__(self, model) -> Generator:
        self.post_init_from_model(model)
        self.valid_masks.clear()
        self.compression_ratios.clear()
        self._cache = None

        hooks = []
        try:
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer in language_model.layers:
                layer.self_attn.rotary_emb = language_model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
            if isinstance(self._cache, CompactedDynamicCache) and self.valid_masks:
                masks = torch.stack([self.valid_masks[layer_idx] for layer_idx in sorted(self.valid_masks)], dim=0)
                self._cache.prune(masks)
        finally:
            for hook in hooks:
                hook.remove()

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        if not self._is_prefill_phase(kwargs, module.layer_idx):
            return output

        cache = kwargs.get("past_key_values", None) or kwargs.get("past_key_value", None)
        if cache is None:
            raise KeyError("Expected past_key_values or past_key_value in attention hook kwargs.")
        layer_idx = int(module.layer_idx)
        keys, values = extract_keys_and_values(cache, layer_idx)
        scores = self._score_hidden_states(module, kwargs["hidden_states"])

        if not isinstance(cache, CompactedDynamicCache):
            raise RuntimeError(
                "KVzapCompactPress requires the compacted KV-cache runtime. "
                "Use KVPressTextGenerationPipeline without overriding the cache."
            )

        valid = self._build_valid_mask(scores, keys.shape[2])
        self.valid_masks[layer_idx] = valid
        kept = float(valid.sum(dim=-1).float().mean().item())
        self.compression_ratios[layer_idx] = 1.0 - (kept / max(keys.shape[2], 1))
        self._cache = cache
        return output

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "KVzapCompactPress no longer supports dense-tensor fallback compaction. "
            "Use it with the compacted KV-cache runtime through KVPressTextGenerationPipeline."
        )
