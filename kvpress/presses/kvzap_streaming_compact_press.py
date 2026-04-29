# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Literal, Optional

import torch
import torch.nn as nn

from kvpress.compacted_cache import StreamingCompactedDynamicCache
from kvpress.presses.base_press import BasePress
from kvpress.presses.kvzap_press import KVzapModel, prepare_kvzap_model_for_runtime, resolve_kvzap_model_name


@dataclass
class KVzapStreamingCompactPress(BasePress):
    """
    Exact-online KVzap compaction following the original DMS-style decode logic.

    Tokens are scored from hidden states, protected while they remain inside the
    sliding window, and then irrevocably kept or dropped in fixed-size decode
    blocks once an entire protected block has matured. Older kept tokens are
    physically committed into compacted per-head storage; recent tokens remain
    in a bounded dense live tail.
    """

    model_type: Literal["linear", "mlp"] = "mlp"
    threshold: float = -5.0
    sliding_window_size: int = 128
    decode_commit_interval: int = 32
    decode_rebuild_interval: int = 16
    decoding: bool = True
    supports_multiple_questions: bool = True
    requires_compacted_cache_runtime: bool = True
    requires_continuous_context: bool = True
    kvzap_model_name: Optional[str] = field(default=None, init=False)
    compression_ratios: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    scores_buffer: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    valid_masks: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _cache: Optional[StreamingCompactedDynamicCache] = field(default=None, init=False, repr=False)

    def post_init_from_model(self, model):
        kvzap_model_name = resolve_kvzap_model_name(self.model_type, model)
        if kvzap_model_name != self.kvzap_model_name:
            self.kvzap_model_name = kvzap_model_name
            self.kvzap_model = KVzapModel.from_pretrained(self.kvzap_model_name)
        self.kvzap_model = prepare_kvzap_model_for_runtime(self.kvzap_model, model)

    def create_cache(self, model):
        commit_interval = self._maturation_block_size()
        return StreamingCompactedDynamicCache(
            model,
            commit_interval=commit_interval,
            rebuild_interval=self.decode_rebuild_interval,
        )

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

    def _update_compression_ratio(self, cache: StreamingCompactedDynamicCache, layer_idx: int):
        storage_length = cache.get_storage_lengths()[layer_idx]
        self.compression_ratios[layer_idx] = 1.0 - (storage_length / max(cache.get_seq_length(layer_idx), 1))

    def _maturation_block_size(self) -> int:
        return max(int(self.sliding_window_size), 1)

    def after_prefill(self):
        if self._cache is not None and not self._cache.pruned and self.valid_masks:
            masks = torch.stack([self.valid_masks[layer_idx] for layer_idx in sorted(self.valid_masks)], dim=0)
            self._cache.prune_with_tail(masks, protected_tail_len=self.sliding_window_size)
            for layer_idx in sorted(self.valid_masks):
                self._update_compression_ratio(self._cache, layer_idx)

    def snapshot_state(self):
        return {
            "scores_buffer": {idx: scores.clone() for idx, scores in self.scores_buffer.items()},
            "compression_ratios": dict(self.compression_ratios),
        }

    def restore_state(self, snapshot):
        self.scores_buffer = {idx: scores.clone() for idx, scores in snapshot["scores_buffer"].items()}
        self.compression_ratios = dict(snapshot["compression_ratios"])

    @contextmanager
    def __call__(self, model) -> Generator:
        self.post_init_from_model(model)
        self.valid_masks.clear()
        self.compression_ratios.clear()
        self.scores_buffer.clear()
        self._cache = None

        hooks = []
        try:
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer in language_model.layers:
                layer.self_attn.rotary_emb = language_model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for hook in hooks:
                hook.remove()

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        cache = kwargs.get("past_key_values", None) or kwargs.get("past_key_value", None)
        if cache is None:
            raise KeyError("Expected past_key_values or past_key_value in attention hook kwargs.")
        if not isinstance(cache, StreamingCompactedDynamicCache):
            raise RuntimeError(
                "KVzapStreamingCompactPress requires the streaming compacted KV-cache runtime. "
                "Use KVPressTextGenerationPipeline without overriding the cache."
            )

        layer_idx = int(module.layer_idx)
        self._cache = cache

        scores = self._score_hidden_states(module, kwargs["hidden_states"])

        # For the streaming compact runtime, cache layout changes after prefill.
        # ``cache.pruned`` is the authoritative phase boundary:
        # dense cache before ``after_prefill()``, compacted prefix + live tail after.
        if not cache.pruned:
            cache_len = cache.get_seq_length(layer_idx)
            valid = self._build_valid_mask(scores, cache_len)
            self.valid_masks[layer_idx] = valid
            protected = min(self.sliding_window_size, scores.shape[-1])
            if protected > 0:
                self.scores_buffer[layer_idx] = scores[..., -protected:].detach().clone()
            else:
                self.scores_buffer[layer_idx] = scores[..., :0].detach().clone()
            kept = float(valid.sum(dim=-1).float().mean().item())
            self.compression_ratios[layer_idx] = 1.0 - (kept / max(cache_len, 1))
            return output

        previous_scores = self.scores_buffer.get(layer_idx)
        if previous_scores is None:
            previous_scores = scores[..., :0].detach().clone()
        scores_buffer = torch.cat([previous_scores, scores.detach()], dim=-1)

        block_size = self._maturation_block_size()
        while scores_buffer.shape[-1] >= (2 * block_size):
            keep_mask = scores_buffer[..., :block_size] >= self.threshold
            cache.mark_matured(layer_idx, keep_mask)
            cache.commit_matured(layer_idx)
            scores_buffer = scores_buffer[..., block_size:]

        self.scores_buffer[layer_idx] = scores_buffer
        self._update_compression_ratio(cache, layer_idx)
        return output
