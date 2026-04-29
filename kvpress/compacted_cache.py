# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import DynamicCache

try:
    from tiny_api_cuda import update_flatten_view as _update_flatten_view_cuda
except ImportError:
    _update_flatten_view_cuda = None

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except ImportError:
    _flash_attn_varlen_func = None


def compacted_runtime_is_available() -> bool:
    return _update_flatten_view_cuda is not None and _flash_attn_varlen_func is not None


def get_compacted_runtime_unavailable_reasons(model=None) -> list[str]:
    reasons = []
    if _update_flatten_view_cuda is None:
        reasons.append("tiny_api_cuda is not installed")
    if _flash_attn_varlen_func is None:
        reasons.append("flash_attn is not installed")
    if model is not None:
        if not torch.cuda.is_available():
            reasons.append("CUDA is not available")
        elif getattr(model, "device", None) is None or model.device.type != "cuda":
            reasons.append("the model is not running on CUDA")
        model_dtype = getattr(model, "dtype", None)
        if model_dtype is not None and model_dtype != torch.float16:
            reasons.append(
                f"the model dtype is {model_dtype}, but compacted KV-cache decoding currently requires torch.float16"
            )
    return reasons


class CompactedDynamicCache(DynamicCache):
    """
    Dynamic cache that can switch to flattened per-head storage after pruning.

    Before pruning it behaves like a regular DynamicCache. After pruning, each layer stores
    a compacted [num_retained_tokens_total, head_dim] view per key/value tensor and keeps the
    per-head sequence metadata needed for varlen FlashAttention during decoding.
    """

    def __init__(self, model):
        try:
            super().__init__(config=model.config)
        except TypeError:
            super().__init__()
        config = model.config.text_config if hasattr(model.config, "text_config") else model.config
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.device = model.device
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.logical_seq_length = 0
        self.pruned = False
        self._cu_head = torch.arange(self.num_key_value_heads + 1, dtype=torch.int32, device=self.device)
        self._flattened_layers: set[int] = set()
        self._head_lengths: dict[int, torch.Tensor] = {}
        self._cu_head_lengths: dict[int, torch.Tensor] = {}
        self._offsets: dict[int, int] = {}

    class _LayerView:
        def __init__(self, outer: "CompactedDynamicCache", layer_idx: int):
            self._outer = outer
            self._layer_idx = layer_idx

        @property
        def keys(self):
            return self._outer.key_cache[self._layer_idx]

        @keys.setter
        def keys(self, value):
            self._outer.key_cache[self._layer_idx] = value

        @property
        def values(self):
            return self._outer.value_cache[self._layer_idx]

        @values.setter
        def values(self, value):
            self._outer.value_cache[self._layer_idx] = value

        @property
        def kvpress_flattened(self):
            return self._layer_idx in self._outer._flattened_layers

        @kvpress_flattened.setter
        def kvpress_flattened(self, value):
            if value:
                self._outer._flattened_layers.add(self._layer_idx)
            else:
                self._outer._flattened_layers.discard(self._layer_idx)

        @property
        def kvpress_head_lengths(self):
            return self._outer._head_lengths[self._layer_idx]

        @kvpress_head_lengths.setter
        def kvpress_head_lengths(self, value):
            self._outer._head_lengths[self._layer_idx] = value

        @property
        def kvpress_cu_head_lengths(self):
            return self._outer._cu_head_lengths[self._layer_idx]

        @kvpress_cu_head_lengths.setter
        def kvpress_cu_head_lengths(self, value):
            self._outer._cu_head_lengths[self._layer_idx] = value

        @property
        def kvpress_offset(self):
            return self._outer._offsets[self._layer_idx]

        @kvpress_offset.setter
        def kvpress_offset(self, value):
            self._outer._offsets[self._layer_idx] = int(value)

        def get_seq_length(self):
            return self._outer.get_seq_length(self._layer_idx)

    @property
    def layers(self):
        return [self._LayerView(self, idx) for idx in range(len(self.key_cache))]

    @layers.setter
    def layers(self, value):
        # Newer transformers versions assign to ``self.layers`` during
        # DynamicCache initialization. We keep the assignment for compatibility
        # but expose our wrapper views from the getter.
        self._base_layers = value

    @staticmethod
    def is_supported_for_model(model) -> bool:
        return len(get_compacted_runtime_unavailable_reasons(model)) == 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs):
        if layer_idx == 0:
            self.logical_seq_length += int(key_states.shape[-2])

        if not self.pruned:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if layer_idx not in self._flattened_layers:
            raise RuntimeError("Flattened cache metadata is missing after compaction.")

        if key_states.shape[0] != 1:
            raise NotImplementedError("CompactedDynamicCache currently supports batch size 1 only.")

        keys = self.key_cache[layer_idx]
        values = self.value_cache[layer_idx]
        head_lengths = (self._head_lengths[layer_idx] + self._offsets[layer_idx]).to(dtype=torch.int32)
        flat_keys = key_states.squeeze(0).contiguous().view(-1, key_states.shape[-1])
        flat_values = value_states.squeeze(0).contiguous().view(-1, value_states.shape[-1])

        if not (
            keys.is_cuda
            and keys.dtype == torch.float16
            and flat_keys.dtype == torch.float16
            and _update_flatten_view_cuda is not None
        ):
            raise RuntimeError(
                "CompactedDynamicCache requires tiny_api_cuda and fp16 CUDA tensors for incremental updates."
            )

        self.key_cache[layer_idx] = _update_flatten_view_cuda(
            keys, flat_keys, head_lengths, self._cu_head_lengths[layer_idx]
        )
        self.value_cache[layer_idx] = _update_flatten_view_cuda(
            values, flat_values, head_lengths, self._cu_head_lengths[layer_idx]
        )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def prune(self, valid_masks: torch.Tensor):
        """
        Convert dense per-layer caches into flattened per-head compacted storage.

        Parameters
        ----------
        valid_masks : torch.Tensor
            Boolean tensor of shape [num_layers, batch, num_kv_heads, seq_len].
        """

        for layer_idx, (keys, values) in enumerate(zip(self.key_cache, self.value_cache)):
            valid = valid_masks[layer_idx].to(device=keys.device, dtype=torch.bool)

            if keys.shape[0] != 1:
                raise NotImplementedError("CompactedDynamicCache currently supports batch size 1 only.")

            head_lengths = valid.sum(dim=-1).squeeze(0).to(dtype=torch.int32)
            cu_head_lengths = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=keys.device),
                    head_lengths.cumsum(dim=0).to(dtype=torch.int32),
                ],
                dim=0,
            )

            self.key_cache[layer_idx] = keys.contiguous().view(-1, keys.shape[-1])[valid.view(-1)]
            self.value_cache[layer_idx] = values.contiguous().view(-1, values.shape[-1])[valid.view(-1)]
            self._flattened_layers.add(layer_idx)
            self._head_lengths[layer_idx] = head_lengths
            self._cu_head_lengths[layer_idx] = cu_head_lengths
            self._offsets[layer_idx] = 0

        self.pruned = True

    def prepare_varlen_attention(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | int]]:
        if query_states.shape[0] != 1:
            raise NotImplementedError("CompactedDynamicCache currently supports batch size 1 only.")

        _, _, q_len, head_dim = query_states.shape
        query_states = query_states.view(1, self.num_key_value_heads, self.num_key_value_groups, q_len, head_dim)
        query_states = query_states.transpose(2, 3).contiguous().view(-1, self.num_key_value_groups, head_dim)

        self._offsets[layer_idx] += q_len
        self._cu_head_lengths[layer_idx] = (
            self._cu_head_lengths[layer_idx] + (self._cu_head * q_len)
        ).to(dtype=torch.int32)

        info = {
            "cu_len_q": (self._cu_head * q_len).to(dtype=torch.int32),
            "cu_len_k": self._cu_head_lengths[layer_idx],
            "max_len_q": q_len,
            "max_len_k": int(self._head_lengths[layer_idx].max().item()) + int(self._offsets[layer_idx]),
        }
        return query_states, key_states.view(-1, 1, head_dim), value_states.view(-1, 1, head_dim), info

    def slice(self, logical_seq_length: int):
        """
        Remove the query/answer tokens appended after the compacted context for cache reuse.
        """

        for layer_idx, (keys, values) in enumerate(zip(self.key_cache, self.value_cache)):
            if layer_idx not in self._flattened_layers:
                self.key_cache[layer_idx] = keys[:, :, :logical_seq_length]
                self.value_cache[layer_idx] = values[:, :, :logical_seq_length]
                continue

            key_segments = []
            value_segments = []
            for head_idx in range(self.num_key_value_heads):
                start = int(self._cu_head_lengths[layer_idx][head_idx].item())
                end = start + int(self._head_lengths[layer_idx][head_idx].item())
                key_segments.append(keys[start:end])
                value_segments.append(values[start:end])

            self.key_cache[layer_idx] = torch.cat(key_segments, dim=0) if key_segments else keys[:0]
            self.value_cache[layer_idx] = torch.cat(value_segments, dim=0) if value_segments else values[:0]
            self._cu_head_lengths[layer_idx] = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=self.key_cache[layer_idx].device),
                    self._head_lengths[layer_idx].cumsum(dim=0).to(dtype=torch.int32),
                ],
                dim=0,
            )
            self._offsets[layer_idx] = 0

        self.logical_seq_length = logical_seq_length

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self.pruned:
            return int(self.logical_seq_length)
        if len(self.key_cache) <= layer_idx:
            return 0
        return int(self.key_cache[layer_idx].shape[2])

    def get_storage_lengths(self) -> list[float]:
        if not self.pruned:
            return [float(self.key_cache[layer_idx].shape[2]) for layer_idx in range(len(self.key_cache))]

        lengths = []
        for layer_idx in range(len(self.key_cache)):
            lengths.append(float((self._head_lengths[layer_idx].float() + self._offsets[layer_idx]).mean().item()))
        return lengths


class StreamingCompactedDynamicCache(CompactedDynamicCache):
    """
    Decode-capable compact cache with a committed prefix and a small live tail.

    The committed prefix stores physically compacted per-head KV pairs for tokens
    whose keep/drop decision is already final. The live tail stores the most
    recent undecided tokens densely, plus a bounded prefix of decided-but-not-yet-
    committed tokens. This mirrors KVzap+DMS semantics while keeping most of the
    cache physically compacted.
    """

    def __init__(self, model, commit_interval: int = 32, rebuild_interval: int = 16):
        super().__init__(model)
        self.commit_interval = int(commit_interval)
        self.rebuild_interval = max(int(rebuild_interval), 1)
        self._tail_key_cache: dict[int, torch.Tensor] = {}
        self._tail_value_cache: dict[int, torch.Tensor] = {}
        self._tail_valid_masks: dict[int, torch.Tensor] = {}
        self._tail_matured_lengths: dict[int, int] = {}
        self._tail_flat_key_cache: dict[int, torch.Tensor] = {}
        self._tail_flat_value_cache: dict[int, torch.Tensor] = {}
        self._tail_head_lengths: dict[int, torch.Tensor] = {}
        self._tail_cu_head_lengths: dict[int, torch.Tensor] = {}
        self._merged_key_cache: dict[int, torch.Tensor] = {}
        self._merged_value_cache: dict[int, torch.Tensor] = {}
        self._merged_head_lengths: dict[int, torch.Tensor] = {}
        self._merged_cu_head_lengths: dict[int, torch.Tensor] = {}
        self._merged_dirty: dict[int, bool] = {}
        self._decode_steps_since_rebuild: dict[int, int] = {}

    @staticmethod
    def _build_cu_head_lengths(head_lengths: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=head_lengths.device),
                head_lengths.cumsum(dim=0).to(dtype=torch.int32),
            ],
            dim=0,
        )

    def _set_merged_view(
        self,
        layer_idx: int,
        flat_keys: torch.Tensor,
        flat_values: torch.Tensor,
        head_lengths: torch.Tensor,
    ):
        head_lengths = head_lengths.to(dtype=torch.int32)
        self._merged_key_cache[layer_idx] = flat_keys
        self._merged_value_cache[layer_idx] = flat_values
        self._merged_head_lengths[layer_idx] = head_lengths
        self._merged_cu_head_lengths[layer_idx] = self._build_cu_head_lengths(head_lengths)
        self._merged_dirty[layer_idx] = False
        self._decode_steps_since_rebuild[layer_idx] = 0

    def _python_pack_valid_tokens(self, states: torch.Tensor, valid_mask: torch.Tensor):
        valid_mask = valid_mask.to(device=states.device, dtype=torch.bool)
        if states.shape[0] != 1:
            raise NotImplementedError("StreamingCompactedDynamicCache currently supports batch size 1 only.")
        head_lengths = valid_mask.sum(dim=-1).squeeze(0).to(dtype=torch.int32)
        cu_head_lengths = self._build_cu_head_lengths(head_lengths)
        flat_states = states.contiguous().view(-1, states.shape[-1])[valid_mask.reshape(-1)]
        return flat_states, head_lengths, cu_head_lengths

    def _set_tail_packed(
        self,
        layer_idx: int,
        flat_keys: torch.Tensor,
        flat_values: torch.Tensor,
        head_lengths: torch.Tensor,
    ):
        head_lengths = head_lengths.to(dtype=torch.int32)
        self._tail_flat_key_cache[layer_idx] = flat_keys
        self._tail_flat_value_cache[layer_idx] = flat_values
        self._tail_head_lengths[layer_idx] = head_lengths
        self._tail_cu_head_lengths[layer_idx] = self._build_cu_head_lengths(head_lengths)

    def _repack_tail(self, layer_idx: int):
        tail_keys = self._tail_key_cache[layer_idx]
        tail_values = self._tail_value_cache[layer_idx]
        tail_valid = self._tail_valid_masks[layer_idx]
        flat_keys, head_lengths, _ = self._python_pack_valid_tokens(tail_keys, tail_valid)
        flat_values, _, _ = self._python_pack_valid_tokens(tail_values, tail_valid)
        self._set_tail_packed(layer_idx, flat_keys, flat_values, head_lengths)

    def _append_to_tail_packed(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        flat_keys = key_states.squeeze(0).contiguous().view(-1, key_states.shape[-1])
        flat_values = value_states.squeeze(0).contiguous().view(-1, value_states.shape[-1])
        append_head_lengths = torch.ones(self.num_key_value_heads, dtype=torch.int32, device=flat_keys.device)
        if layer_idx not in self._tail_flat_key_cache:
            self._set_tail_packed(layer_idx, flat_keys, flat_values, append_head_lengths)
            return

        self._tail_flat_key_cache[layer_idx] = self._python_append_flattened_heads(
            self._tail_flat_key_cache[layer_idx],
            flat_keys,
            self._tail_head_lengths[layer_idx],
            self._tail_cu_head_lengths[layer_idx],
        )
        self._tail_flat_value_cache[layer_idx] = self._python_append_flattened_heads(
            self._tail_flat_value_cache[layer_idx],
            flat_values,
            self._tail_head_lengths[layer_idx],
            self._tail_cu_head_lengths[layer_idx],
        )
        self._set_tail_packed(
            layer_idx,
            self._tail_flat_key_cache[layer_idx],
            self._tail_flat_value_cache[layer_idx],
            self._tail_head_lengths[layer_idx] + append_head_lengths,
        )

    def _pop_tail_front(self, layer_idx: int, front_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        front_lengths = front_lengths.to(dtype=torch.int32)
        old_head_lengths = self._tail_head_lengths[layer_idx]
        old_cu_head_lengths = self._tail_cu_head_lengths[layer_idx]
        packed_keys = self._tail_flat_key_cache[layer_idx]
        packed_values = self._tail_flat_value_cache[layer_idx]

        front_key_segments = []
        front_value_segments = []
        remaining_key_segments = []
        remaining_value_segments = []
        remaining_head_lengths = []
        for head_idx in range(self.num_key_value_heads):
            segment_start = int(old_cu_head_lengths[head_idx].item())
            segment_end = segment_start + int(old_head_lengths[head_idx].item())
            take_len = int(front_lengths[head_idx].item())

            key_segment = packed_keys[segment_start:segment_end]
            value_segment = packed_values[segment_start:segment_end]
            front_key_segments.append(key_segment[:take_len])
            front_value_segments.append(value_segment[:take_len])
            remaining_key_segments.append(key_segment[take_len:])
            remaining_value_segments.append(value_segment[take_len:])
            remaining_head_lengths.append(key_segment.shape[0] - take_len)

        head_dim = packed_keys.shape[-1]
        front_flat_keys = (
            torch.cat(front_key_segments, dim=0) if front_key_segments else packed_keys.new_empty((0, head_dim))
        )
        front_flat_values = (
            torch.cat(front_value_segments, dim=0) if front_value_segments else packed_values.new_empty((0, head_dim))
        )
        remaining_flat_keys = (
            torch.cat(remaining_key_segments, dim=0) if remaining_key_segments else packed_keys.new_empty((0, head_dim))
        )
        remaining_flat_values = (
            torch.cat(remaining_value_segments, dim=0)
            if remaining_value_segments
            else packed_values.new_empty((0, head_dim))
        )
        self._set_tail_packed(
            layer_idx,
            remaining_flat_keys,
            remaining_flat_values,
            torch.tensor(remaining_head_lengths, dtype=torch.int32, device=packed_keys.device),
        )
        return front_flat_keys, front_flat_values

    def _rebuild_merged_view(self, layer_idx: int):
        prefix_keys = self.key_cache[layer_idx]
        prefix_values = self.value_cache[layer_idx]
        prefix_head_lengths = self._head_lengths[layer_idx]
        prefix_cu_head_lengths = self._cu_head_lengths[layer_idx]
        tail_keys = self._tail_flat_key_cache.get(layer_idx)
        tail_values = self._tail_flat_value_cache.get(layer_idx)
        tail_head_lengths = self._tail_head_lengths.get(layer_idx)
        tail_cu_head_lengths = self._tail_cu_head_lengths.get(layer_idx)
        head_dim = prefix_keys.shape[-1] if prefix_keys.numel() > 0 else self._tail_key_cache[layer_idx].shape[-1]

        key_segments = []
        value_segments = []
        merged_head_lengths = []
        for head_idx in range(self.num_key_value_heads):
            prefix_start = int(prefix_cu_head_lengths[head_idx].item())
            prefix_end = prefix_start + int(prefix_head_lengths[head_idx].item())
            prefix_key_segment = prefix_keys[prefix_start:prefix_end]
            prefix_value_segment = prefix_values[prefix_start:prefix_end]

            if (
                tail_keys is not None
                and tail_values is not None
                and tail_head_lengths is not None
                and tail_cu_head_lengths is not None
            ):
                tail_start = int(tail_cu_head_lengths[head_idx].item())
                tail_end = tail_start + int(tail_head_lengths[head_idx].item())
                tail_key_segment = tail_keys[tail_start:tail_end]
                tail_value_segment = tail_values[tail_start:tail_end]
            else:
                tail_key_segment = prefix_keys.new_empty((0, head_dim))
                tail_value_segment = prefix_values.new_empty((0, head_dim))

            key_segment = torch.cat([prefix_key_segment, tail_key_segment], dim=0)
            value_segment = torch.cat([prefix_value_segment, tail_value_segment], dim=0)
            key_segments.append(key_segment)
            value_segments.append(value_segment)
            merged_head_lengths.append(key_segment.shape[0])

        flat_keys = torch.cat(key_segments, dim=0) if key_segments else prefix_keys.new_empty((0, head_dim))
        flat_values = torch.cat(value_segments, dim=0) if value_segments else prefix_values.new_empty((0, head_dim))
        self._set_merged_view(
            layer_idx,
            flat_keys,
            flat_values,
            torch.tensor(merged_head_lengths, dtype=torch.int32, device=flat_keys.device),
        )

    def _python_append_flattened_heads(
        self,
        existing: torch.Tensor,
        new_flat: torch.Tensor,
        head_lengths: torch.Tensor,
        cu_head_lengths: torch.Tensor,
    ) -> torch.Tensor:
        head_dim = existing.shape[-1] if existing.numel() > 0 else new_flat.shape[-1]
        segments = []
        for head_idx in range(self.num_key_value_heads):
            start = int(cu_head_lengths[head_idx].item())
            end = start + int(head_lengths[head_idx].item())
            existing_segment = existing[start:end]
            new_segment = new_flat[head_idx : head_idx + 1]
            segments.append(torch.cat([existing_segment, new_segment], dim=0))
        return torch.cat(segments, dim=0) if segments else existing.new_empty((0, head_dim))

    def _python_append_packed_segments(
        self,
        cache: torch.Tensor,
        state: torch.Tensor,
        head_lengths: torch.Tensor,
        state_head_lengths: torch.Tensor,
        cu_head_lengths: torch.Tensor,
        state_cu_head_lengths: torch.Tensor,
    ) -> torch.Tensor:
        head_dim = cache.shape[-1] if cache.numel() > 0 else state.shape[-1]
        segments = []
        for head_idx in range(self.num_key_value_heads):
            cache_start = int(cu_head_lengths[head_idx].item())
            cache_end = cache_start + int(head_lengths[head_idx].item())
            state_start = int(state_cu_head_lengths[head_idx].item())
            state_end = state_start + int(state_head_lengths[head_idx].item())
            cache_segment = cache[cache_start:cache_end]
            state_segment = state[state_start:state_end]
            segments.append(torch.cat([cache_segment, state_segment], dim=0))
        return torch.cat(segments, dim=0) if segments else cache.new_empty((0, head_dim))

    def _append_to_merged_view(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        flat_keys = key_states.squeeze(0).contiguous().view(-1, key_states.shape[-1])
        flat_values = value_states.squeeze(0).contiguous().view(-1, value_states.shape[-1])
        if layer_idx not in self._merged_key_cache:
            head_lengths = torch.ones(self.num_key_value_heads, dtype=torch.int32, device=flat_keys.device)
            self._set_merged_view(layer_idx, flat_keys, flat_values, head_lengths)
            return

        merged_head_lengths = self._merged_head_lengths[layer_idx]
        merged_cu_head_lengths = self._merged_cu_head_lengths[layer_idx]
        merged_keys = self._merged_key_cache[layer_idx]
        merged_values = self._merged_value_cache[layer_idx]

        if (
            merged_keys.is_cuda
            and merged_keys.dtype == torch.float16
            and flat_keys.is_cuda
            and flat_keys.dtype == torch.float16
            and _update_flatten_view_cuda is not None
        ):
            self._merged_key_cache[layer_idx] = _update_flatten_view_cuda(
                merged_keys, flat_keys, merged_head_lengths, merged_cu_head_lengths
            )
            self._merged_value_cache[layer_idx] = _update_flatten_view_cuda(
                merged_values, flat_values, merged_head_lengths, merged_cu_head_lengths
            )
        else:
            self._merged_key_cache[layer_idx] = self._python_append_flattened_heads(
                merged_keys, flat_keys, merged_head_lengths, merged_cu_head_lengths
            )
            self._merged_value_cache[layer_idx] = self._python_append_flattened_heads(
                merged_values, flat_values, merged_head_lengths, merged_cu_head_lengths
            )

        new_head_lengths = merged_head_lengths + 1
        self._merged_head_lengths[layer_idx] = new_head_lengths
        self._merged_cu_head_lengths[layer_idx] = self._build_cu_head_lengths(new_head_lengths)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs):
        if not self.pruned:
            return CompactedDynamicCache.update(self, key_states, value_states, layer_idx, *args, **kwargs)

        if layer_idx == 0:
            self.logical_seq_length += int(key_states.shape[-2])

        if key_states.shape[0] != 1:
            raise NotImplementedError("StreamingCompactedDynamicCache currently supports batch size 1 only.")

        if layer_idx not in self._tail_key_cache:
            self._tail_key_cache[layer_idx] = key_states.contiguous()
            self._tail_value_cache[layer_idx] = value_states.contiguous()
            self._tail_valid_masks[layer_idx] = torch.ones_like(key_states[..., 0], dtype=torch.bool)
            self._tail_matured_lengths[layer_idx] = 0
            self._set_tail_packed(
                layer_idx,
                key_states.squeeze(0).contiguous().view(-1, key_states.shape[-1]),
                value_states.squeeze(0).contiguous().view(-1, value_states.shape[-1]),
                torch.ones(self.num_key_value_heads, dtype=torch.int32, device=key_states.device),
            )
        else:
            self._tail_key_cache[layer_idx] = torch.cat([self._tail_key_cache[layer_idx], key_states], dim=2)
            self._tail_value_cache[layer_idx] = torch.cat([self._tail_value_cache[layer_idx], value_states], dim=2)
            self._tail_valid_masks[layer_idx] = torch.cat(
                [
                    self._tail_valid_masks[layer_idx],
                    torch.ones_like(key_states[..., 0], dtype=torch.bool),
                ],
                dim=2,
            )
            self._append_to_tail_packed(layer_idx, key_states, value_states)

        if not self._merged_dirty.get(layer_idx, False):
            self._append_to_merged_view(layer_idx, key_states, value_states)
        return key_states, value_states

    def prune_with_tail(self, valid_masks: torch.Tensor, protected_tail_len: int):
        """
        Split dense prefill caches into a compacted committed prefix and a live tail.
        """

        for layer_idx, (keys, values) in enumerate(zip(self.key_cache, self.value_cache)):
            valid = valid_masks[layer_idx].to(device=keys.device, dtype=torch.bool)

            if keys.shape[0] != 1:
                raise NotImplementedError("StreamingCompactedDynamicCache currently supports batch size 1 only.")

            seq_len = keys.shape[2]
            tail_len = min(int(protected_tail_len), seq_len)
            prefix_len = seq_len - tail_len

            if prefix_len > 0:
                prefix_valid = valid[..., :prefix_len]
                head_lengths = prefix_valid.sum(dim=-1).squeeze(0).to(dtype=torch.int32)
                flat_keys = keys[:, :, :prefix_len].contiguous().reshape(-1, keys.shape[-1])[prefix_valid.reshape(-1)]
                flat_values = (
                    values[:, :, :prefix_len].contiguous().reshape(-1, values.shape[-1])[prefix_valid.reshape(-1)]
                )
            else:
                head_lengths = torch.zeros(self.num_key_value_heads, dtype=torch.int32, device=keys.device)
                flat_keys = keys.new_empty((0, keys.shape[-1]))
                flat_values = values.new_empty((0, values.shape[-1]))

            cu_head_lengths = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=keys.device),
                    head_lengths.cumsum(dim=0).to(dtype=torch.int32),
                ],
                dim=0,
            )

            self.key_cache[layer_idx] = flat_keys
            self.value_cache[layer_idx] = flat_values
            self._flattened_layers.add(layer_idx)
            self._head_lengths[layer_idx] = head_lengths
            self._cu_head_lengths[layer_idx] = cu_head_lengths
            self._offsets[layer_idx] = 0

            self._tail_key_cache[layer_idx] = keys[:, :, prefix_len:].contiguous()
            self._tail_value_cache[layer_idx] = values[:, :, prefix_len:].contiguous()
            self._tail_valid_masks[layer_idx] = valid[..., prefix_len:].contiguous()
            self._tail_matured_lengths[layer_idx] = 0
            self._repack_tail(layer_idx)
            self._rebuild_merged_view(layer_idx)

        self.pruned = True

    def mark_matured(self, layer_idx: int, keep_mask: torch.Tensor):
        """
        Mark the next block of oldest live-tail tokens as decided.
        """
        keep_mask = keep_mask.to(device=self.device, dtype=torch.bool)
        n_mature = int(keep_mask.shape[-1])
        if n_mature == 0:
            return

        matured_start = self._tail_matured_lengths[layer_idx]
        matured_end = matured_start + n_mature
        if matured_end > self._tail_valid_masks[layer_idx].shape[-1]:
            raise RuntimeError("Matured tail range exceeds the live tail length.")

        self._tail_valid_masks[layer_idx][..., matured_start:matured_end] = keep_mask
        self._tail_matured_lengths[layer_idx] = matured_end
        self._merged_dirty[layer_idx] = True

    def should_commit(self, layer_idx: int) -> bool:
        return self._tail_matured_lengths.get(layer_idx, 0) >= self.commit_interval

    def commit_matured(self, layer_idx: int):
        """
        Physically fold decided live-tail tokens into the compacted committed prefix.
        """
        matured_len = self._tail_matured_lengths.get(layer_idx, 0)
        if matured_len <= 0:
            return

        if self._merged_dirty.get(layer_idx, False):
            self._repack_tail(layer_idx)

        matured_head_lengths = (
            self._tail_valid_masks[layer_idx][..., :matured_len].sum(dim=-1).squeeze(0).to(dtype=torch.int32)
        )
        flat_matured_keys, flat_matured_values = self._pop_tail_front(layer_idx, matured_head_lengths)
        matured_cu_head_lengths = self._build_cu_head_lengths(matured_head_lengths)

        if flat_matured_keys.numel() > 0:
            self.key_cache[layer_idx] = self._python_append_packed_segments(
                self.key_cache[layer_idx],
                flat_matured_keys,
                self._head_lengths[layer_idx],
                matured_head_lengths,
                self._cu_head_lengths[layer_idx],
                matured_cu_head_lengths,
            )
            self.value_cache[layer_idx] = self._python_append_packed_segments(
                self.value_cache[layer_idx],
                flat_matured_values,
                self._head_lengths[layer_idx],
                matured_head_lengths,
                self._cu_head_lengths[layer_idx],
                matured_cu_head_lengths,
            )
            self._head_lengths[layer_idx] = self._head_lengths[layer_idx] + matured_head_lengths
            self._cu_head_lengths[layer_idx] = self._build_cu_head_lengths(self._head_lengths[layer_idx])

        tail_keys = self._tail_key_cache[layer_idx]
        tail_values = self._tail_value_cache[layer_idx]
        tail_valid = self._tail_valid_masks[layer_idx]
        self._tail_key_cache[layer_idx] = tail_keys[:, :, matured_len:].contiguous()
        self._tail_value_cache[layer_idx] = tail_values[:, :, matured_len:].contiguous()
        self._tail_valid_masks[layer_idx] = tail_valid[:, :, matured_len:].contiguous()
        self._tail_matured_lengths[layer_idx] = 0
        self._rebuild_merged_view(layer_idx)

    def commit_all_matured(self):
        for layer_idx in range(len(self.key_cache)):
            self.commit_matured(layer_idx)

    def prepare_varlen_attention(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | int]]:
        if query_states.shape[0] != 1:
            raise NotImplementedError("StreamingCompactedDynamicCache currently supports batch size 1 only.")

        _, _, q_len, head_dim = query_states.shape
        query_states = query_states.view(1, self.num_key_value_heads, self.num_key_value_groups, q_len, head_dim)
        query_states = query_states.transpose(2, 3).contiguous().view(-1, self.num_key_value_groups, head_dim)

        if layer_idx not in self._merged_key_cache:
            self._rebuild_merged_view(layer_idx)
        if self._merged_dirty.get(layer_idx, False) and (
            self._decode_steps_since_rebuild.get(layer_idx, 0) >= self.rebuild_interval
        ):
            self._rebuild_merged_view(layer_idx)

        flat_keys = self._merged_key_cache[layer_idx]
        flat_values = self._merged_value_cache[layer_idx]
        head_lengths_tensor = self._merged_head_lengths[layer_idx]
        cu_head_lengths = self._merged_cu_head_lengths[layer_idx]
        self._decode_steps_since_rebuild[layer_idx] = self._decode_steps_since_rebuild.get(layer_idx, 0) + int(q_len)

        info = {
            "cu_len_q": (self._cu_head * q_len).to(dtype=torch.int32),
            "cu_len_k": cu_head_lengths,
            "max_len_q": q_len,
            "max_len_k": int(head_lengths_tensor.max().item()) if head_lengths_tensor.numel() > 0 else 0,
        }
        return query_states, flat_keys.view(-1, 1, head_dim), flat_values.view(-1, 1, head_dim), info

    def snapshot(self):
        return {
            "key_cache": [tensor.clone() for tensor in self.key_cache],
            "value_cache": [tensor.clone() for tensor in self.value_cache],
            "flattened_layers": set(self._flattened_layers),
            "head_lengths": {idx: tensor.clone() for idx, tensor in self._head_lengths.items()},
            "cu_head_lengths": {idx: tensor.clone() for idx, tensor in self._cu_head_lengths.items()},
            "offsets": dict(self._offsets),
            "tail_key_cache": {idx: tensor.clone() for idx, tensor in self._tail_key_cache.items()},
            "tail_value_cache": {idx: tensor.clone() for idx, tensor in self._tail_value_cache.items()},
            "tail_valid_masks": {idx: tensor.clone() for idx, tensor in self._tail_valid_masks.items()},
            "tail_matured_lengths": dict(self._tail_matured_lengths),
            "tail_flat_key_cache": {idx: tensor.clone() for idx, tensor in self._tail_flat_key_cache.items()},
            "tail_flat_value_cache": {idx: tensor.clone() for idx, tensor in self._tail_flat_value_cache.items()},
            "tail_head_lengths": {idx: tensor.clone() for idx, tensor in self._tail_head_lengths.items()},
            "tail_cu_head_lengths": {idx: tensor.clone() for idx, tensor in self._tail_cu_head_lengths.items()},
            "merged_key_cache": {idx: tensor.clone() for idx, tensor in self._merged_key_cache.items()},
            "merged_value_cache": {idx: tensor.clone() for idx, tensor in self._merged_value_cache.items()},
            "merged_head_lengths": {idx: tensor.clone() for idx, tensor in self._merged_head_lengths.items()},
            "merged_cu_head_lengths": {idx: tensor.clone() for idx, tensor in self._merged_cu_head_lengths.items()},
            "merged_dirty": dict(self._merged_dirty),
            "decode_steps_since_rebuild": dict(self._decode_steps_since_rebuild),
            "logical_seq_length": int(self.logical_seq_length),
            "pruned": bool(self.pruned),
        }

    def restore(self, snapshot):
        self.key_cache = [tensor.clone() for tensor in snapshot["key_cache"]]
        self.value_cache = [tensor.clone() for tensor in snapshot["value_cache"]]
        self._flattened_layers = set(snapshot["flattened_layers"])
        self._head_lengths = {idx: tensor.clone() for idx, tensor in snapshot["head_lengths"].items()}
        self._cu_head_lengths = {idx: tensor.clone() for idx, tensor in snapshot["cu_head_lengths"].items()}
        self._offsets = dict(snapshot["offsets"])
        self._tail_key_cache = {idx: tensor.clone() for idx, tensor in snapshot["tail_key_cache"].items()}
        self._tail_value_cache = {idx: tensor.clone() for idx, tensor in snapshot["tail_value_cache"].items()}
        self._tail_valid_masks = {idx: tensor.clone() for idx, tensor in snapshot["tail_valid_masks"].items()}
        self._tail_matured_lengths = dict(snapshot["tail_matured_lengths"])
        self._tail_flat_key_cache = {
            idx: tensor.clone() for idx, tensor in snapshot.get("tail_flat_key_cache", {}).items()
        }
        self._tail_flat_value_cache = {
            idx: tensor.clone() for idx, tensor in snapshot.get("tail_flat_value_cache", {}).items()
        }
        self._tail_head_lengths = {idx: tensor.clone() for idx, tensor in snapshot.get("tail_head_lengths", {}).items()}
        self._tail_cu_head_lengths = {
            idx: tensor.clone() for idx, tensor in snapshot.get("tail_cu_head_lengths", {}).items()
        }
        self._merged_key_cache = {idx: tensor.clone() for idx, tensor in snapshot.get("merged_key_cache", {}).items()}
        self._merged_value_cache = {
            idx: tensor.clone() for idx, tensor in snapshot.get("merged_value_cache", {}).items()
        }
        self._merged_head_lengths = {
            idx: tensor.clone() for idx, tensor in snapshot.get("merged_head_lengths", {}).items()
        }
        self._merged_cu_head_lengths = {
            idx: tensor.clone() for idx, tensor in snapshot.get("merged_cu_head_lengths", {}).items()
        }
        self._merged_dirty = dict(snapshot.get("merged_dirty", {}))
        self._decode_steps_since_rebuild = dict(snapshot.get("decode_steps_since_rebuild", {}))
        self.logical_seq_length = int(snapshot["logical_seq_length"])
        self.pruned = bool(snapshot["pruned"])
        if self.pruned:
            for layer_idx in range(len(self.key_cache)):
                if layer_idx not in self._tail_flat_key_cache:
                    self._repack_tail(layer_idx)
                if layer_idx not in self._merged_key_cache:
                    self._rebuild_merged_view(layer_idx)

    def slice(self, logical_seq_length: int):
        # Streaming decode mutates both the compacted prefix and live tail. For
        # context reuse callers should prefer snapshot/restore over slice().
        self.logical_seq_length = int(logical_seq_length)

    def get_storage_lengths(self) -> list[float]:
        if not self.pruned:
            return super().get_storage_lengths()

        lengths = []
        for layer_idx in range(len(self.key_cache)):
            prefix_lengths = self._head_lengths[layer_idx].float()
            tail_lengths = self._tail_valid_masks[layer_idx].sum(dim=-1).squeeze(0).float()
            lengths.append(float((prefix_lengths + tail_lengths).mean().item()))
        return lengths
