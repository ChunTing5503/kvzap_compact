# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import wraps

from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from kvpress.attention_timing import time_region
from kvpress.compacted_cache import CompactedDynamicCache, _flash_attn_varlen_func


_ORIGINAL_FORWARDS: dict[type, callable] = {}


def _parse_attention_args(args, kwargs):
    position_embeddings = kwargs.get("position_embeddings")
    attention_mask = kwargs.get("attention_mask")
    past_key_values = kwargs.get("past_key_values")
    if past_key_values is None:
        past_key_values = kwargs.get("past_key_value")

    if position_embeddings is None and len(args) > 0:
        position_embeddings = args[0]
    if attention_mask is None and len(args) > 1:
        attention_mask = args[1]
    if past_key_values is None and len(args) > 2:
        past_key_values = args[2]

    return position_embeddings, attention_mask, past_key_values


def _compute_qkv(module, hidden_states):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    if hasattr(module, "q_norm"):
        query_states = module.q_norm(query_states)
    if hasattr(module, "k_norm"):
        key_states = module.k_norm(key_states)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    return query_states, key_states, value_states


def _compacted_forward(original_forward):
    @wraps(original_forward)
    def wrapper(self, hidden_states, *args, **kwargs):
        position_embeddings, _attention_mask, past_key_values = _parse_attention_args(args, kwargs)
        if not isinstance(past_key_values, CompactedDynamicCache) or not past_key_values.pruned:
            return original_forward(self, hidden_states, *args, **kwargs)

        if _flash_attn_varlen_func is None:
            raise RuntimeError("CompactedDynamicCache requires flash_attn to decode with flattened KV caches.")
        if position_embeddings is None:
            raise ValueError("position_embeddings are required for compacted-cache attention.")

        batch_size, query_length = hidden_states.shape[:2]
        query_states, key_states, value_states = _compute_qkv(self, hidden_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        measure_decode = query_length == 1
        query_states, key_states, value_states, info = time_region(
            "kv_prepare_decode_ms",
            measure_decode,
            lambda: past_key_values.prepare_varlen_attention(self.layer_idx, query_states, key_states, value_states),
        )
        attn_output = time_region(
            "attention_decode_ms",
            measure_decode,
            lambda: _flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=info["cu_len_q"],
                cu_seqlens_k=info["cu_len_k"],
                max_seqlen_q=info["max_len_q"],
                max_seqlen_k=info["max_len_k"],
                dropout_p=0.0 if not self.training else self.attention_dropout,
                causal=True,
            ),
        )

        attn_output = attn_output.view(
            batch_size,
            self.config.num_key_value_heads,
            query_length,
            self.num_key_value_groups,
            self.head_dim,
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, query_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    return wrapper


def patch_compacted_attention():
    for cls in (LlamaAttention, MistralAttention, Qwen2Attention, Qwen3Attention):
        if cls in _ORIGINAL_FORWARDS:
            continue
        _ORIGINAL_FORWARDS[cls] = cls.forward
        cls.forward = _compacted_forward(cls.forward)
