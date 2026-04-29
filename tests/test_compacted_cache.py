# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
import pytest

from kvpress.compacted_cache import CompactedDynamicCache, StreamingCompactedDynamicCache


class _DummyModel:
    def __init__(self):
        self.device = torch.device("cpu")
        config = SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        config.get_text_config = lambda decoder=True: config
        self.config = config


def test_compacted_cache_prune_update_and_slice_round_trip():
    cache = CompactedDynamicCache(_DummyModel())

    keys = torch.arange(1 * 2 * 5 * 4, dtype=torch.float32).view(1, 2, 5, 4)
    values = keys + 1000
    cache.update(keys, values, 0)

    valid_masks = torch.tensor([[[[True, False, True, False, True], [False, True, False, True, True]]]])
    cache.prune(valid_masks)

    assert cache.pruned is True
    assert cache.layers[0].keys.shape == (6, 4)
    assert cache.get_seq_length() == 5
    assert cache.get_storage_lengths() == [3.0]

    new_keys = torch.arange(1 * 2 * 1 * 4, dtype=torch.float32).view(1, 2, 1, 4) + 2000
    new_values = new_keys + 1000
    with pytest.raises(RuntimeError, match="requires tiny_api_cuda"):
        cache.update(new_keys, new_values, 0)

    cache.slice(5)
    assert cache.layers[0].keys.shape == (6, 4)
    assert cache.get_seq_length() == 5
    assert cache.get_storage_lengths() == [3.0]


def test_streaming_compacted_cache_splits_prefix_and_tail_then_commits_matured_tokens():
    cache = StreamingCompactedDynamicCache(_DummyModel(), commit_interval=2)

    keys = torch.arange(1 * 2 * 5 * 4, dtype=torch.float32).view(1, 2, 5, 4)
    values = keys + 1000
    cache.update(keys, values, 0)
    assert cache.get_seq_length() == 5

    valid_masks = torch.tensor([[[[True, False, True, True, True], [True, True, False, True, True]]]])
    cache.prune_with_tail(valid_masks, protected_tail_len=2)

    assert cache.pruned is True
    assert cache.layers[0].keys.shape == (4, 4)
    assert cache._tail_key_cache[0].shape == (1, 2, 2, 4)
    assert cache.get_seq_length() == 5
    assert cache.get_storage_lengths() == [4.0]

    new_keys_1 = torch.arange(1 * 2 * 1 * 4, dtype=torch.float32).view(1, 2, 1, 4) + 2000
    new_values_1 = new_keys_1 + 1000
    cache.update(new_keys_1, new_values_1, 0)
    assert cache.get_seq_length() == 6
    cache.mark_matured(0, torch.tensor([[[True], [True]]]))
    assert cache.should_commit(0) is False
    assert cache._tail_matured_lengths[0] == 1

    new_keys_2 = torch.arange(1 * 2 * 1 * 4, dtype=torch.float32).view(1, 2, 1, 4) + 3000
    new_values_2 = new_keys_2 + 1000
    cache.update(new_keys_2, new_values_2, 0)
    cache.mark_matured(0, torch.tensor([[[True], [False]]]))
    assert cache.should_commit(0) is True

    cache.commit_matured(0)

    assert cache._head_lengths[0].tolist() == [4, 3]
    assert cache._tail_key_cache[0].shape[2] == 2
    assert cache._tail_matured_lengths[0] == 0
    assert cache.get_storage_lengths() == [5.5]


def test_streaming_compacted_cache_snapshot_and_restore_round_trip():
    cache = StreamingCompactedDynamicCache(_DummyModel(), commit_interval=1)

    keys = torch.arange(1 * 2 * 4 * 4, dtype=torch.float32).view(1, 2, 4, 4)
    values = keys + 1000
    cache.update(keys, values, 0)
    valid_masks = torch.tensor([[[[True, False, True, True], [True, True, False, True]]]])
    cache.prune_with_tail(valid_masks, protected_tail_len=1)

    snapshot = cache.snapshot()

    new_keys = torch.arange(1 * 2 * 1 * 4, dtype=torch.float32).view(1, 2, 1, 4) + 2000
    new_values = new_keys + 1000
    cache.update(new_keys, new_values, 0)
    cache.mark_matured(0, torch.tensor([[[True], [False]]]))
    cache.commit_matured(0)

    cache.restore(snapshot)

    assert cache.layers[0].keys.shape == snapshot["key_cache"][0].shape
    assert torch.equal(cache._tail_key_cache[0], snapshot["tail_key_cache"][0])
    assert torch.equal(cache._tail_valid_masks[0], snapshot["tail_valid_masks"][0])
    assert cache.get_seq_length() == int(snapshot["logical_seq_length"])


def test_streaming_compacted_cache_defers_merged_view_rebuild_until_interval():
    cache = StreamingCompactedDynamicCache(_DummyModel(), commit_interval=8, rebuild_interval=2)

    keys = torch.arange(1 * 2 * 4 * 4, dtype=torch.float32).view(1, 2, 4, 4)
    values = keys + 1000
    cache.update(keys, values, 0)
    valid_masks = torch.tensor([[[[True, True, True, True], [True, True, True, True]]]])
    cache.prune_with_tail(valid_masks, protected_tail_len=2)

    cache.mark_matured(0, torch.tensor([[[False], [True]]]))
    assert cache._merged_dirty[0] is True

    query = torch.zeros(1, 4, 1, 4)
    key = torch.zeros(1, 2, 1, 4)
    value = torch.zeros(1, 2, 1, 4)

    _q, _k, _v, info_1 = cache.prepare_varlen_attention(0, query, key, value)
    _q, _k, _v, info_2 = cache.prepare_varlen_attention(0, query, key, value)
    _q, _k, _v, info_3 = cache.prepare_varlen_attention(0, query, key, value)

    assert info_1["cu_len_k"].tolist() == [0, 4, 8]
    assert info_2["cu_len_k"].tolist() == [0, 4, 8]
    assert info_3["cu_len_k"].tolist() == [0, 3, 7]
