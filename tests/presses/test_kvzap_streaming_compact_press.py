# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from kvpress.compacted_cache import StreamingCompactedDynamicCache
from kvpress.presses.kvzap_streaming_compact_press import KVzapStreamingCompactPress


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


class _TestKVzapStreamingCompactPress(KVzapStreamingCompactPress):
    def __init__(self, score_schedule, **kwargs):
        super().__init__(**kwargs)
        self.score_schedule = [score.clone() for score in score_schedule]

    def post_init_from_model(self, model):
        del model

    def _score_hidden_states(self, module, hidden_states):
        del module, hidden_states
        return self.score_schedule.pop(0)


def test_streaming_kvzap_press_finalizes_decode_tokens_in_blocks():
    press = _TestKVzapStreamingCompactPress(
        score_schedule=[
            torch.tensor([[[10.0, -1.0, 5.0, 4.0, 3.0], [10.0, 7.0, -2.0, 4.0, 3.0]]]),
            torch.tensor([[[-5.0], [6.0]]]),
            torch.tensor([[[-6.0], [-7.0]]]),
            torch.tensor([[[8.0], [9.0]]]),
            torch.tensor([[[10.0], [11.0]]]),
        ],
        threshold=0.0,
        sliding_window_size=2,
        decode_commit_interval=2,
    )
    cache = StreamingCompactedDynamicCache(_DummyModel(), commit_interval=2)
    module = SimpleNamespace(layer_idx=0, head_dim=1)

    prefill_keys = torch.arange(1 * 2 * 5 * 1, dtype=torch.float32).view(1, 2, 5, 1)
    prefill_values = prefill_keys + 100
    cache.update(prefill_keys, prefill_values, 0)
    press.forward_hook(
        module,
        [],
        {"hidden_states": torch.zeros(1, 5, 1), "past_key_values": cache},
        [None, None],
    )
    press.after_prefill()

    assert cache._head_lengths[0].tolist() == [2, 2]
    assert cache._tail_key_cache[0].shape[2] == 2
    assert torch.equal(press.scores_buffer[0], torch.tensor([[[4.0, 3.0], [4.0, 3.0]]]))

    decode_keys_1 = torch.arange(1 * 2 * 1 * 1, dtype=torch.float32).view(1, 2, 1, 1) + 200
    decode_values_1 = decode_keys_1 + 100
    cache.update(decode_keys_1, decode_values_1, 0)
    press.forward_hook(
        module,
        [],
        {"hidden_states": torch.zeros(1, 1, 1), "past_key_values": cache},
        [None, None],
    )

    assert cache._tail_matured_lengths[0] == 0
    assert torch.equal(press.scores_buffer[0], torch.tensor([[[4.0, 3.0, -5.0], [4.0, 3.0, 6.0]]]))

    decode_keys_2 = torch.arange(1 * 2 * 1 * 1, dtype=torch.float32).view(1, 2, 1, 1) + 300
    decode_values_2 = decode_keys_2 + 100
    cache.update(decode_keys_2, decode_values_2, 0)
    press.forward_hook(
        module,
        [],
        {"hidden_states": torch.zeros(1, 1, 1), "past_key_values": cache},
        [None, None],
    )

    assert cache._head_lengths[0].tolist() == [4, 4]
    assert cache._tail_key_cache[0].shape[2] == 2
    assert cache._tail_matured_lengths[0] == 0
    assert torch.equal(press.scores_buffer[0], torch.tensor([[[-5.0, -6.0], [6.0, -7.0]]]))

    decode_keys_3 = torch.arange(1 * 2 * 1 * 1, dtype=torch.float32).view(1, 2, 1, 1) + 400
    decode_values_3 = decode_keys_3 + 100
    cache.update(decode_keys_3, decode_values_3, 0)
    press.forward_hook(
        module,
        [],
        {"hidden_states": torch.zeros(1, 1, 1), "past_key_values": cache},
        [None, None],
    )

    assert cache._head_lengths[0].tolist() == [4, 4]
    assert cache._tail_matured_lengths[0] == 0
    assert torch.equal(press.scores_buffer[0], torch.tensor([[[-5.0, -6.0, 8.0], [6.0, -7.0, 9.0]]]))

    decode_keys_4 = torch.arange(1 * 2 * 1 * 1, dtype=torch.float32).view(1, 2, 1, 1) + 500
    decode_values_4 = decode_keys_4 + 100
    cache.update(decode_keys_4, decode_values_4, 0)
    press.forward_hook(
        module,
        [],
        {"hidden_states": torch.zeros(1, 1, 1), "past_key_values": cache},
        [None, None],
    )

    assert cache._head_lengths[0].tolist() == [4, 5]
    assert cache._tail_key_cache[0].shape[2] == 2
    assert cache._tail_matured_lengths[0] == 0
    assert torch.equal(press.scores_buffer[0], torch.tensor([[[8.0, 10.0], [9.0, 11.0]]]))


def test_streaming_kvzap_press_uses_cache_state_not_cache_position_for_decode_phase():
    press = _TestKVzapStreamingCompactPress(
        score_schedule=[
            torch.tensor([[[10.0, -1.0, 5.0, 4.0, 3.0], [10.0, 7.0, -2.0, 4.0, 3.0]]]),
            torch.tensor([[[-5.0], [6.0]]]),
        ],
        threshold=0.0,
        sliding_window_size=2,
        decode_commit_interval=2,
    )
    cache = StreamingCompactedDynamicCache(_DummyModel(), commit_interval=2)
    module = SimpleNamespace(layer_idx=0, head_dim=1)

    prefill_keys = torch.arange(1 * 2 * 5 * 1, dtype=torch.float32).view(1, 2, 5, 1)
    prefill_values = prefill_keys + 100
    cache.update(prefill_keys, prefill_values, 0)
    press.forward_hook(
        module,
        [],
        {"hidden_states": torch.zeros(1, 5, 1), "past_key_values": cache},
        [None, None],
    )
    press.after_prefill()

    decode_keys = torch.arange(1 * 2 * 1 * 1, dtype=torch.float32).view(1, 2, 1, 1) + 200
    decode_values = decode_keys + 100
    cache.update(decode_keys, decode_values, 0)
    press.forward_hook(
        module,
        [],
        {
            "hidden_states": torch.zeros(1, 1, 1),
            "past_key_values": cache,
            "cache_position": torch.tensor([0]),
        },
        [None, None],
    )

    assert cache._tail_matured_lengths[0] == 0


def test_streaming_kvzap_press_passes_rebuild_interval_to_cache():
    press = KVzapStreamingCompactPress(decode_rebuild_interval=23)
    cache = press.create_cache(_DummyModel())

    assert isinstance(cache, StreamingCompactedDynamicCache)
    assert cache.rebuild_interval == 23
