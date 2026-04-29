# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from kvpress.presses.kvzap_compact_press import KVzapCompactPress


def test_kvzap_compact_press_masks_only_recent_window_when_threshold_is_high():
    press = KVzapCompactPress(threshold=1.0, sliding_window_size=3)
    scores = torch.zeros(1, 2, 6)

    valid = press._build_valid_mask(scores, k_len=6)

    expected = torch.tensor(
        [[
            [False, False, False, True, True, True],
            [False, False, False, True, True, True],
        ]]
    )
    assert torch.equal(valid, expected)


def test_kvzap_compact_press_matches_dms_threshold_rule_per_head():
    press = KVzapCompactPress(threshold=0.0, sliding_window_size=1)
    scores = torch.tensor([[[4.0, -1.0, -2.0, -3.0, 0.0], [4.0, 3.0, 2.0, -1.0, 0.0]]])

    valid = press._build_valid_mask(scores, k_len=5)

    expected = torch.tensor(
        [[
            [True, False, False, False, True],
            [True, True, True, False, True],
        ]]
    )
    assert torch.equal(valid, expected)


def test_kvzap_compact_press_thresholds_all_tokens_when_window_is_zero():
    press = KVzapCompactPress(threshold=0.0, sliding_window_size=0)
    scores = torch.tensor([[[1.0, -1.0, 2.0], [-2.0, 0.0, 3.0]]])

    valid = press._build_valid_mask(scores, k_len=3)

    expected = torch.tensor(
        [[
            [True, False, True],
            [False, True, True],
        ]]
    )
    assert torch.equal(valid, expected)


def test_kvzap_compact_press_dense_fallback_is_disabled():
    press = KVzapCompactPress()

    with pytest.raises(RuntimeError, match="no longer supports dense-tensor fallback compaction"):
        press.compress(
            module=object(),
            hidden_states=torch.empty(1, 1, 1),
            keys=torch.empty(1, 1, 1, 1),
            values=torch.empty(1, 1, 1, 1),
            attentions=None,
            kwargs={},
        )
