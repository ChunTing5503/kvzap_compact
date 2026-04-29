# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import torch

from kvpress.presses.kvzip_press import KVzipPress


def test_kvzip_press_does_not_warn_on_init(caplog):
    with caplog.at_level(logging.WARNING):
        KVzipPress(compression_ratio=0.5, layerwise=False)

    assert not any("multiple forward passes for chunked context reconstruction" in record.message for record in caplog.records)


def test_kvzip_press_builds_exact_pairwise_valid_masks():
    press = KVzipPress(compression_ratio=0.5, layerwise=False)
    press.score_val = torch.tensor(
        [
            [
                [[10.0, 1.0, 9.0, 0.5], [8.0, 7.0, 0.1, 0.2]],
            ]
        ]
    )

    valid_masks = press._build_valid_masks()

    assert valid_masks.shape == press.score_val.shape
    assert int(valid_masks.sum().item()) == 4
    assert torch.equal(
        valid_masks,
        torch.tensor(
            [
                [
                    [[True, False, True, False], [True, True, False, False]],
                ]
            ]
        ),
    )
