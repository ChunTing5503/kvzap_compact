# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from transformers import DynamicCache

from kvpress.pipeline import KVPressTextGenerationPipeline
from kvpress.presses.base_press import BasePress


class _SingleQuestionDecodePress(BasePress):
    decoding = True
    supports_multiple_questions = False


class _StreamingMultiQuestionPress(BasePress):
    decoding = True
    supports_multiple_questions = True
    requires_continuous_context = True


def test_pipeline_multi_question_support_guard_distinguishes_decode_paths():
    assert not KVPressTextGenerationPipeline._supports_multiple_questions(_SingleQuestionDecodePress())
    assert KVPressTextGenerationPipeline._supports_multiple_questions(_StreamingMultiQuestionPress())


def test_generate_answer_supports_sampling_with_top_k_1():
    class _DummyModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.generation_config = SimpleNamespace(eos_token_id=3)
            self.calls = 0

        def __call__(self, input_ids, past_key_values, position_ids, num_logits_to_keep=None):
            del input_ids, past_key_values, position_ids, num_logits_to_keep
            self.calls += 1
            logits = torch.full((1, 1, 4), -10.0, dtype=torch.float32)
            if self.calls == 1:
                logits[0, -1, 2] = 10.0
            else:
                logits[0, -1, 3] = 10.0
            return SimpleNamespace(logits=logits)

    class _DummyTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            del skip_special_tokens
            return ",".join(str(int(token_id)) for token_id in token_ids)

    pipeline = object.__new__(KVPressTextGenerationPipeline)
    pipeline.model = _DummyModel()
    pipeline.tokenizer = _DummyTokenizer()

    answer = pipeline.generate_answer(
        question_ids=torch.tensor([[11]]),
        cache=DynamicCache(),
        context_length=20,
        max_new_tokens=3,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=1,
        min_p=0.0,
    )

    assert answer == "2,3"


def test_pipeline_skips_cache_snapshot_for_single_question():
    class _DummyInnerModel:
        def __call__(self, input_ids, past_key_values):
            past_key_values.seq_length = int(input_ids.shape[1])

    class _DummyModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.model = _DummyInnerModel()

    class _SnapshotCache:
        def __init__(self):
            self.seq_length = 0
            self.snapshot_calls = 0
            self.restore_calls = 0
            self.slice_calls = 0

        def __len__(self):
            return 1

        def get_seq_length(self, layer_idx=0):
            del layer_idx
            return self.seq_length

        def snapshot(self):
            self.snapshot_calls += 1
            return {"seq_length": self.seq_length}

        def restore(self, snapshot):
            self.restore_calls += 1
            self.seq_length = int(snapshot["seq_length"])

        def slice(self, logical_seq_length):
            self.slice_calls += 1
            self.seq_length = int(logical_seq_length)

    pipeline = object.__new__(KVPressTextGenerationPipeline)
    pipeline.model = _DummyModel()

    def _generate_answer(question_ids, cache, context_length, max_new_tokens):
        del context_length, max_new_tokens
        cache.seq_length += int(question_ids.shape[1]) + 1
        return "answer"

    pipeline.generate_answer = _generate_answer
    cache = _SnapshotCache()

    answers = pipeline._forward(
        {
            "context_ids": torch.tensor([[1, 2, 3]]),
            "questions_ids": [torch.tensor([[4, 5]])],
        },
        max_new_tokens=1,
        cache=cache,
    )

    assert answers == ["answer"]
    assert cache.snapshot_calls == 0
    assert cache.restore_calls == 0
    assert cache.slice_calls == 1


def test_pipeline_estimates_logical_cache_payload_bytes():
    class _DummyModel:
        def __init__(self):
            self.dtype = torch.float16
            self.config = SimpleNamespace(
                hidden_size=8,
                num_attention_heads=2,
                num_key_value_heads=1,
            )

    class _LengthCache:
        def get_storage_lengths(self):
            return [3.0, 5.0]

    pipeline = object.__new__(KVPressTextGenerationPipeline)
    pipeline.model = _DummyModel()

    assert pipeline._estimate_cache_payload_bytes(_LengthCache()) == 128
