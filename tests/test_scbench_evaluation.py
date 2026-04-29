# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[1]
_EVALUATION_DIR = _REPO_ROOT / "evaluation"
if str(_EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALUATION_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scbench_load = _load_module("scbench_load", _EVALUATION_DIR / "benchmarks" / "scbench" / "load.py")
scbench_metrics = _load_module(
    "scbench_metrics",
    _EVALUATION_DIR / "benchmarks" / "scbench" / "calculate_metrics.py",
)
evaluate_registry = _load_module("evaluate_registry_module", _EVALUATION_DIR / "evaluate_registry.py")
evaluate_module = _load_module("evaluate_module", _EVALUATION_DIR / "evaluate.py")


def test_scbench_registry_entry_is_available():
    assert evaluate_registry.DATASET_REGISTRY["scbench_kv"] == "Jang-Hyun/SCBench-preprocessed"
    assert callable(evaluate_registry.SCORER_REGISTRY["scbench_kv"])


def test_load_scbench_dataframe_flattens_grouped_questions(monkeypatch):
    sample = {
        "prompts": ["shared context", "question 1", "question 2"],
        "ground_truth": ["alpha", ["beta", "gamma"]],
    }

    monkeypatch.setattr(scbench_load, "load_dataset", lambda *args, **kwargs: [sample])

    df = scbench_load.load_scbench_dataframe("scbench_kv")

    assert list(df.columns) == ["context", "question", "answer_prefix", "answer", "max_new_tokens"]
    assert df.to_dict("records") == [
        {
            "context": "shared context",
            "question": "question 1",
            "answer_prefix": "",
            "answer": "alpha",
            "max_new_tokens": 96,
        },
        {
            "context": "shared context",
            "question": "question 2",
            "answer_prefix": "",
            "answer": "beta, gamma",
            "max_new_tokens": 96,
        },
    ]


def test_scbench_kv_metrics_use_include_match():
    df = pd.DataFrame(
        [
            {"predicted_answer": "The value is Alpha.", "answer": "alpha"},
            {"predicted_answer": "wrong", "answer": "beta"},
        ]
    )

    metrics = scbench_metrics.calculate_metrics(df)

    assert metrics == {"accuracy": 0.5, "total": 2}


def test_evaluation_config_tracks_num_in_results_dir(tmp_path):
    config = evaluate_module.EvaluationConfig(
        dataset="scbench_kv",
        model="Qwen/Qwen3-8B",
        press_name="kvzap_mlp",
        compression_ratio=None,
        threshold=-5.0,
        num=5,
    )

    results_dir = config.get_results_dir(tmp_path)

    assert "num5" in results_dir.name


def test_streaming_kvzap_uses_decode_inference_path():
    press = evaluate_registry.PRESS_REGISTRY["kvzap_mlp_streaming_compact"]

    assert evaluate_module.EvaluationRunner._uses_decoding_inference(press)


def test_aime25_dataframe_is_normalized_to_question_only_prompt():
    raw = pd.DataFrame(
        [
            {
                "context": "very long prebuilt prompt that should be ignored",
                "question": "Solve x^2 = 1.",
                "answer": "1",
            }
        ]
    )

    normalized = evaluate_module._normalize_aime25_dataframe(raw)

    assert normalized.to_dict("records") == [
        {
            "context": "",
            "question": "Solve x^2 = 1.",
            "answer_prefix": "",
            "answer": "1",
            "max_new_tokens": 32000,
        }
    ]


def test_streaming_compact_aime_uses_exact_prompt_path(monkeypatch):
    runner = object.__new__(evaluate_module.EvaluationRunner)
    runner.config = SimpleNamespace(
        dataset="aime25",
        max_new_tokens=None,
        max_context_length=None,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )
    runner.df = pd.DataFrame(
        [
            {
                "context": "",
                "question": "Solve x^2 = 1.",
                "answer_prefix": "",
                "answer": "1",
                "max_new_tokens": 128,
            }
        ]
    )
    runner.call_profiles = []
    runner.call_runtimes = []
    runner.press = SimpleNamespace(compression_ratio=0.25)
    runner.pipeline = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda text, add_special_tokens=False: [1, 2, 3]),
    )

    monkeypatch.setattr(runner, "_uses_decoding_inference", lambda press: True)
    monkeypatch.setattr(runner, "_should_use_exact_aime25_prompt_path", lambda: True)
    monkeypatch.setattr(
        runner,
        "_run_exact_aime25_prompt_path",
        lambda question, max_new_tokens: {
            "answer": "boxed{1}",
            "profile": {
                "generated_tokens": 7,
                "peak_memory_bytes": 123,
                "prefill_s": 1.0,
                "decode_s": 2.0,
                "prefill_peak_memory_bytes": 11,
                "decode_peak_memory_bytes": 22,
                "prefill_input_tokens": 5,
                "post_prefill_memory_bytes": 33,
                "post_prefill_cache_payload_bytes": 44,
                "post_prefill_cache_length_min": 55,
                "post_prefill_cache_length_max": 66,
                "post_prefill_cache_length_mean": 77.0,
                "post_decode_cache_payload_bytes_mean": 88.0,
                "post_decode_cache_payload_bytes_max": 99,
                "post_decode_cache_length_min": 111.0,
                "post_decode_cache_length_max": 222.0,
                "post_decode_cache_length_mean": 333.0,
            },
        },
    )

    evaluate_module.EvaluationRunner._run_inference(runner)

    assert runner.df.loc[0, "predicted_answer"] == "boxed{1}"
    assert runner.df.loc[0, "generated_tokens"] == 7
    assert runner.df.loc[0, "peak_memory_bytes"] == 123
    assert runner.df.loc[0, "compression_ratio"] == 0.25
