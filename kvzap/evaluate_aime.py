# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import DMSPress, KVzapPress
from kvpress.attention_timing import get_attention_timing, reset_attention_timing


def calculate_metrics(df):
    """
    Calculate metrics for the AIME25 benchmark.
    """
    correct = 0
    answered = 0
    for _, row in df.iterrows():
        try:
            y_pred = str(row["predicted_answer"].split("boxed{")[-1].split("}")[0])
            y_true = str(row["answer"])
            score = int(y_pred == y_true)
        except IndexError:
            score = 0
        correct += score
        answered += "boxed{" in row["predicted_answer"]
    return {"correct": correct, "answered": answered, "accuracy": correct / len(df), "total": len(df)}


def _synchronize_if_needed():
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_idx)


def _peak_memory_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return sum(int(torch.cuda.max_memory_allocated(device_idx)) for device_idx in range(torch.cuda.device_count()))


def _allocated_memory_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return sum(int(torch.cuda.memory_allocated(device_idx)) for device_idx in range(torch.cuda.device_count()))


def _reset_peak_memory_stats():
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_idx)


def _get_storage_lengths(cache) -> list[float]:
    if hasattr(cache, "get_storage_lengths"):
        return list(cache.get_storage_lengths())  # type: ignore[call-arg]
    return [float(cache.get_seq_length(layer_idx)) for layer_idx in range(len(cache))]


def _extract_input_ids(tokenized) -> torch.Tensor:
    if isinstance(tokenized, torch.Tensor):
        return tokenized
    if hasattr(tokenized, "input_ids"):
        return tokenized.input_ids
    if "input_ids" in tokenized:
        return tokenized["input_ids"]
    raise TypeError(f"Unsupported chat template return type: {type(tokenized)!r}")


def _apply_sampling_filters(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> torch.Tensor:
    filtered_logits = logits.clone()

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    filtered_logits = filtered_logits / temperature

    if top_k > 0:
        top_k = min(top_k, filtered_logits.shape[-1])
        threshold = torch.topk(filtered_logits, top_k, dim=-1).values[..., -1, None]
        filtered_logits = filtered_logits.masked_fill(filtered_logits < threshold, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        remove_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        filtered_logits = filtered_logits.masked_fill(remove_mask, float("-inf"))

    if min_p > 0.0:
        probs = torch.softmax(filtered_logits, dim=-1)
        max_probs = probs.max(dim=-1, keepdim=True).values
        filtered_logits = filtered_logits.masked_fill(probs < (min_p * max_probs), float("-inf"))

    return filtered_logits


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> torch.Tensor:
    filtered_logits = _apply_sampling_filters(
        logits=logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )
    probs = torch.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _generate_answer_with_profile(
    model,
    tokenizer,
    tokens: torch.Tensor,
    press,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
):
    cache = DynamicCache()
    prefill_ids = tokens[:, :-1]
    decode_input_ids = tokens[:, -1:]
    generated_ids: list[torch.Tensor] = []

    eos_token_ids = model.generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]
    eos_token_ids = [token_id for token_id in eos_token_ids if token_id is not None]

    with torch.inference_mode():
        with press(model):
            _reset_peak_memory_stats()
            _synchronize_if_needed()
            prefill_start = perf_counter()
            if prefill_ids.shape[1] > 0:
                model.model(input_ids=prefill_ids, past_key_values=cache)
            _synchronize_if_needed()
            prefill_s = perf_counter() - prefill_start
            prefill_peak_memory_bytes = _peak_memory_bytes()
            post_prefill_memory_bytes = _allocated_memory_bytes()
            cache_lengths = _get_storage_lengths(cache)

            _reset_peak_memory_stats()
            _synchronize_if_needed()
            reset_attention_timing()
            decode_start = perf_counter()
            current_input_ids = decode_input_ids
            for _ in range(max_new_tokens):
                outputs = model(input_ids=current_input_ids, past_key_values=cache, use_cache=True)
                next_token = _sample_next_token(
                    logits=outputs.logits[:, -1, :],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                )
                generated_ids.append(next_token[0, 0].detach().cpu())
                current_input_ids = next_token
                if eos_token_ids and next_token[0, 0].item() in eos_token_ids:
                    break
            _synchronize_if_needed()
            decode_s = perf_counter() - decode_start
            decode_peak_memory_bytes = _peak_memory_bytes()
            attention_timing = get_attention_timing()

    answer = tokenizer.decode(torch.stack(generated_ids)) if generated_ids else ""
    generated_tokens = len(generated_ids)
    attention_decode_ms = float(attention_timing["attention_decode_ms"])
    kv_prepare_decode_ms = float(attention_timing["kv_prepare_decode_ms"])
    non_attention_decode_ms = max((decode_s * 1000.0) - attention_decode_ms - kv_prepare_decode_ms, 0.0)
    token_denominator = max(generated_tokens, 1)
    profile = {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_peak_memory_bytes": prefill_peak_memory_bytes,
        "decode_peak_memory_bytes": decode_peak_memory_bytes,
        "attention_decode_ms": attention_decode_ms,
        "kv_prepare_decode_ms": kv_prepare_decode_ms,
        "non_attention_decode_ms": non_attention_decode_ms,
        "attention_decode_ms_per_token": attention_decode_ms / token_denominator,
        "kv_prepare_decode_ms_per_token": kv_prepare_decode_ms / token_denominator,
        "non_attention_decode_ms_per_token": non_attention_decode_ms / token_denominator,
        "prefill_input_tokens": int(prefill_ids.shape[1]),
        "post_prefill_memory_bytes": post_prefill_memory_bytes,
        "post_prefill_cache_length_min": float(min(cache_lengths)) if cache_lengths else 0.0,
        "post_prefill_cache_length_max": float(max(cache_lengths)) if cache_lengths else 0.0,
        "post_prefill_cache_length_mean": float(sum(cache_lengths) / len(cache_lengths)) if cache_lengths else 0.0,
    }
    return answer, profile, generated_tokens


def evaluate(
    kvzap_model_type: str,
    threshold: float = 0.0,
    model_name: str = "Qwen/Qwen3-8B",
    device: str = "cuda:0",
    max_new_tokens: int = 32000,
    num: Optional[int] = None,
):
    """Evaluate KVzap on the AIME25 benchmark with explicit prompt-prefill and
    sampled decoding, mirroring the generate-time sampling policy while exposing
    split prefill/decode runtime metrics.

    Parameters
    ----------
    kvzap_model_type : str
        Model type - "mlp", "linear", or "no_press"
    threshold : float, optional
        Threshold for KVzap scores, by default 0.0
    model_name : str, optional
        HuggingFace model name, by default "Qwen/Qwen3-8B"
    device : str, optional
        Device to use, by default "cuda:0"
    max_new_tokens : int, optional
        Maximum number of tokens to generate, by default 32000
    num : int, optional
        Number of examples from the start of the AIME25 test split to evaluate.
        If None, evaluates the full split.
    """

    # Create press
    press: DMSPress | type[nullcontext[None]]
    if kvzap_model_type == "no_press":
        press = nullcontext
    else:
        press = DMSPress(
            KVzapPress(model_type=kvzap_model_type),
            threshold=threshold,
            decoding=True,
        )

    # Load tokenizer, model and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto").to(device)
    df = load_dataset("alessiodevoto/aime25", split="test").to_pandas()
    if num is not None:
        if num <= 0:
            raise ValueError(f"num must be positive, got {num}")
        df = df.iloc[:num].copy()
    df["latency_s"] = 0.0
    df["peak_memory_bytes"] = 0
    df["prompt_tokens"] = 0
    df["generated_tokens"] = 0
    df["tokens_per_second"] = 0.0
    df["compression_ratio"] = 0.0
    df["prefill_s"] = 0.0
    df["decode_s"] = 0.0
    df["prefill_peak_memory_bytes"] = 0
    df["decode_peak_memory_bytes"] = 0
    df["prefill_input_tokens"] = 0
    df["post_prefill_memory_bytes"] = 0
    df["post_prefill_cache_length_min"] = 0.0
    df["post_prefill_cache_length_max"] = 0.0
    df["post_prefill_cache_length_mean"] = 0.0
    df["attention_decode_ms"] = 0.0
    df["kv_prepare_decode_ms"] = 0.0
    df["non_attention_decode_ms"] = 0.0
    df["attention_decode_ms_per_token"] = 0.0
    df["kv_prepare_decode_ms_per_token"] = 0.0
    df["non_attention_decode_ms_per_token"] = 0.0

    # Run evaluation
    for idx, row in tqdm(df.iterrows(), total=len(df)):

        # Tokenize question
        messages = [{"role": "user", "content": row["question"]}]
        tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        if hasattr(tokenized, "to"):
            tokenized = tokenized.to(model.device)
        tokens = _extract_input_ids(tokenized)
        prompt_tokens = int(tokens.shape[1])

        answer, profile, generated_tokens = _generate_answer_with_profile(
            model=model,
            tokenizer=tokenizer,
            tokens=tokens,
            press=press,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
        )
        elapsed = float(profile["prefill_s"]) + float(profile["decode_s"])
        peak_memory_bytes = max(int(profile["prefill_peak_memory_bytes"]), int(profile["decode_peak_memory_bytes"]))
        df.loc[idx, "predicted_answer"] = answer
        df.loc[idx, "latency_s"] = elapsed
        df.loc[idx, "peak_memory_bytes"] = peak_memory_bytes
        df.loc[idx, "prompt_tokens"] = prompt_tokens
        df.loc[idx, "generated_tokens"] = generated_tokens
        df.loc[idx, "tokens_per_second"] = generated_tokens / max(elapsed, 1e-6)
        df.loc[idx, "prefill_s"] = float(profile["prefill_s"])
        df.loc[idx, "decode_s"] = float(profile["decode_s"])
        df.loc[idx, "prefill_peak_memory_bytes"] = int(profile["prefill_peak_memory_bytes"])
        df.loc[idx, "decode_peak_memory_bytes"] = int(profile["decode_peak_memory_bytes"])
        df.loc[idx, "prefill_input_tokens"] = int(profile["prefill_input_tokens"])
        df.loc[idx, "post_prefill_memory_bytes"] = int(profile["post_prefill_memory_bytes"])
        df.loc[idx, "post_prefill_cache_length_min"] = float(profile["post_prefill_cache_length_min"])
        df.loc[idx, "post_prefill_cache_length_max"] = float(profile["post_prefill_cache_length_max"])
        df.loc[idx, "post_prefill_cache_length_mean"] = float(profile["post_prefill_cache_length_mean"])
        df.loc[idx, "attention_decode_ms"] = float(profile["attention_decode_ms"])
        df.loc[idx, "kv_prepare_decode_ms"] = float(profile["kv_prepare_decode_ms"])
        df.loc[idx, "non_attention_decode_ms"] = float(profile["non_attention_decode_ms"])
        df.loc[idx, "attention_decode_ms_per_token"] = float(profile["attention_decode_ms_per_token"])
        df.loc[idx, "kv_prepare_decode_ms_per_token"] = float(profile["kv_prepare_decode_ms_per_token"])
        df.loc[idx, "non_attention_decode_ms_per_token"] = float(profile["non_attention_decode_ms_per_token"])
        if isinstance(press, DMSPress):
            df.loc[idx, "compression_ratio"] = press.compression_ratio
        else:
            df.loc[idx, "compression_ratio"] = 0

    # Save results in a new directory
    dir_id = uuid.uuid4().hex
    output_dir = Path(
        f"results/aime25__{model_name.replace('/', '--')}__kvzap_{kvzap_model_type}__{threshold:.2f}/{dir_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "predictions.csv", index=False)

    # Calculate and save metrics
    metrics = calculate_metrics(df)
    total_generated_tokens = int(df["generated_tokens"].sum())
    total_attention_decode_ms = float(df["attention_decode_ms"].sum())
    total_kv_prepare_decode_ms = float(df["kv_prepare_decode_ms"].sum())
    total_non_attention_decode_ms = float(df["non_attention_decode_ms"].sum())
    metrics["runtime"] = {
        "avg_latency_s": float(df["latency_s"].mean()),
        "p95_latency_s": float(np.quantile(df["latency_s"], 0.95)),
        "avg_peak_memory_gb": float(df["peak_memory_bytes"].mean() / (1024**3)),
        "max_peak_memory_gb": float(df["peak_memory_bytes"].max() / (1024**3)),
        "avg_prompt_tokens": float(df["prompt_tokens"].mean()),
        "avg_generated_tokens": float(df["generated_tokens"].mean()),
        "avg_tokens_per_second": float(df["generated_tokens"].sum() / max(df["latency_s"].sum(), 1e-6)),
        "avg_decode_tokens_per_second": float(df["generated_tokens"].sum() / max(df["decode_s"].sum(), 1e-6)),
        "avg_prefill_tokens_per_second": float(df["prefill_input_tokens"].sum() / max(df["prefill_s"].sum(), 1e-6)),
        "total_attention_decode_ms": total_attention_decode_ms,
        "total_kv_prepare_decode_ms": total_kv_prepare_decode_ms,
        "total_non_attention_decode_ms": total_non_attention_decode_ms,
        "attention_decode_ms_per_token": float(total_attention_decode_ms / max(total_generated_tokens, 1)),
        "kv_prepare_decode_ms_per_token": float(total_kv_prepare_decode_ms / max(total_generated_tokens, 1)),
        "non_attention_decode_ms_per_token": float(total_non_attention_decode_ms / max(total_generated_tokens, 1)),
        "avg_compression_ratio": float(df["compression_ratio"].mean()),
        "avg_prefill_s": float(df["prefill_s"].mean()),
        "avg_decode_s": float(df["decode_s"].mean()),
        "avg_prefill_peak_memory_gb": float(df["prefill_peak_memory_bytes"].mean() / (1024**3)),
        "max_prefill_peak_memory_gb": float(df["prefill_peak_memory_bytes"].max() / (1024**3)),
        "avg_decode_peak_memory_gb": float(df["decode_peak_memory_bytes"].mean() / (1024**3)),
        "max_decode_peak_memory_gb": float(df["decode_peak_memory_bytes"].max() / (1024**3)),
        "avg_post_prefill_memory_gb": float(df["post_prefill_memory_bytes"].mean() / (1024**3)),
        "max_post_prefill_memory_gb": float(df["post_prefill_memory_bytes"].max() / (1024**3)),
        "avg_post_prefill_cache_length_min": float(df["post_prefill_cache_length_min"].mean()),
        "avg_post_prefill_cache_length_max": float(df["post_prefill_cache_length_max"].mean()),
        "avg_post_prefill_cache_length_mean": float(df["post_prefill_cache_length_mean"].mean()),
        "backend": "manual_prefill_decode",
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results saved to {output_dir}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    import fire

    fire.Fire(evaluate)
