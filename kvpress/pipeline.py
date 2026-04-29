# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import contextlib
import logging
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, Cache, DynamicCache, Pipeline, QuantizedCache
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.base import GenericTensor

from kvpress.attention_timing import get_attention_timing, reset_attention_timing
from kvpress.compacted_cache import (
    CompactedDynamicCache,
    get_compacted_runtime_unavailable_reasons,
)
from kvpress.presses.base_press import BasePress
from kvpress.presses.dms_press import DMSPress

logger = logging.getLogger(__name__)


def _synchronize_all_devices():
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_idx)


def _allocated_memory_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return sum(int(torch.cuda.memory_allocated(device_idx)) for device_idx in range(torch.cuda.device_count()))


def _peak_memory_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return sum(int(torch.cuda.max_memory_allocated(device_idx)) for device_idx in range(torch.cuda.device_count()))


def _reset_peak_memory_stats_all_devices():
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_idx)


def _apply_sampling_filters(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> torch.Tensor:
    filtered_logits = logits.clone() / temperature

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


def _select_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> torch.Tensor:
    if not do_sample:
        return logits.argmax(dim=-1)

    filtered_logits = _apply_sampling_filters(
        logits=logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )
    probs = torch.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class KVPressTextGenerationPipeline(Pipeline):
    """
    Pipeline for key-value cache compression in causal language models.

    Enables efficient processing of long contexts by applying KV cache compression
    during pre-filling, then generating answers using greedy decoding.

    Example:
    ```python
    pipeline = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
    press = DMSPress(press=KVzapPress(model_type="mlp"), threshold=-5.0, decoding=True)
    result = pipeline(context="Long text...", question="A question about the long context.", press=press)
    ```
    """

    @staticmethod
    def _requires_compacted_cache_runtime(press: Optional[BasePress]) -> bool:
        if press is None:
            return False
        return bool(getattr(press, "requires_compacted_cache_runtime", False))

    @staticmethod
    def _supports_multiple_questions(press: Optional[BasePress]) -> bool:
        if press is None:
            return True
        if bool(getattr(press, "requires_continuous_context", False)):
            return bool(getattr(press, "supports_multiple_questions", False))
        if bool(getattr(press, "decoding", False)):
            return bool(getattr(press, "supports_multiple_questions", False))
        return True

    def _create_default_cache(self, press: Optional[BasePress]) -> Cache:
        if self._requires_compacted_cache_runtime(press):
            if not CompactedDynamicCache.is_supported_for_model(self.model):
                reasons = get_compacted_runtime_unavailable_reasons(self.model)
                reason_text = "; ".join(reasons) if reasons else "the compacted runtime is unavailable"
                raise RuntimeError(
                    f"{type(press).__name__} requires the compacted KV-cache runtime, but {reason_text}."
                )
        if press is not None:
            custom_cache = press.create_cache(self.model)
            if custom_cache is not None:
                return custom_cache
        if self._requires_compacted_cache_runtime(press):
            return CompactedDynamicCache(self.model)
        return DynamicCache()

    @staticmethod
    def _get_storage_lengths(cache: Cache) -> list[float]:
        if hasattr(cache, "get_storage_lengths"):
            return list(cache.get_storage_lengths())  # type: ignore[call-arg]
        return [float(cache.get_seq_length(layer_idx)) for layer_idx in range(len(cache))]

    def _estimate_cache_payload_bytes(self, cache: Cache) -> int:
        """
        Estimate logical KV payload bytes from retained per-layer token counts.

        This intentionally excludes auxiliary runtime buffers, model weights, and
        allocator overhead so evaluations can separate algorithmic KV retention
        from implementation-specific peak-memory costs.
        """
        storage_lengths = self._get_storage_lengths(cache)
        if not storage_lengths:
            return 0

        config = self.model.config.text_config if hasattr(self.model.config, "text_config") else self.model.config
        num_attention_heads = int(config.num_attention_heads)
        num_key_value_heads = int(getattr(config, "num_key_value_heads", num_attention_heads))
        head_dim = int(getattr(config, "head_dim", config.hidden_size // num_attention_heads))

        dtype = getattr(self.model, "dtype", None)
        if dtype is None:
            try:
                dtype = next(self.model.parameters()).dtype
            except (AttributeError, StopIteration):
                dtype = torch.float32
        element_size = torch.empty((), dtype=dtype).element_size()
        return int(round(sum(storage_lengths) * num_key_value_heads * head_dim * 2 * element_size))

    def _sanitize_parameters(
        self,
        question: Optional[str] = None,
        questions: Optional[list[str]] = None,
        answer_prefix: Optional[str] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
        enable_thinking: bool = False,
        cache: Optional[Cache] = None,
        return_profile: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        **kwargs,
    ):
        """
        Sanitize the input parameters for the pipeline.
        The user can either provide a single question or a list of questions to be asked about the context.

        Parameters
        ----------
        question : str, optional
            The question to be asked about the context. Exclusive with `questions`.
        questions : list[str], optional
            A list of questions to be asked about the context. Exclusive with `question`.
        answer_prefix : str, optional
            The prefix to be added to the generated answer.
        press : BasePress, optional
            The key-value cache compression method to apply during pre-filling.

            Typical project presses are `DMSPress(KVzapPress(...))`, `KVzipPress`,
            `KVzapCompactPress`, and `KVzapStreamingCompactPress`. If None, no
            compression is applied.
        max_new_tokens : int, optional
            The maximum number of new tokens to generate for each answer.
        max_context_length : int, optional
            The maximum number of tokens in the context. By default will use the maximum length supported by the model.
        enable_thinking: bool = False,
            Whether to enable thinking in the chat template (chat template must support this argument)
        cache : Cache, optional
            The cache to use for the forward pass. Defaults to None (DynamicCache).
        **kwargs : dict
            Additional keyword arguments, currently ignored.

        Returns
        -------
        Tuple[dict, dict, dict]
            A tuple containing three dictionaries:
                - preprocess_kwargs: The keyword arguments for the preprocess function.
                - forward_kwargs: The keyword arguments for the forward function.
                - postprocess_kwargs: The keyword arguments for the postprocess function.
        """

        answer_prefix = answer_prefix or ""
        postprocess_kwargs = {"single_question": questions is None, "return_profile": return_profile}
        assert question is None or questions is None, "Either question or questions should be provided, not both."
        questions = questions or ([question] if question else [""])
        if max_context_length is None:
            max_context_length = min(self.tokenizer.model_max_length, int(1e10))  # 1e10 to avoid overflow
        preprocess_kwargs = {
            "questions": questions,
            "answer_prefix": answer_prefix,
            "max_context_length": max_context_length,
            "enable_thinking": enable_thinking,
        }
        forward_kwargs = {
            "press": press,
            "max_new_tokens": max_new_tokens,
            "cache": cache,
            "return_profile": return_profile,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
        }
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
        enable_thinking: bool = False,
    ):
        """
        Apply chat template and tokenize the context and questions.

        Prepares input text for KV cache compression and generation by applying
        appropriate chat templates and tokenizing. Handles models with and without
        chat templates.

        Parameters
        ----------
        context : str
            Long context text to be compressed using the press method.
        questions : list[str]
            Questions to be asked about the context.
        answer_prefix : str
            Optional prefix for generated answers.
        max_context_length : int
            Maximum tokens allowed in context (truncated if exceeded).
        enable_thinking : bool
            Whether to enable thinking in the chat template (chat template must support this argument)

        Returns
        -------
        dict[str, GenericTensor]
            Dictionary with "context_ids" and "questions_ids" tensors.
        """

        # Apply chat template if available
        if self.tokenizer.chat_template is None:
            bos_token = getattr(self.tokenizer, "bos_token", "")
            context = bos_token + context
            question_suffix = "\n"  # to separate the question from the answer
        else:
            separator = "#" * (len(context) + 10)
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context + separator}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
            context, question_suffix = context.split(separator)

        # Add question_suffix and answer prefix
        # e.g. for llama3.1, question_suffix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        questions = [question + question_suffix + answer_prefix for question in questions]

        # Tokenize the context and questions
        context_ids = self.tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
        question_ids = [
            self.tokenizer.encode(question, return_tensors="pt", add_special_tokens=False) for question in questions
        ]

        # Truncate context
        if context_ids.shape[1] > max_context_length:
            logger.warning(
                f"Context length has been truncated from {context_ids.shape[1]} to {max_context_length} tokens."
            )
            context_ids = context_ids[:, :max_context_length]

        return {"context_ids": context_ids, "questions_ids": question_ids}

    def _forward(
        self,
        input_tensors: dict[str, GenericTensor],
        max_new_tokens: int = 50,
        press: Optional[BasePress] = None,
        cache: Optional[Cache] = None,
        return_profile: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
    ):
        """
        Execute KV cache compression and text generation pipeline.

        Performs context compression using the press method during pre-filling,
        then generates answers using greedy decoding.

        Parameters
        ----------
        input_tensors : dict[str, GenericTensor]
            Tokenized inputs with "context_ids" and "questions_ids".
        max_new_tokens : int, default=50
            Maximum tokens to generate for each answer.
        press : BasePress, optional
            Compression method for context pre-filling. If None, no compression.
        cache : Cache, optional
            Cache object for forward pass. If None, creates new DynamicCache.

        Returns
        -------
        list[str]
            Generated answers for each input question.
        """
        if len(input_tensors["questions_ids"]) > 1 and not self._supports_multiple_questions(press):
            raise ValueError(
                f"{type(press).__name__} is not compatible with multiple questions. Please specify a single question."
            )

        context_ids = input_tensors["context_ids"].to(self.model.device)
        context_length = context_ids.shape[1]
        profile: dict[str, int | float] | None = None

        # Prefilling using the press on the context
        if cache is None:
            cache = self._create_default_cache(press)
        elif self._requires_compacted_cache_runtime(press) and not isinstance(cache, CompactedDynamicCache):
            raise RuntimeError(
                f"{type(press).__name__} requires a CompactedDynamicCache. "
                "Do not pass a regular DynamicCache for this press."
            )

        # We only perform prefill compression if the press is a prefill press
        perform_prefill_compression = press is not None

        # We only perform decoding compression if the press explicitly supports decoding.
        perform_decoding_compression = bool(getattr(press, "decoding", False))

        use_continuous_context = bool(getattr(press, "requires_continuous_context", False))

        def run_prefill():
            nonlocal context_length, profile
            if return_profile:
                _synchronize_all_devices()
                _reset_peak_memory_stats_all_devices()
                prefill_start = time.perf_counter()
            # We run the model without the lm head for pre-filling.
            self.model.model(
                input_ids=context_ids,
                past_key_values=cache,
            )
            if press is not None:
                press.after_prefill()

            logger.debug(f"Context Length: {context_length}")
            storage_lengths = self._get_storage_lengths(cache)
            compressed_context_length = sum(storage_lengths) / len(storage_lengths) if storage_lengths else 0.0
            logger.debug(f"Compressed Context Length: {compressed_context_length:g}")

            # If prefill compression physically shortened the cache, decoding must use
            # the post-compression cache length rather than the original input length.
            context_length = cache.get_seq_length()

            if return_profile:
                _synchronize_all_devices()
                prefill_s = time.perf_counter() - prefill_start
                prefill_peak_memory_bytes = _peak_memory_bytes()
                post_prefill_memory_bytes = _allocated_memory_bytes()
                cache_lengths = self._get_storage_lengths(cache)
                profile = {
                    "prefill_s": prefill_s,
                    "prefill_peak_memory_bytes": prefill_peak_memory_bytes,
                    "prefill_input_tokens": int(input_tensors["context_ids"].shape[1]),
                    "post_prefill_memory_bytes": post_prefill_memory_bytes,
                    "post_prefill_cache_payload_bytes": self._estimate_cache_payload_bytes(cache),
                    "post_prefill_cache_length_min": float(min(cache_lengths)) if cache_lengths else 0.0,
                    "post_prefill_cache_length_max": float(max(cache_lengths)) if cache_lengths else 0.0,
                    "post_prefill_cache_length_mean": (
                        float(sum(cache_lengths) / len(cache_lengths)) if cache_lengths else 0.0
                    ),
                }

        def run_decode():
            if return_profile:
                reset_attention_timing()
                _synchronize_all_devices()
                _reset_peak_memory_stats_all_devices()
                decode_start = time.perf_counter()

            answers = []
            generated_tokens = 0
            question_input_tokens = 0
            decode_cache_length_mins = []
            decode_cache_length_maxs = []
            decode_cache_length_means = []
            decode_cache_payload_bytes = []
            should_snapshot_for_reuse = len(input_tensors["questions_ids"]) > 1
            for question_ids in input_tensors["questions_ids"]:
                question_input_tokens += int(question_ids.shape[1])
                cache_snapshot = cache.snapshot() if should_snapshot_for_reuse and hasattr(cache, "snapshot") else None
                press_snapshot = (
                    press.snapshot_state()
                    if (should_snapshot_for_reuse and press is not None and hasattr(press, "snapshot_state"))
                    else None
                )
                cache_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]
                answer = self.generate_answer(
                    question_ids=question_ids.to(self.model.device),
                    cache=cache,
                    context_length=context_length,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                )
                generated_tokens += max(cache.get_seq_length() - cache_seq_lengths[0] - int(question_ids.shape[1]), 0)
                if return_profile:
                    cache_lengths = self._get_storage_lengths(cache)
                    if cache_lengths:
                        decode_cache_length_mins.append(float(min(cache_lengths)))
                        decode_cache_length_maxs.append(float(max(cache_lengths)))
                        decode_cache_length_means.append(float(sum(cache_lengths) / len(cache_lengths)))
                    else:
                        decode_cache_length_mins.append(0.0)
                        decode_cache_length_maxs.append(0.0)
                        decode_cache_length_means.append(0.0)
                    decode_cache_payload_bytes.append(self._estimate_cache_payload_bytes(cache))
                if cache_snapshot is not None:
                    cache.restore(cache_snapshot)
                else:
                    self._remove_answer_from_cache(cache, cache_seq_lengths)
                if press_snapshot is not None and press is not None:
                    press.restore_state(press_snapshot)
                answers.append(answer)

            if return_profile:
                _synchronize_all_devices()
                assert profile is not None
                attention_timing = get_attention_timing()
                profile["decode_s"] = time.perf_counter() - decode_start
                profile["decode_peak_memory_bytes"] = _peak_memory_bytes()
                profile["generated_tokens"] = int(generated_tokens)
                profile["question_input_tokens"] = int(question_input_tokens)
                profile["effective_decode_tokens"] = int(generated_tokens + question_input_tokens)
                profile["attention_decode_ms"] = float(attention_timing["attention_decode_ms"])
                profile["kv_prepare_decode_ms"] = float(attention_timing["kv_prepare_decode_ms"])
                profile["post_decode_cache_payload_bytes_mean"] = (
                    float(sum(decode_cache_payload_bytes) / len(decode_cache_payload_bytes))
                    if decode_cache_payload_bytes
                    else 0.0
                )
                profile["post_decode_cache_payload_bytes_max"] = (
                    int(max(decode_cache_payload_bytes)) if decode_cache_payload_bytes else 0
                )
                profile["post_decode_cache_length_min"] = (
                    float(min(decode_cache_length_mins)) if decode_cache_length_mins else 0.0
                )
                profile["post_decode_cache_length_max"] = (
                    float(max(decode_cache_length_maxs)) if decode_cache_length_maxs else 0.0
                )
                profile["post_decode_cache_length_mean"] = (
                    float(sum(decode_cache_length_means) / len(decode_cache_length_means))
                    if decode_cache_length_means
                    else 0.0
                )
                profile["total_s"] = float(profile["prefill_s"]) + float(profile["decode_s"])
                return {"answers": answers, "profile": profile}
            return answers

        if use_continuous_context and press is not None:
            with press(self.model):
                run_prefill()
                return run_decode()

        with press(self.model) if perform_prefill_compression else contextlib.nullcontext():
            run_prefill()

        with press(self.model) if perform_decoding_compression else contextlib.nullcontext():
            return run_decode()

    def _remove_answer_from_cache(self, cache: Cache, cache_seq_lengths: list[int]):
        if hasattr(cache, "crop"):
            cache.crop(cache_seq_lengths[0])  # type: ignore[call-arg]
            return

        if hasattr(cache, "slice"):
            cache.slice(cache_seq_lengths[0])  # type: ignore[call-arg]
            return

        for layer_idx, sequence_length in enumerate(cache_seq_lengths):
            cache.layers[layer_idx].keys = cache.layers[layer_idx].keys[:, :, :sequence_length]
            cache.layers[layer_idx].values = cache.layers[layer_idx].values[:, :, :sequence_length]

        if isinstance(cache, QuantizedCache):
            for layer_idx, sequence_length in enumerate(cache_seq_lengths):
                cache.layers[layer_idx]._quantized_keys = cache.layers[layer_idx]._quantized_keys[
                    :, :, :sequence_length
                ]
                cache.layers[layer_idx]._quantized_values = cache.layers[layer_idx]._quantized_values[
                    :, :, :sequence_length
                ]

    def generate_answer(
        self,
        question_ids: torch.Tensor,
        cache: Cache,
        context_length: int,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
    ) -> str:
        """
        Generate an answer to a question using greedy decoding.

        Parameters
        ----------
        question_ids : torch.Tensor
            The tokenized question.
        cache : Cache
            The compressed key-value cache.
        context_length : int
            The length of the context.
        max_new_tokens : int
            The maximum number of new tokens to generate.

        Returns
        -------
        str
            The generated answer.
        """
        position_ids = torch.arange(
            context_length, context_length + question_ids.shape[1], device=self.model.device
        ).unsqueeze(0)

        # if the user doesn't provide a question, skip forward pass
        question_ids = question_ids.to(self.model.device)
        outputs = self.model(
            input_ids=question_ids,
            past_key_values=cache,
            position_ids=position_ids,
            num_logits_to_keep=1,
        )

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [
            _select_next_token(
                logits=outputs.logits[0, -1],
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
            )
        ]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in range(max_new_tokens - 1):
            outputs = self.model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=position_ids + i,
            )
            new_id = _select_next_token(
                logits=outputs.logits[0, -1],
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
            )
            generated_ids.append(new_id)
            if new_id.item() in should_stop_token_ids:
                break
        answer = str(self.tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True))
        return answer

    def postprocess(self, model_outputs, single_question, return_profile: bool = False):
        if return_profile:
            answers = model_outputs["answers"]
            profile = model_outputs["profile"]
            if single_question:
                return {"answer": answers[0], "profile": profile}
            return {"answers": answers, "profile": profile}
        if single_question:
            return {"answer": model_outputs[0]}
        return {"answers": model_outputs}


PIPELINE_REGISTRY.register_pipeline(
    "kv-press-text-generation",
    pipeline_class=KVPressTextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
)
