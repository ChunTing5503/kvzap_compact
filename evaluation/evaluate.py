# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from runtime_requirements import assert_core_runtime_requirements
except ImportError:
    from evaluation.runtime_requirements import assert_core_runtime_requirements

assert_core_runtime_requirements()

import numpy as np
import pandas as pd
import torch
import yaml
try:
    from benchmarks.scbench.load import is_scbench_dataset, load_scbench_dataframe
    from evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
except ImportError:
    from evaluation.benchmarks.scbench.load import is_scbench_dataset, load_scbench_dataframe
    from evaluation.evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import FineGrainedFP8Config, Pipeline, pipeline

from kvpress import (
    BasePress,
    DMSPress,
    KVzapCompactPress,
    KVzapStreamingCompactPress,
)
from kvpress.attention_timing import get_attention_timing, reset_attention_timing
from kvpress.compacted_cache import get_compacted_runtime_unavailable_reasons

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _normalize_aime25_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw AIME25 rows to the same prompt shape used by kvzap/evaluate_aime.py.

    The dedicated AIME runner feeds a single user chat turn containing only the
    problem statement (`row["question"]`). To reproduce that behavior in the
    generic KVPress pipeline, we keep the context empty and pass the raw problem
    text through the `question` field with an empty answer prefix.
    """
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("AIME25 dataset must contain 'question' and 'answer' columns.")

    normalized = pd.DataFrame(
        {
            "context": [""] * len(df),
            "question": df["question"].astype(str),
            "answer_prefix": [""] * len(df),
            "answer": df["answer"],
            "max_new_tokens": df["max_new_tokens"] if "max_new_tokens" in df.columns else 32000,
        }
    )
    return normalized


def _extract_input_ids(tokenized) -> torch.Tensor:
    if isinstance(tokenized, torch.Tensor):
        return tokenized
    if hasattr(tokenized, "input_ids"):
        return tokenized.input_ids
    if "input_ids" in tokenized:
        return tokenized["input_ids"]
    raise TypeError(f"Unsupported chat template return type: {type(tokenized)!r}")


@dataclass
class EvaluationConfig:
    """Dataclass to handle all the configuration for the evaluation."""

    # Core evaluation parameters
    dataset: str = "ruler"
    data_dir: Optional[str] = None
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: Optional[str] = None
    press_name: str = "knorm"
    compression_ratio: Optional[float] = 1.0
    key_channel_compression_ratio: Optional[float] = None
    threshold: Optional[float] = None
    sliding_window_size: int = 128

    # Dataset and generation parameters
    fraction: float = 1.0
    num: Optional[int] = None
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    query_aware: bool = False
    needle_depth: Optional[int] = None
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0

    # Decoding parameters
    compression_interval: Optional[int] = None
    target_size: Optional[int] = None
    hidden_states_buffer_size: Optional[int] = None
    decode_rebuild_interval: Optional[int] = None

    # Output and logging
    output_dir: str = "./results"
    log_level: str = "INFO"

    # Model-specific parameters
    model_kwargs: Optional[Dict[str, Any]] = None

    # Press information (will be set after press setup)
    press_init_command: Optional[str] = None

    # For reproducibility
    seed: int = 42

    # Quantization
    fp8: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate dataset
        assert self.dataset in DATASET_REGISTRY, f"No dataset found for {self.dataset}"
        assert self.dataset in SCORER_REGISTRY, f"No scorer found for {self.dataset}"

        # Validate press
        assert self.press_name in PRESS_REGISTRY, f"Press '{self.press_name}' not found in PRESS_REGISTRY"

        press = PRESS_REGISTRY[self.press_name]
        threshold_based_press = isinstance(
            press,
            (DMSPress, KVzapCompactPress, KVzapStreamingCompactPress),
        )

        if self.press_name == "no_press":
            # override compression_ratio to 0.0
            logger.info("Using 'no_press' configuration. Overriding compression_ratio to 0.0")
            self.compression_ratio = 0.0
        elif self.compression_ratio is None and not threshold_based_press:
            raise AssertionError(f"compression_ratio must be set for press '{self.press_name}'")

        # Only validate key_channel_compression_ratio if it's not None
        if self.key_channel_compression_ratio is not None:
            assert (
                0.0 <= self.key_channel_compression_ratio <= 1.0
            ), f"key_channel_compression_ratio must be between 0.0 and 1.0, got {self.key_channel_compression_ratio}"

        # Validate fraction
        assert 0.0 < self.fraction <= 1.0, f"fraction must be between 0.0 and 1.0, got {self.fraction}"
        if self.num is not None:
            assert self.num > 0, f"num must be positive, got {self.num}"
        assert self.temperature > 0.0, f"temperature must be positive, got {self.temperature}"
        assert 0.0 <= self.top_p <= 1.0, f"top_p must be between 0.0 and 1.0, got {self.top_p}"
        assert self.top_k >= 0, f"top_k must be non-negative, got {self.top_k}"
        assert 0.0 <= self.min_p <= 1.0, f"min_p must be between 0.0 and 1.0, got {self.min_p}"

        # Initialize model_kwargs if None
        if self.model_kwargs is None:
            self.model_kwargs = {}

    def get_results_dir(self, output_dir: Path) -> Path:
        """
        Generates the unique save directory and filenames based on configuration parameters.

        Parameters
        ----------
        output_dir : Path
            The output directory path

        Returns
        -------
        Path
            The path to the results directory
        """
        # Build directory name components
        ratio_component = "dynamic"
        if self.compression_ratio is not None:
            ratio_component = f"{self.compression_ratio:.2f}"
        components = [
            self.dataset,
            str(self.data_dir) if self.data_dir else "",
            self.model.replace("/", "--"),
            self.press_name,
            ratio_component,
        ]

        if self.threshold is not None:
            components[-1] = f"{self.threshold:.2f}"
        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.num is not None:
            components.append(f"num{self.num}")
        if self.max_context_length is not None:
            components.append(f"max_context{self.max_context_length}")
        if self.query_aware:
            components.append("query_aware")
        if self.key_channel_compression_ratio is not None:
            components.append(f"key_channel_cr{self.key_channel_compression_ratio:.2f}")
        dir_name = "__".join(filter(None, components))  # Filter None/empty strings
        config_dir = output_dir / dir_name

        # Make sure the directory does not exist, if it does, add a number to the end
        # This is to avoid overwriting results
        if config_dir.exists():
            i = 1
            while (config_dir / f"{i}").exists():
                i += 1
            config_dir = config_dir / f"{i}"

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def save_config(self, config_filename: Path):
        """
        Saves the evaluation configuration to a YAML file.
        """
        with open(str(config_filename), "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, indent=2, sort_keys=False)


def _load_yaml_config(path: str | Path) -> dict:
    """Loads a YAML file. Returns an empty dict if it doesn't exist."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found at {path}. Using only command-line arguments and defaults.")
        return {}


class EvaluationRunner:
    """
    EvaluationRunner class that orchestrates the entire evaluation process.

    Parameters
    ----------
    config : EvaluationConfig
        The configuration for the evaluation run.

    The final output will be predictions_<config>.csv and metrics_<config>.json in the output_dir.
    If the evaluation files already exist, evaluation will be skipped.

    """

    def __init__(self, config: EvaluationConfig):
        """
        Initializes the EvaluationRunner with a given configuration.

        Parameters
        ----------
        config : EvaluationConfig
            The configuration for the evaluation run.
        """
        self.config = config
        self.pipeline: Optional[Pipeline] = None  # Will be set by _setup_model_pipeline()
        self.press: Optional[BasePress] = None  # Will be set by _setup_press()
        self.df: Optional[pd.DataFrame] = None  # Will be set by _load_dataset()
        self.call_profiles: list[dict[str, int | float]] = []
        self.call_runtimes: list[dict[str, int | float]] = []
        self._setup_logging()
        self._setup_deterministic_seeds()
        logger.info(f"Initialized EvaluationRunner with config:\n{json.dumps(asdict(self.config), indent=2)}")

    def _setup_deterministic_seeds(self):
        """Set deterministic seeds for reproducible results."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Set deterministic seeds to {self.config.seed}")

    def _setup_logging(self):
        """Configures the logging level based on the config."""
        log_level = self.config.log_level.upper()

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(log_level)

    def _setup_directories(self) -> Path:
        """
        Creates the output directory for saving results if it doesn't exist.

        Returns
        -------
        Path
            The path to the output directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
        return output_dir

    @staticmethod
    def _requires_compacted_runtime(press: Optional[BasePress]) -> bool:
        if press is None:
            return False
        return bool(getattr(press, "requires_compacted_cache_runtime", False))

    def _preflight_compacted_runtime_requirements(self):
        if not self._requires_compacted_runtime(self.press):
            return

        reasons = get_compacted_runtime_unavailable_reasons()
        if self.config.device == "cpu":
            reasons.append("the configured device is cpu, but compacted KV-cache runs require CUDA")
        elif self.config.device in (None, "auto") and not torch.cuda.is_available():
            reasons.append("CUDA is not available, but compacted KV-cache runs require CUDA")
        dtype = (self.config.model_kwargs or {}).get("dtype")
        if dtype not in ("float16", torch.float16):
            reasons.append(
                "model_kwargs.dtype must be set to 'float16' for compacted KV-cache runs; 'auto' can resolve to bfloat16"
            )
        if self.config.fp8:
            reasons.append("fp8 quantization is not supported by the compacted KV-cache runtime")

        if reasons:
            reason_text = "; ".join(reasons)
            install_hint = (
                "Install the upstream kernel with "
                f"`cd {_REPO_ROOT / '_upstream' / 'KVzip' / 'csrc'} && python build.py install`."
            )
            raise RuntimeError(
                f"{type(self.press).__name__} requires the compacted KV-cache runtime, "
                f"but {reason_text}. {install_hint}"
            )

    @staticmethod
    def _uses_decoding_inference(press: Optional[BasePress]) -> bool:
        return press is not None and bool(getattr(press, "decoding", False))

    def _setup_press(self):
        """
        Initializes the KVPress instance and applies compression ratios based on its type.
        """
        press_name = self.config.press_name
        compression_ratio = self.config.compression_ratio
        key_channel_compression_ratio = self.config.key_channel_compression_ratio

        press = PRESS_REGISTRY[press_name]

        if isinstance(press, (DMSPress, KVzapCompactPress, KVzapStreamingCompactPress)):
            assert self.config.threshold is not None, "threshold must be set for threshold-based presses"
            press.threshold = self.config.threshold
            press.sliding_window_size = self.config.sliding_window_size
            if isinstance(press, KVzapStreamingCompactPress) and self.config.decode_rebuild_interval is not None:
                press.decode_rebuild_interval = int(self.config.decode_rebuild_interval)
            logger.info(f"Set {press.__class__.__name__} threshold to {press.threshold}")
        else:
            if hasattr(press, "compression_ratio"):
                press.compression_ratio = compression_ratio
                logger.info(f"Set {press.__class__.__name__} compression_ratio to {compression_ratio}")
            else:
                logger.warning(
                    f"Press {press.__class__.__name__} has no 'compression_ratio' attribute. This is expected is you set `no_press`."
                )

        self.press = press
        # Set the press info in the config for saving to YAML
        self.config.press_init_command = str(press)
        logger.info(f"KV Press '{press_name}' setup.")

    def _load_and_prepare_dataset(self):
        """
        Loads the dataset specified in the config and applies sampling/filtering.
        """
        dataset_name = self.config.dataset
        data_dir = str(self.config.data_dir) if self.config.data_dir else None
        fraction = self.config.fraction

        logger.info(f"Loading dataset: {DATASET_REGISTRY[dataset_name]} (data_dir: {data_dir})")
        if is_scbench_dataset(dataset_name):
            if data_dir is not None:
                logger.warning("Ignoring data_dir for SCBench datasets; the dataset name already selects the parquet shard.")
            df = load_scbench_dataframe(dataset_name)
        else:
            df = load_dataset(DATASET_REGISTRY[dataset_name], data_dir=data_dir, split="test").to_pandas()
            if dataset_name == "aime25":
                df = _normalize_aime25_dataframe(df)

        if fraction < 1.0:
            original_len = len(df)
            df = df.sample(frac=fraction, random_state=self.config.seed)
            logger.info(f"Sampled {len(df)} samples ({fraction:.2f}) from original {original_len} samples.")

        if self.config.num is not None:
            original_len = len(df)
            df = df.head(self.config.num)
            logger.info(f"Trimmed dataset to first {len(df)} samples from {original_len} total using num={self.config.num}.")

        logger.info(f"Dataset loaded with {len(df)} entries.")

        if self.config.query_aware:
            logger.info("Query-aware compression: including question in context for compression.")
            df["context"] = df["context"] + df["question"]  # type: ignore[index]
            df["question"] = ""  # type: ignore[index]

        self.df = df
        logger.info(f"Dataset processed with {len(self.df)} entries.")

    def _setup_model_pipeline(self):
        model_name = self.config.model
        device = self.config.device

        if device is None:
            device = "auto" if torch.cuda.is_available() else "cpu"
            logger.info(f"No device specified, auto-detected device: {device}")
        elif isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Configured device is '{device}', but torch.cuda.is_available() is False. "
                "Run this evaluation on a GPU node or change the config to a CPU-safe setup. "
                "The compacted runtime and FlashAttention-based paths require CUDA."
            )

        model_kwargs = dict(self.config.model_kwargs or {})
        self._preflight_compacted_runtime_requirements()

        # Configs use `dtype`, but transformers model loading expects `torch_dtype`.
        dtype = model_kwargs.pop("dtype", None)
        if dtype is not None:
            if dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif dtype == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif dtype == "float32":
                model_kwargs["torch_dtype"] = torch.float32
            else:
                model_kwargs["torch_dtype"] = dtype

        if self.config.fp8:
            model_kwargs["quantization_config"] = FineGrainedFP8Config()
            logger.info("FP8 quantization enabled.")

        wants_cuda_attention = torch.cuda.is_available() and (
            device == "auto" or (isinstance(device, str) and device.startswith("cuda"))
        )
        if wants_cuda_attention:
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 detected, setting attn_implementation to 'flash_attention_2'.")
            except ImportError:
                logger.info("Flash Attention 2 not available, using default attn_implementation.")
        else:
            logger.info("Skipping Flash Attention 2 because the selected device is not CUDA.")

        logger.info(f"Loading model pipeline for: {model_name} on device: {device} with model_kwargs: {model_kwargs}")
        pipeline_kwargs = {
            "model": model_name,
            "model_kwargs": model_kwargs,
            "trust_remote_code": True,
        }
        if device == "auto":
            pipeline_kwargs["device_map"] = "auto"
        else:
            pipeline_kwargs["device"] = device
        self.pipeline = pipeline("kv-press-text-generation", **pipeline_kwargs)

        self.pipeline.model.eval()
        if self._requires_compacted_runtime(self.press):
            reasons = get_compacted_runtime_unavailable_reasons(self.pipeline.model)
            if reasons:
                reason_text = "; ".join(reasons)
                raise RuntimeError(
                    f"{type(self.press).__name__} requires the compacted KV-cache runtime, but {reason_text}."
                )
        logger.info("Model pipeline loaded.")

    def _should_use_exact_aime25_prompt_path(self) -> bool:
        return self.config.dataset == "aime25" and isinstance(self.press, KVzapStreamingCompactPress)

    def _run_exact_aime25_prompt_path(self, question: str, max_new_tokens: int) -> dict[str, object]:
        assert self.pipeline is not None
        tokenizer = self.pipeline.tokenizer
        model = self.pipeline.model
        cache = self.pipeline._create_default_cache(self.press)
        messages = [{"role": "user", "content": str(question)}]
        prompt_ids = _extract_input_ids(
            tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        )
        prefill_ids = prompt_ids[:, :-1].to(model.device)
        decode_input_ids = prompt_ids[:, -1:].to(model.device)

        profile: dict[str, int | float] = {}
        with (self.press(model) if self.press is not None else nullcontext()):
            _reset_peak_memory_stats()
            _synchronize_if_needed()
            prefill_start = time.perf_counter()
            if prefill_ids.shape[1] > 0:
                model.model(input_ids=prefill_ids, past_key_values=cache)
            if self.press is not None:
                self.press.after_prefill()
            _synchronize_if_needed()
            prefill_s = time.perf_counter() - prefill_start
            prefill_peak_memory_bytes = _peak_memory_bytes()
            post_prefill_memory_bytes = _allocated_memory_bytes()
            prefill_cache_lengths = self.pipeline._get_storage_lengths(cache)
            profile.update(
                {
                    "prefill_s": prefill_s,
                    "prefill_peak_memory_bytes": prefill_peak_memory_bytes,
                    "prefill_input_tokens": int(prefill_ids.shape[1]),
                    "post_prefill_memory_bytes": post_prefill_memory_bytes,
                    "post_prefill_cache_payload_bytes": self.pipeline._estimate_cache_payload_bytes(cache),
                    "post_prefill_cache_length_min": (
                        float(min(prefill_cache_lengths)) if prefill_cache_lengths else 0.0
                    ),
                    "post_prefill_cache_length_max": (
                        float(max(prefill_cache_lengths)) if prefill_cache_lengths else 0.0
                    ),
                    "post_prefill_cache_length_mean": (
                        float(sum(prefill_cache_lengths) / len(prefill_cache_lengths)) if prefill_cache_lengths else 0.0
                    ),
                }
            )

            _reset_peak_memory_stats()
            _synchronize_if_needed()
            reset_attention_timing()
            decode_start = time.perf_counter()
            cache_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]
            answer = self.pipeline.generate_answer(
                question_ids=decode_input_ids,
                cache=cache,
                context_length=int(prefill_ids.shape[1]),
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                min_p=self.config.min_p,
            )
            _synchronize_if_needed()
            decode_s = time.perf_counter() - decode_start
            decode_peak_memory_bytes = _peak_memory_bytes()
            attention_timing = get_attention_timing()
            generated_tokens = max(cache.get_seq_length() - cache_seq_lengths[0] - int(decode_input_ids.shape[1]), 0)
            decode_cache_lengths = self.pipeline._get_storage_lengths(cache)
            decode_cache_payload_bytes = self.pipeline._estimate_cache_payload_bytes(cache)
            profile.update(
                {
                    "decode_s": decode_s,
                    "decode_peak_memory_bytes": decode_peak_memory_bytes,
                    "generated_tokens": int(generated_tokens),
                    "question_input_tokens": int(decode_input_ids.shape[1]),
                    "effective_decode_tokens": int(generated_tokens + int(decode_input_ids.shape[1])),
                    "attention_decode_ms": float(attention_timing["attention_decode_ms"]),
                    "kv_prepare_decode_ms": float(attention_timing["kv_prepare_decode_ms"]),
                    "post_decode_cache_payload_bytes_mean": float(decode_cache_payload_bytes),
                    "post_decode_cache_payload_bytes_max": int(decode_cache_payload_bytes),
                    "post_decode_cache_length_min": (
                        float(min(decode_cache_lengths)) if decode_cache_lengths else 0.0
                    ),
                    "post_decode_cache_length_max": (
                        float(max(decode_cache_lengths)) if decode_cache_lengths else 0.0
                    ),
                    "post_decode_cache_length_mean": (
                        float(sum(decode_cache_lengths) / len(decode_cache_lengths)) if decode_cache_lengths else 0.0
                    ),
                    "peak_memory_bytes": int(max(prefill_peak_memory_bytes, decode_peak_memory_bytes)),
                }
            )
        return {"answer": answer, "profile": profile}

    @torch.inference_mode()
    def _run_inference(self):
        """
        Executes the inference process on the prepared dataset using the model pipeline.
        """

        self.df["predicted_answer"] = None  # type: ignore[index]

        self.df["compression_ratio"] = 0.0  # type: ignore[index]
        self.df["latency_s"] = 0.0  # type: ignore[index]
        self.df["peak_memory_bytes"] = 0  # type: ignore[index]
        self.df["generated_tokens"] = 0  # type: ignore[index]
        self.df["prefill_s"] = 0.0  # type: ignore[index]
        self.df["decode_s"] = 0.0  # type: ignore[index]
        self.df["prefill_peak_memory_bytes"] = 0  # type: ignore[index]
        self.df["decode_peak_memory_bytes"] = 0  # type: ignore[index]
        self.df["prefill_input_tokens"] = 0  # type: ignore[index]
        self.df["post_prefill_memory_bytes"] = 0  # type: ignore[index]
        self.df["post_prefill_cache_payload_bytes"] = 0  # type: ignore[index]
        self.df["post_prefill_cache_length_min"] = 0  # type: ignore[index]
        self.df["post_prefill_cache_length_max"] = 0  # type: ignore[index]
        self.df["post_prefill_cache_length_mean"] = 0.0  # type: ignore[index]
        self.df["post_decode_cache_payload_bytes_mean"] = 0.0  # type: ignore[index]
        self.df["post_decode_cache_payload_bytes_max"] = 0  # type: ignore[index]
        self.df["post_decode_cache_length_min"] = 0.0  # type: ignore[index]
        self.df["post_decode_cache_length_max"] = 0.0  # type: ignore[index]
        self.df["post_decode_cache_length_mean"] = 0.0  # type: ignore[index]

        self.call_profiles = []
        self.call_runtimes = []

        uses_decoding = self._uses_decoding_inference(self.press)

        if uses_decoding:
            logger.info("Decoding-capable press detected, running inference for each context-question pair.")
            for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Running Inference"):
                context = row["context"]
                question = row["question"]
                answer_prefix = row["answer_prefix"]
                max_new_tokens = self.config.max_new_tokens or row["max_new_tokens"]
                start_time = time.perf_counter()
                if self._should_use_exact_aime25_prompt_path():
                    output = self._run_exact_aime25_prompt_path(question=question, max_new_tokens=max_new_tokens)
                else:
                    _reset_peak_memory_stats()
                    _synchronize_if_needed()
                    output = self.pipeline(
                        context,
                        question=question,
                        answer_prefix=answer_prefix,
                        press=self.press,
                        max_new_tokens=max_new_tokens,
                        max_context_length=self.config.max_context_length,
                        do_sample=self.config.do_sample,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        min_p=self.config.min_p,
                        return_profile=True,
                    )
                _synchronize_if_needed()
                elapsed = time.perf_counter() - start_time
                answer = output["answer"]  # type: ignore[index]
                profile = output["profile"]  # type: ignore[index]
                generated_tokens = int(
                    profile.get(
                        "generated_tokens",
                        len(self.pipeline.tokenizer.encode(answer, add_special_tokens=False)),
                    )
                )
                compression_ratio = self.press.compression_ratio if self.press is not None else 0.0  # type: ignore[attr-defined]
                peak_memory_bytes = int(profile.get("peak_memory_bytes", _peak_memory_bytes()))

                self.call_profiles.append(profile)
                self.call_runtimes.append(
                    {
                        "latency_s": elapsed,
                        "peak_memory_bytes": peak_memory_bytes,
                        "generated_tokens": generated_tokens,
                        "compression_ratio": compression_ratio,
                    }
                )

                self.df.loc[index, "predicted_answer"] = answer  # type: ignore[union-attr]
                self.df.loc[index, "compression_ratio"] = compression_ratio  # type: ignore[union-attr]
                self.df.loc[index, "latency_s"] = elapsed  # type: ignore[union-attr]
                self.df.loc[index, "peak_memory_bytes"] = peak_memory_bytes  # type: ignore[union-attr]
                self.df.loc[index, "generated_tokens"] = generated_tokens  # type: ignore[union-attr]
                self.df.loc[index, "prefill_s"] = float(profile["prefill_s"])  # type: ignore[union-attr]
                self.df.loc[index, "decode_s"] = float(profile["decode_s"])  # type: ignore[union-attr]
                self.df.loc[index, "prefill_peak_memory_bytes"] = int(profile["prefill_peak_memory_bytes"])  # type: ignore[union-attr]
                self.df.loc[index, "decode_peak_memory_bytes"] = int(profile["decode_peak_memory_bytes"])  # type: ignore[union-attr]
                self.df.loc[index, "prefill_input_tokens"] = int(profile["prefill_input_tokens"])  # type: ignore[union-attr]
                self.df.loc[index, "post_prefill_memory_bytes"] = int(profile["post_prefill_memory_bytes"])  # type: ignore[union-attr]
                self.df.loc[index, "post_prefill_cache_payload_bytes"] = int(profile["post_prefill_cache_payload_bytes"])  # type: ignore[union-attr]
                self.df.loc[index, "post_prefill_cache_length_min"] = int(profile["post_prefill_cache_length_min"])  # type: ignore[union-attr]
                self.df.loc[index, "post_prefill_cache_length_max"] = int(profile["post_prefill_cache_length_max"])  # type: ignore[union-attr]
                self.df.loc[index, "post_prefill_cache_length_mean"] = float(profile["post_prefill_cache_length_mean"])  # type: ignore[union-attr]
                self.df.loc[index, "post_decode_cache_payload_bytes_mean"] = float(profile["post_decode_cache_payload_bytes_mean"])  # type: ignore[union-attr]
                self.df.loc[index, "post_decode_cache_payload_bytes_max"] = int(profile["post_decode_cache_payload_bytes_max"])  # type: ignore[union-attr]
                self.df.loc[index, "post_decode_cache_length_min"] = float(profile["post_decode_cache_length_min"])  # type: ignore[union-attr]
                self.df.loc[index, "post_decode_cache_length_max"] = float(profile["post_decode_cache_length_max"])  # type: ignore[union-attr]
                self.df.loc[index, "post_decode_cache_length_mean"] = float(profile["post_decode_cache_length_mean"])  # type: ignore[union-attr]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        else:
            df_context_grouped = self.df.groupby("context")  # type: ignore[union-attr]
            assert all(
                df_context_grouped["answer_prefix"].nunique() == 1
            ), "Inconsistent 'answer_prefix' within the same context group detected."

            logger.info("Starting inference...")
            for context, df_group in tqdm(
                df_context_grouped, total=self.df["context"].nunique(), desc="Running Inference"
            ):  # type: ignore[union-attr]
                questions = df_group["question"].to_list()
                max_new_tokens = self.config.max_new_tokens or df_group["max_new_tokens"].iloc[0]
                answer_prefix = df_group["answer_prefix"].iloc[0]

                _reset_peak_memory_stats()
                _synchronize_if_needed()
                start_time = time.perf_counter()
                output = self.pipeline(  # type: ignore[misc]
                    context,
                    questions=questions,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    min_p=self.config.min_p,
                    return_profile=True,
                )
                _synchronize_if_needed()
                elapsed = time.perf_counter() - start_time
                answers = output["answers"]
                profile = output["profile"]
                compression_ratio = self.press.compression_ratio if self.press is not None else 0.0  # type: ignore[attr-defined]
                peak_memory_bytes = _peak_memory_bytes()
                fallback_generated_tokens = [
                    len(self.pipeline.tokenizer.encode(answer, add_special_tokens=False)) for answer in answers
                ]
                generated_tokens_total = int(profile.get("generated_tokens", sum(fallback_generated_tokens)))

                self.call_profiles.append(profile)
                self.call_runtimes.append(
                    {
                        "latency_s": elapsed,
                        "peak_memory_bytes": peak_memory_bytes,
                        "generated_tokens": generated_tokens_total,
                        "compression_ratio": compression_ratio,
                    }
                )

                self.df.loc[df_group.index, "predicted_answer"] = answers  # type: ignore[union-attr]
                self.df.loc[df_group.index, "compression_ratio"] = compression_ratio  # type: ignore[union-attr]
                self.df.loc[df_group.index, "latency_s"] = elapsed  # type: ignore[union-attr]
                self.df.loc[df_group.index, "peak_memory_bytes"] = peak_memory_bytes  # type: ignore[union-attr]
                self.df.loc[df_group.index, "generated_tokens"] = fallback_generated_tokens  # type: ignore[union-attr]
                self.df.loc[df_group.index, "prefill_s"] = float(profile["prefill_s"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "decode_s"] = float(profile["decode_s"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "prefill_peak_memory_bytes"] = int(profile["prefill_peak_memory_bytes"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "decode_peak_memory_bytes"] = int(profile["decode_peak_memory_bytes"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "prefill_input_tokens"] = int(profile["prefill_input_tokens"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "post_prefill_memory_bytes"] = int(profile["post_prefill_memory_bytes"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "post_prefill_cache_payload_bytes"] = int(profile["post_prefill_cache_payload_bytes"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "post_prefill_cache_length_min"] = int(profile["post_prefill_cache_length_min"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "post_prefill_cache_length_max"] = int(profile["post_prefill_cache_length_max"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "post_prefill_cache_length_mean"] = float(profile["post_prefill_cache_length_mean"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "post_decode_cache_payload_bytes_mean"] = float(profile["post_decode_cache_payload_bytes_mean"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "post_decode_cache_payload_bytes_max"] = int(profile["post_decode_cache_payload_bytes_max"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "post_decode_cache_length_min"] = float(profile["post_decode_cache_length_min"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "post_decode_cache_length_max"] = float(profile["post_decode_cache_length_max"])  # type: ignore[union-attr]
                self.df.loc[df_group.index, "post_decode_cache_length_mean"] = float(profile["post_decode_cache_length_mean"])  # type: ignore[union-attr]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info("Inference completed.")

    def _save_results(self, save_filename: Path):
        """
        Saves the predicted answers and compression ratios to a CSV file.

        Parameters
        ----------
        save_filename : Path
            The full path including filename to save the CSV.
        """
        if save_filename.exists():
            logger.warning(f"Results CSV already exists at {save_filename}. Overwriting.")

        self.df[list(set(self.df.columns) - set(["context"]))].to_csv(
            str(save_filename), index=False
        )  # type: ignore[index]
        logger.info(f"Results saved to {save_filename}")

    def _calculate_and_save_metrics(self, save_filename: Path):
        """
        Calculates evaluation metrics and saves them to a JSON file.

        Parameters
        ----------
        save_filename : Path
            The base filename (e.g., CSV path) to derive the JSON path from.
        """
        dataset_name = self.config.dataset
        scorer = SCORER_REGISTRY[dataset_name]

        logger.info(f"Calculating metrics for dataset: {dataset_name}")
        quality_metrics = scorer(self.df)  # type: ignore[call-arg]
        metrics = dict(quality_metrics)

        if self.call_runtimes:
            question_input_tokens = [
                int(profile.get("question_input_tokens", 0)) for profile in self.call_profiles
            ]
            effective_decode_tokens = [
                int(profile.get("effective_decode_tokens", int(call["generated_tokens"])))
                for profile, call in zip(self.call_profiles, self.call_runtimes)
            ]
            attention_decode_ms = [float(profile.get("attention_decode_ms", 0.0)) for profile in self.call_profiles]
            kv_prepare_decode_ms = [
                float(profile.get("kv_prepare_decode_ms", 0.0)) for profile in self.call_profiles
            ]
            post_prefill_cache_payload_bytes = [
                int(profile.get("post_prefill_cache_payload_bytes", 0)) for profile in self.call_profiles
            ]
            post_decode_cache_payload_bytes_mean = [
                float(profile.get("post_decode_cache_payload_bytes_mean", 0.0)) for profile in self.call_profiles
            ]
            post_decode_cache_payload_bytes_max = [
                int(profile.get("post_decode_cache_payload_bytes_max", 0)) for profile in self.call_profiles
            ]
            post_prefill_cache_length_mean = [
                float(profile["post_prefill_cache_length_mean"]) for profile in self.call_profiles
            ]
            post_decode_cache_length_mean = [
                float(profile.get("post_decode_cache_length_mean", 0.0)) for profile in self.call_profiles
            ]
            runtime_metrics = {
                "avg_latency_s": float(np.mean([float(call["latency_s"]) for call in self.call_runtimes])),
                "p95_latency_s": float(np.quantile([float(call["latency_s"]) for call in self.call_runtimes], 0.95)),
                "avg_peak_memory_gb": float(
                    np.mean([int(call["peak_memory_bytes"]) for call in self.call_runtimes]) / (1024**3)
                ),
                "max_peak_memory_gb": float(
                    max(int(call["peak_memory_bytes"]) for call in self.call_runtimes) / (1024**3)
                ),
                "avg_generated_tokens": float(np.mean([int(call["generated_tokens"]) for call in self.call_runtimes])),
                "avg_tokens_per_second": float(
                    sum(int(call["generated_tokens"]) for call in self.call_runtimes)
                    / max(sum(float(call["latency_s"]) for call in self.call_runtimes), 1e-6)
                ),
                "avg_decode_tokens_per_second": float(
                    sum(int(call["generated_tokens"]) for call in self.call_runtimes)
                    / max(sum(float(profile["decode_s"]) for profile in self.call_profiles), 1e-6)
                ),
                "avg_effective_decode_tokens_per_second": float(
                    sum(effective_decode_tokens) / max(sum(float(profile["decode_s"]) for profile in self.call_profiles), 1e-6)
                ),
                "avg_prefill_tokens_per_second": float(
                    sum(int(profile["prefill_input_tokens"]) for profile in self.call_profiles)
                    / max(sum(float(profile["prefill_s"]) for profile in self.call_profiles), 1e-6)
                ),
                "avg_compression_ratio": float(
                    np.mean([float(call["compression_ratio"]) for call in self.call_runtimes])
                ),
                "avg_prefill_s": float(np.mean([float(profile["prefill_s"]) for profile in self.call_profiles])),
                "avg_decode_s": float(np.mean([float(profile["decode_s"]) for profile in self.call_profiles])),
                "avg_question_input_tokens": float(np.mean(question_input_tokens)),
                "avg_effective_decode_tokens": float(np.mean(effective_decode_tokens)),
                "avg_attention_decode_ms": float(np.mean(attention_decode_ms)),
                "avg_kv_prepare_decode_ms": float(np.mean(kv_prepare_decode_ms)),
                "avg_prefill_peak_memory_gb": float(
                    np.mean([int(profile["prefill_peak_memory_bytes"]) for profile in self.call_profiles]) / (1024**3)
                ),
                "max_prefill_peak_memory_gb": float(
                    max(int(profile["prefill_peak_memory_bytes"]) for profile in self.call_profiles) / (1024**3)
                ),
                "avg_decode_peak_memory_gb": float(
                    np.mean([int(profile["decode_peak_memory_bytes"]) for profile in self.call_profiles]) / (1024**3)
                ),
                "max_decode_peak_memory_gb": float(
                    max(int(profile["decode_peak_memory_bytes"]) for profile in self.call_profiles) / (1024**3)
                ),
                "avg_post_prefill_memory_gb": float(
                    np.mean([int(profile["post_prefill_memory_bytes"]) for profile in self.call_profiles]) / (1024**3)
                ),
                "max_post_prefill_memory_gb": float(
                    max(int(profile["post_prefill_memory_bytes"]) for profile in self.call_profiles) / (1024**3)
                ),
                "avg_post_prefill_cache_payload_gb": float(
                    np.mean(post_prefill_cache_payload_bytes) / (1024**3)
                ),
                "max_post_prefill_cache_payload_gb": float(max(post_prefill_cache_payload_bytes) / (1024**3)),
                "avg_post_decode_cache_payload_gb": float(
                    np.mean(post_decode_cache_payload_bytes_mean) / (1024**3)
                ),
                "max_post_decode_cache_payload_gb": float(max(post_decode_cache_payload_bytes_max) / (1024**3)),
                "avg_decode_cache_payload_growth_gb": float(
                    np.mean(
                        [
                            decode_bytes - prefill_bytes
                            for decode_bytes, prefill_bytes in zip(
                                post_decode_cache_payload_bytes_mean, post_prefill_cache_payload_bytes
                            )
                        ]
                    )
                    / (1024**3)
                ),
                "avg_post_prefill_cache_length_min": float(
                    np.mean([int(profile["post_prefill_cache_length_min"]) for profile in self.call_profiles])
                ),
                "avg_post_prefill_cache_length_max": float(
                    np.mean([int(profile["post_prefill_cache_length_max"]) for profile in self.call_profiles])
                ),
                "avg_post_prefill_cache_length_mean": float(
                    np.mean(post_prefill_cache_length_mean)
                ),
                "avg_post_decode_cache_length_min": float(
                    np.mean([float(profile.get("post_decode_cache_length_min", 0.0)) for profile in self.call_profiles])
                ),
                "avg_post_decode_cache_length_max": float(
                    np.mean([float(profile.get("post_decode_cache_length_max", 0.0)) for profile in self.call_profiles])
                ),
                "avg_post_decode_cache_length_mean": float(
                    np.mean(post_decode_cache_length_mean)
                ),
                "avg_decode_cache_length_growth_mean": float(
                    np.mean(
                        [
                            decode_length - prefill_length
                            for decode_length, prefill_length in zip(
                                post_decode_cache_length_mean, post_prefill_cache_length_mean
                            )
                        ]
                    )
                ),
            }
            metrics["runtime"] = runtime_metrics

        with open(str(save_filename), "w") as f:
            json.dump(metrics, f, indent=4)  # Pretty print JSON

        logger.info(f"Metrics saved to {save_filename}")
        logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")

    def run_evaluation(self):
        """
        Orchestrates the entire evaluation process.
        """
        logger.info("Starting evaluation run...")
        output_dir = self._setup_directories()

        results_dir = self.config.get_results_dir(output_dir)
        predictions_filename = results_dir / "predictions.csv"
        metrics_filename = results_dir / "metrics.json"
        config_filename = results_dir / "config.yaml"

        if predictions_filename.exists() and metrics_filename.exists():
            logger.info(
                f"Evaluation files already exist at \n {predictions_filename} \n {metrics_filename}.\nSkipping..."
            )
            return

        self._setup_press()
        self._setup_model_pipeline()
        self._load_and_prepare_dataset()

        self._run_inference()
        self._save_results(predictions_filename)
        self._calculate_and_save_metrics(metrics_filename)
        self.config.save_config(config_filename)
        logger.info("Evaluation run completed successfully.")


# --- Command-Line Interface ---
class CliEntryPoint:
    """
    CLI entry point for building configuration and running the evaluation.

    This class provides a command-line interface for running KVPress evaluations.
    Configuration can be specified via:
    1. YAML config file (default: "./evaluate_config.yaml")
    2. Command-line arguments (highest priority)
    """

    def __call__(self, config_file: Optional[str] = "./evaluate_config.yaml", **cli_overrides):
        """
        Builds the configuration and runs the evaluation.

        Configuration is built by layering:
        1. Default values from EvaluationConfig
        2. Values from YAML config file
        3. Command-line arguments (highest priority)
        """
        # 1. Start with dataclass defaults.
        final_args = asdict(EvaluationConfig())

        # 2. Layer YAML values on top.
        yaml_config = _load_yaml_config(config_file)
        final_args.update(yaml_config)

        # 3. Layer CLI arguments on top (highest priority).
        # Filter out None values from CLI overrides
        cli_args = {k: v for k, v in cli_overrides.items() if v is not None}
        final_args.update(cli_args)

        # 4. Create and validate the final config object.
        try:
            config = EvaluationConfig(**final_args)
        except TypeError as e:
            # Provide a user-friendly error for bad arguments.
            print(f"Error: Invalid configuration argument provided. {e}", file=sys.stderr)
            sys.exit(1)

        runner = EvaluationRunner(config)
        runner.run_evaluation()


if __name__ == "__main__":
    Fire(CliEntryPoint)
