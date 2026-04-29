## KVzap Streaming Compaction

This repository contains the code for a course project in *Introduction to High Performance Machine Learning*. It is adapted from NVIDIA's [`kvpress`](https://github.com/NVIDIA/kvpress) codebase and narrowed to one specific research question: whether KVzap can realize its expected memory and throughput gains when pruned KV-cache entries are physically compacted instead of only being logically masked.

The main contribution in this repository is `kvzap_streaming_compact`, a decode-time KV-cache compaction method that preserves the original KVzap keep/drop semantics while physically evicting mature tokens into a compacted runtime representation.

## Project Scope

The cleaned repository keeps only the components needed to study and reproduce the final comparison among:

- the official KVzap baseline with logical masking
- upstream KVzip as the physical-compaction reference
- the proposed `kvzap_streaming_compact` method

It also retains the code required to train or reuse KVzap surrogate scoring models and the benchmark/evaluation scripts needed for the final project experiments.

## Attribution and Provenance

This project is based on and adapted from NVIDIA's `kvpress` repository.

- Original upstream framework: `kvpress`
- Retained upstream dependency: `_upstream/KVzip`
- Project-specific additions: compacted runtime integration, `KVzapStreamingCompactPress`, cleaned evaluation configs, and repository reorganization for the final project

The repository is not a drop-in mirror of upstream `kvpress`. Unrelated methods, experiments, notebooks, and benchmark paths were removed so the codebase reflects only the final project story.

## Repository Structure

```text
.
├── kvpress/                 Adapted subset of NVIDIA kvpress
├── kvzap/                   KVzap surrogate training and AIME helper scripts
├── evaluation/              Retained benchmark loaders, scorers, configs, and runners
├── _upstream/KVzip/         Official KVzip implementation and runtime source
└── KVZAP_STREAMING_COMPACTION.md
```

Key retained modules:

- `kvpress/presses/kvzap_press.py`: official KVzap surrogate scorer
- `kvpress/presses/dms_press.py`: official logical-masking KVzap baseline
- `kvpress/presses/kvzip_press.py`: KVzip integration for comparison
- `kvpress/presses/kvzap_compact_press.py`: prefill-only physical compaction ablation
- `kvpress/presses/kvzap_streaming_compact_press.py`: main proposed method
- `kvpress/compacted_cache.py` and `kvpress/compacted_attention.py`: compacted cache runtime used by KVzip and streaming compact KVzap

## Methods Compared

- `no_press`: uncompressed reference
- `kvzap_mlp`: official KVzap with `DMSPress`, using logical masking
- `kvzip`: upstream-style KVzip with physical compaction
- `kvzap_mlp_streaming_compact`: proposed streaming physical compaction method
- `kvzap_mlp_compact`: optional prefill-only compaction ablation

## Environment Setup

Activate the conda environment you want to use:

```bash
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate "$CONDA_ENV"
```

Install the retained project dependencies into that environment:

```bash
python -m pip install -e ".[eval,train]"
```

Build the upstream KVzip runtime used by the compacted cache:

```bash
cd _upstream/KVzip/csrc
python build.py install
cd ../..
```

Or run:

```bash
bash install.sh
```

Compatibility note:

- the retained code targets the upstream-compatible line `torch 2.3.x` + `transformers 4.51.x`
- if your environment resolves to `transformers>=5`, upgrade Torch to `>=2.4` or pin Transformers back to `<5`

## Running Experiments

The main evaluation entry point is:

```bash
python evaluation/evaluate.py --config_file <config>
```

Useful configs for the final project are under `evaluation/configs/`, including:

- `eval_scbench_kv_qwen3_8b_kvzap_mlp.yaml`
- `eval_scbench_kv_qwen3_8b_kvzip.yaml`
- `eval_scbench_kv_qwen3_8b_kvzap_mlp_streaming_compact.yaml`
- `eval_ruler_qwen3_8b_kvzap_mlp.yaml`
- `eval_ruler_qwen3_8b_kvzip.yaml`
- `eval_ruler_qwen3_8b_kvzap_mlp_streaming_compact.yaml`
- `eval_aime25_qwen3_8b_kvzap_mlp.yaml`
- `eval_aime25_qwen3_8b_kvzip.yaml`
- `eval_aime25_qwen3_8b_kvzap_mlp_streaming_compact.yaml`

Examples:

```bash
python evaluation/evaluate.py \
  --config_file evaluation/configs/eval_scbench_kv_qwen3_8b_kvzap_mlp_streaming_compact.yaml
```

```bash
python evaluation/evaluate.py \
  --config_file evaluation/configs/eval_ruler_qwen3_8b_kvzip.yaml \
  --compression_ratio 0.60
```

For AIME25 with sampling-based decoding:

```bash
python kvzap/evaluate_aime.py mlp --threshold -5 --model_name Qwen/Qwen3-8B
```

Additional details are in [evaluation/README.md](evaluation/README.md).

## Training KVzap Surrogates

`kvzap/train.py` extracts KVzip-style supervision from text samples and trains both linear and MLP surrogate models for KVzap.

Example:

```bash
python kvzap/train.py --model_name Qwen/Qwen3-8B --output_dir kvzap_runs/qwen3_8b_run1
```

Training requires the `train` dependencies declared in `pyproject.toml`.

## Notes for Reproduction

- `kvzap_streaming_compact` depends on the upstream `tiny_api_cuda` runtime from `_upstream/KVzip/csrc`
- compacted-runtime experiments require CUDA, FlashAttention, and `torch.float16`
- benchmark outputs are written under `evaluation/results/` by default
- the repository was intentionally cleaned to remove unrelated experiments, especially the failed segmented-streaming branch
