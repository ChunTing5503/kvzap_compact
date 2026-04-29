# Evaluation

This directory was reduced to the final-project comparison path:

- official KVzap logical masking: `kvzap_mlp`
- upstream-style KVzip baseline: `kvzip`
- streaming physical compaction: `kvzap_mlp_streaming_compact`
- optional references: `no_press`, `kvzap_mlp_compact`

Supported datasets in the cleaned repo:

- `aime25`
- `ruler`
- `scbench_kv`

## Setup

Activate your conda environment first:

```bash
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate "$CONDA_ENV"
```

Then install the retained evaluation dependencies into that environment:

```bash
python -m pip install -e ".[eval,train]"
```

Build the upstream KVzip runtime once:

```bash
cd _upstream/KVzip/csrc
python build.py install
cd ../..
```

## Entry Points

- `evaluate.py`: generic evaluation runner
- `evaluate_registry.py`: retained methods and datasets
- `evaluate.sh`: SLURM-friendly wrapper with optional threshold sweeps
- `configs/`: project configs for `Qwen/Qwen3-8B`

## Quick Start

Single run from a config:

```bash
python evaluation/evaluate.py \
  --config_file evaluation/configs/eval_scbench_kv_qwen3_8b_kvzap_mlp_streaming_compact.yaml
```

Override thresholds or compression ratios from the CLI:

```bash
python evaluation/evaluate.py \
  --config_file evaluation/configs/eval_ruler_qwen3_8b_kvzap_mlp.yaml \
  --threshold -4.0
```

```bash
python evaluation/evaluate.py \
  --config_file evaluation/configs/eval_ruler_qwen3_8b_kvzip.yaml \
  --compression_ratio 0.60
```

Outputs are written under `evaluation/results/` by default. The evaluator records:

- predictions and task metrics
- end-to-end runtime
- split prefill/decode runtime
- peak memory
- retained KV payload and cache length statistics

## Config Conventions

- `kvzap_*` configs use `threshold`
- `kvzip` configs use `compression_ratio`
- compacted-runtime configs require CUDA, FlashAttention, and `model_kwargs.dtype: "float16"`
- AIME25 configs enable sampling to mirror the dedicated `kvzap/evaluate_aime.py` setup

## Benchmark Files

- `benchmarks/aime25/calculate_metrics.py`
- `benchmarks/ruler/calculate_metrics.py`
- `benchmarks/scbench/load.py`
- `benchmarks/scbench/calculate_metrics.py`
