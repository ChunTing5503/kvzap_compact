# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from kvpress.presses.scorer_press import ScorerPress


def resolve_kvzap_model_name(model_type: str, model) -> str:
    return f"nvidia/KVzap-{model_type}-{model.config.name_or_path.split('/')[-1]}"


def prepare_kvzap_model_for_runtime(kvzap_model: "KVzapModel", model) -> "KVzapModel":
    device = getattr(model, "device", None)
    dtype = getattr(model, "dtype", None)

    if device is None or dtype is None:
        try:
            first_param = next(model.parameters())
        except (AttributeError, StopIteration):
            first_param = None
        if first_param is not None:
            device = device or first_param.device
            dtype = dtype or first_param.dtype

    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device
    if dtype is not None:
        to_kwargs["dtype"] = dtype
    if to_kwargs:
        kvzap_model = kvzap_model.to(**to_kwargs)
    return kvzap_model.eval()


class KVzapConfig(PretrainedConfig):
    model_type = "kvzap"
    has_no_defaults_at_init = True

    def __init__(
        self,
        input_dim: int = 0,
        output_dim: int = 0,
        n_modules: int = 0,
        hidden_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_modules = n_modules
        self.hidden_dim = hidden_dim


class KVzapModel(PreTrainedModel):
    config_class = KVzapConfig  # type: ignore[assignment]

    def __init__(self, config):
        super().__init__(config)
        self.all_tied_weights_keys = {}
        if config.hidden_dim is None:
            # Linear model
            self.layers = nn.ModuleList(
                [nn.Linear(config.input_dim, config.output_dim) for _ in range(config.n_modules)]
            )
        else:
            # 2-layer MLP model
            self.layers = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(config.input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim, config.output_dim),
                )
                for _ in range(config.n_modules)
            )

    def forward(self, x):
        return torch.stack([module(x[:, i, :]) for i, module in enumerate(self.layers)], dim=1)


@dataclass
class KVzapPress(ScorerPress):
    """
    KVzap (https://arxiv.org/abs/2601.07891) is a fast approximation of KVzip that works
    in both prefilling and decoding. It applies a lightweight surrogate model to the hidden
    states to predict importance scores for every KV pair.
    KVzapPress is designed to be used in conjunction with the DMSPress
    model_type can be "linear" or "mlp".
    """

    model_type: Literal["linear", "mlp"] = "mlp"
    kvzap_model_name: Optional[str] = field(default=None, init=False)

    def post_init_from_model(self, model):
        kvzap_model_name = resolve_kvzap_model_name(self.model_type, model)
        if kvzap_model_name != self.kvzap_model_name:
            self.kvzap_model_name = kvzap_model_name
            self.kvzap_model = KVzapModel.from_pretrained(self.kvzap_model_name)
        self.kvzap_model = prepare_kvzap_model_for_runtime(self.kvzap_model, model)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        kvzap_module = self.kvzap_model.layers[module.layer_idx]
        scores = kvzap_module(hidden_states).transpose(1, 2)
        return scores
