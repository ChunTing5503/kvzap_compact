# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "runtime_requirements.py"
_SPEC = importlib.util.spec_from_file_location("evaluation_runtime_requirements", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
runtime_requirements = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runtime_requirements)


def test_detects_transformers5_with_old_torch():
    message = runtime_requirements.get_core_runtime_incompatibility(
        torch_version="2.3.1+cu118",
        transformers_version="5.5.0",
    )

    assert message is not None
    assert "transformers 5.5.0 requires torch>=2.4" in message
    assert "torch 2.3.1+cu118 is installed" in message
    assert "transformers>=4.51.3,<5" in message


def test_allows_upstream_kvzip_compatible_versions():
    message = runtime_requirements.get_core_runtime_incompatibility(
        torch_version="2.3.1+cu118",
        transformers_version="4.51.3",
    )

    assert message is None


def test_allows_transformers5_with_supported_torch():
    message = runtime_requirements.get_core_runtime_incompatibility(
        torch_version="2.4.0",
        transformers_version="5.5.0",
    )

    assert message is None


def test_assert_core_runtime_requirements_raises_actionable_error(monkeypatch):
    versions = {"torch": "2.3.1+cu118", "transformers": "5.5.0"}
    monkeypatch.setattr(runtime_requirements, "version", lambda name: versions[name])

    with pytest.raises(RuntimeError, match="requires torch>=2.4"):
        runtime_requirements.assert_core_runtime_requirements()
