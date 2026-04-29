# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from importlib.metadata import PackageNotFoundError, version
from typing import Optional


_MIN_TORCH_FOR_TRANSFORMERS_5 = (2, 4, 0)


def _parse_release(version_str: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version_str)
    if match is None:
        return (0, 0, 0)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def get_core_runtime_incompatibility(
    torch_version: Optional[str] = None,
    transformers_version: Optional[str] = None,
) -> Optional[str]:
    """Returns an actionable error message when the installed core runtime is incompatible."""
    missing_packages: list[str] = []
    if torch_version is None:
        try:
            torch_version = version("torch")
        except PackageNotFoundError:
            missing_packages.append("torch")
    if transformers_version is None:
        try:
            transformers_version = version("transformers")
        except PackageNotFoundError:
            missing_packages.append("transformers")

    if missing_packages:
        missing = ", ".join(sorted(missing_packages))
        return (
            f"Missing required package(s): {missing}. "
            "Activate your conda environment and install the project dependencies with "
            "`python -m pip install -e \".[eval,train]\"`."
        )

    assert torch_version is not None
    assert transformers_version is not None

    if _parse_release(transformers_version)[0] >= 5 and _parse_release(torch_version) < _MIN_TORCH_FOR_TRANSFORMERS_5:
        return (
            "Incompatible environment detected: "
            f"transformers {transformers_version} requires torch>=2.4, "
            f"but torch {torch_version} is installed. "
            "Either upgrade torch, or pin transformers back to the upstream-compatible line, for example with "
            "`python -m pip install --upgrade 'torch>=2.3.1,<3' 'transformers>=4.51.3,<5'`."
        )

    return None


def assert_core_runtime_requirements():
    """Raises an actionable RuntimeError when the installed core runtime is incompatible."""
    incompatibility = get_core_runtime_incompatibility()
    if incompatibility is not None:
        raise RuntimeError(incompatibility)
