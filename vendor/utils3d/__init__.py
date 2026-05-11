"""
A minimal vendored compatibility surface for the official TRELLIS utils3d API.

The live extension previously vendored the unrelated PyPI package `utils3d`
(version 0.1.3 by Kalash Jain), which only exposes `pctodepthimage` and does
not provide the `utils3d.torch` namespace TRELLIS expects.

For the native TRELLIS text-to-mesh path we only need the torch helpers, so the
vendored package exposes that official surface lazily here.
"""

import importlib
from typing import TYPE_CHECKING

__all__ = ["torch"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    return importlib.import_module(f".{name}", __package__)


if TYPE_CHECKING:
    from . import torch
