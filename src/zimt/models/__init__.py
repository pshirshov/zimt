"""Model registry — see :mod:`zimt.models.registry` for the canonical dict."""

from .registry import MODELS
from .spec import ModelSpec

__all__ = ["MODELS", "ModelSpec"]
