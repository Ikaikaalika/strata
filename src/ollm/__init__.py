"""Compatibility package surface for Strata.

Keep optional, heavyweight runtime dependencies lazy. This lets low-level
modules such as ``ollm.backends`` and the MLX kernels being tested without
requiring the tokenizer/download stack to be imported first.
"""

from .utils import file_get_contents
from .tracing import TensorTracer, TraceEvent

__all__ = [
    "Inference",
    "TensorTracer",
    "TextStreamer",
    "TraceEvent",
    "file_get_contents",
]


def __getattr__(name):
    if name == "Inference":
        from .inference import Inference

        return Inference
    if name == "TextStreamer":
        from transformers import TextStreamer

        return TextStreamer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
