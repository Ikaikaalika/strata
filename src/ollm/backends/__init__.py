from .base import (
    Backend,
    BackendError,
    BackendSelection,
    get_backend,
    list_backends,
    register_backend,
    select_backend,
)
from .ane_backend import ANEBackend
from .ane_executor import ANEProjectionError, ANEProjectionExecutor, ANEProjectionReport
from .metal_backend import MetalBackend
from .mlx_backend import MLXBackend

# Register built-in backends
register_backend(MLXBackend())
register_backend(MetalBackend())
register_backend(ANEBackend())

__all__ = [
    "ANEBackend",
    "ANEProjectionError",
    "ANEProjectionExecutor",
    "ANEProjectionReport",
    "Backend",
    "BackendError",
    "BackendSelection",
    "MetalBackend",
    "get_backend",
    "list_backends",
    "register_backend",
    "select_backend",
]
