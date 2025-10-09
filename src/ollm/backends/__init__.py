from .base import (
    Backend,
    BackendError,
    BackendSelection,
    get_backend,
    list_backends,
    register_backend,
    select_backend,
)
from .torch_backend import TorchBackend
from .mlx_backend import MLXBackend

# Register built-in backends
register_backend(TorchBackend())
register_backend(MLXBackend())

__all__ = [
    "Backend",
    "BackendError",
    "BackendSelection",
    "get_backend",
    "list_backends",
    "register_backend",
    "select_backend",
]
