"""Backend abstraction primitives for oLLM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


class BackendError(RuntimeError):
    """Raised when a backend is unavailable or misconfigured."""


class Backend:
    """Abstract backend interface."""

    name: str = "base"

    def is_available(self) -> bool:
        return True

    def resolve_device(self, device_request: Optional[str]) -> Any:
        """Return a backend-specific device handle."""
        raise NotImplementedError

    def move_model_to_device(self, model: Any, device: Any) -> Any:
        """Place model on the resolved device."""
        raise NotImplementedError

    def create_kv_cache(self, cache_dir: str, stats: Optional[Any]):
        """Create a backend-native KV cache implementation."""
        raise NotImplementedError

    def attention_kernel(self):
        """Return callable implementing grouped attention for the backend."""
        raise NotImplementedError


@dataclass(frozen=True)
class BackendSelection:
    backend: Backend
    device_request: Optional[str]
    resolved_device: Any


_REGISTRY: Dict[str, Backend] = {}


def register_backend(backend: Backend) -> None:
    if backend.name in _REGISTRY:
        raise BackendError(f"Backend '{backend.name}' already registered")
    _REGISTRY[backend.name] = backend


def get_backend(name: str) -> Backend:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise BackendError(f"Backend '{name}' is not registered") from exc


def list_backends() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY.keys()))


def select_backend(device: Any = None) -> BackendSelection:
    """Resolve requested backend/device tuple."""
    backend_name: str
    device_request: Optional[str]

    backend_name, device_request = _parse_device_request(device)
    backend = get_backend(backend_name)
    if not backend.is_available():
        raise BackendError(f"Backend '{backend_name}' is not available on this system")
    resolved = backend.resolve_device(device_request)
    return BackendSelection(backend=backend, device_request=device_request, resolved_device=resolved)


def _parse_device_request(device: Any) -> Tuple[str, Optional[str]]:
    if device is None:
        return "torch", None

    if isinstance(device, str):
        prefix, device_suffix = _split_backend_prefix(device)
        return prefix, device_suffix

    # torch.device compatibility without importing torch explicitly
    type_name = type(device).__name__
    module_name = type(device).__module__
    if type_name == "device" and module_name.startswith("torch"):
        index = getattr(device, "index", None)
        device_type = getattr(device, "type", "cpu")
        device_str = device_type if index is None else f"{device_type}:{index}"
        return "torch", device_str

    raise BackendError(f"Unsupported device specification: {device!r}")


def _split_backend_prefix(raw: str) -> Tuple[str, Optional[str]]:
    if ":" not in raw:
        return "torch", raw
    prefix, remainder = raw.split(":", 1)
    if prefix in _REGISTRY:
        return prefix, remainder or None
    return "torch", raw
