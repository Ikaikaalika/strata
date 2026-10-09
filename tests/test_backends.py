import importlib.util
import unittest

try:
    from lokahi.backends import BackendError, get_backend, list_backends, select_backend
    _BACKEND_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    BackendError = None
    select_backend = None
    get_backend = None
    list_backends = None
    _BACKEND_IMPORT_ERROR = exc

MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None
requires_mlx = unittest.skipUnless(MLX_AVAILABLE, "MLX is required")


@unittest.skipIf(
    _BACKEND_IMPORT_ERROR is not None,
    f"backend tests require optional deps: {_BACKEND_IMPORT_ERROR}",
)
class BackendSelectionTest(unittest.TestCase):
    @requires_mlx
    def test_default_backend(self):
        selection = select_backend()
        self.assertEqual(selection.backend.name, "mlx")

    @requires_mlx
    def test_mlx_cpu_prefix(self):
        selection = select_backend("mlx:cpu")
        self.assertEqual(selection.backend.name, "mlx")
        self.assertEqual(selection.device_request, "cpu")

    @requires_mlx
    def test_mlx_gpu_prefix(self):
        selection = select_backend("mlx:0")
        self.assertEqual(selection.backend.name, "mlx")
        self.assertEqual(selection.device_request, "0")

    @requires_mlx
    def test_plain_cpu_device_string(self):
        selection = select_backend("cpu")
        self.assertEqual(selection.backend.name, "mlx")
        self.assertEqual(selection.device_request, "cpu")

    def test_unknown_backend(self):
        with self.assertRaises(BackendError):
            select_backend("unknown:0")

    def test_mlx_backend_registration(self):
        backend = get_backend("mlx")
        if not backend.is_available():
            with self.assertRaises(BackendError):
                select_backend("mlx:0")
        else:
            selection = select_backend("mlx")
            self.assertEqual(selection.backend.name, "mlx")

    def test_builtin_backend_registry_includes_hardware_lanes(self):
        self.assertEqual(list_backends(), ("ane", "metal", "mlx"))
        self.assertEqual(get_backend("metal").name, "metal")
        self.assertEqual(get_backend("ane").name, "ane")

    @requires_mlx
    def test_attention_kernel_available(self):
        selection = select_backend("mlx")
        kernel = selection.backend.attention_kernel()
        self.assertTrue(callable(kernel))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
