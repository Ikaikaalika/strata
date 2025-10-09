import unittest

import torch

try:
    from ollm.backends import BackendError, select_backend, get_backend
    _BACKEND_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover - triggered only when deps missing
    BackendError = None
    select_backend = None
    _BACKEND_IMPORT_ERROR = exc


@unittest.skipIf(_BACKEND_IMPORT_ERROR is not None, f"backend tests require optional deps: {_BACKEND_IMPORT_ERROR}")
class BackendSelectionTest(unittest.TestCase):
    def test_default_backend(self):
        selection = select_backend()
        self.assertEqual(selection.backend.name, "torch")
        self.assertIsInstance(selection.resolved_device, torch.device)

    def test_torch_prefix(self):
        selection = select_backend("torch:cpu")
        self.assertEqual(selection.backend.name, "torch")
        self.assertEqual(str(selection.resolved_device), "cpu")

    def test_plain_torch_device_string(self):
        selection = select_backend("cuda:0")
        self.assertEqual(selection.backend.name, "torch")
        self.assertEqual(str(selection.resolved_device), "cuda:0")

    def test_torch_device_instance(self):
        selection = select_backend(torch.device("cpu"))
        self.assertEqual(selection.backend.name, "torch")
        self.assertEqual(str(selection.resolved_device), "cpu")

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

    def test_attention_kernel_available(self):
        selection = select_backend("torch")
        kernel = selection.backend.attention_kernel()
        self.assertTrue(callable(kernel))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
