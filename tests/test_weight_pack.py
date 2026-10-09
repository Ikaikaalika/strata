import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from lokahi.storage.weight_pack import (
    FORMAT_NAME,
    FORMAT_VERSION,
    WeightPack,
    WeightPackChecksumError,
    WeightPackFormatError,
    WeightPackManifest,
    write_weight_pack,
)


def _tensor(name, offset, length, fill=b"x"):
    return {
        "name": name,
        "offset": offset,
        "length": length,
        "sha256": hashlib.sha256(fill * length).hexdigest(),
    }


def _manifest(tensors, *, alignment=64, data_offset=512, file_size=None, **extra):
    if file_size is None:
        final = tensors[-1]
        file_size = final["offset"] + final["length"]
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "alignment": alignment,
        "data_offset": data_offset,
        "file_size": file_size,
        "tensors": tensors,
    }
    payload.update(extra)
    return payload


class WeightPackTest(unittest.TestCase):
    def test_round_trip_exact_reads_and_mmap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.lokahi-pack"
            payloads = [("layer.0", b"abc" * 17), ("layer.1", bytes(range(64)))]
            written = write_weight_pack(path, payloads, alignment=256)

            pack = WeightPack(path)
            self.assertEqual(pack.manifest, written)
            self.assertEqual(pack.read_tensor("layer.0"), payloads[0][1])
            self.assertEqual(pack.read_tensor("layer.1"), payloads[1][1])
            self.assertEqual(pack.manifest.file_size, path.stat().st_size)
            for tensor in pack.manifest.tensors:
                self.assertEqual(tensor.offset % 256, 0)

            with pack.mmap_tensor("layer.1") as view:
                self.assertEqual(bytes(view), payloads[1][1])
                self.assertTrue(view.readonly)
            pack.verify_all()

    def test_failed_replacement_keeps_existing_pack_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recoverable.lokahi-pack"
            write_weight_pack(path, [("original", b"preserved")])

            with mock.patch(
                "lokahi.storage.weight_pack.os.replace",
                side_effect=OSError("simulated atomic replacement failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated atomic"):
                    write_weight_pack(path, [("replacement", b"not installed")])

            self.assertEqual(WeightPack(path).read_tensor("original"), b"preserved")
            self.assertEqual([item.name for item in Path(directory).iterdir()], [path.name])

    def test_payload_corruption_is_detected_on_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.lokahi-pack"
            write_weight_pack(path, [("weight", b"immutable bytes")], alignment=64)
            tensor = WeightPack(path).tensor_info("weight")
            with path.open("r+b") as output:
                output.seek(tensor.offset + 2)
                output.write(b"X")

            reopened = WeightPack(path)
            with self.assertRaisesRegex(WeightPackChecksumError, "checksum mismatch"):
                reopened.read_tensor("weight")

    def test_duplicate_overlap_and_bounds_are_rejected(self):
        valid_checksum = hashlib.sha256(b"x" * 64).hexdigest()
        cases = {
            "duplicate": _manifest(
                [
                    _tensor("same", 512, 64),
                    {
                        "name": "same",
                        "offset": 576,
                        "length": 64,
                        "sha256": valid_checksum,
                    },
                ],
                file_size=640,
            ),
            "overlaps": _manifest(
                [_tensor("first", 512, 64), _tensor("second", 544, 64)],
                file_size=608,
            ),
            "beyond declared file bounds": _manifest(
                [_tensor("outside", 512, 64)],
                file_size=540,
            ),
        }
        for message, payload in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(WeightPackFormatError, message):
                    WeightPackManifest.from_dict(payload)

    def test_manifest_schema_and_version_are_strict(self):
        tensor = _tensor("weight", 512, 64)
        with self.assertRaisesRegex(WeightPackFormatError, "unsupported.*version"):
            WeightPackManifest.from_dict(
                _manifest([tensor], version=FORMAT_VERSION + 1)
            )
        with self.assertRaisesRegex(WeightPackFormatError, "unsupported.*version"):
            WeightPackManifest.from_dict(_manifest([tensor], version=True))
        with self.assertRaisesRegex(WeightPackFormatError, "unknown=.*extra"):
            WeightPackManifest.from_dict(_manifest([tensor], extra="not allowed"))

    def test_truncated_file_is_rejected_before_tensor_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.lokahi-pack"
            write_weight_pack(path, [("weight", bytes(range(32)))], alignment=64)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(WeightPackFormatError, "physical file"):
                WeightPack(path)

    def test_writer_rejects_invalid_alignment_and_empty_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.lokahi-pack"
            with self.assertRaisesRegex(ValueError, "power of two"):
                write_weight_pack(path, [("weight", b"x")], alignment=96)
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                write_weight_pack(path, [("weight", b"")])
            with self.assertRaisesRegex(TypeError, "bytes-like"):
                write_weight_pack(path, [("weight", [1, 2, 3])])
            with self.assertRaisesRegex(ValueError, "duplicate tensor"):
                write_weight_pack(path, [("same", b"a"), ("same", b"b")])


if __name__ == "__main__":
    unittest.main()
