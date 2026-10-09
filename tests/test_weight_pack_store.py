import hashlib
from pathlib import Path
import tempfile
import unittest

from lokahi.core import ExecutionPlan, ModelSpec, TensorRef, WeightGroup
from lokahi.scheduling import DenseLayerPipeline, PrefetchScheduler, ResidencyManager
from lokahi.storage import (
    ImmutableWeightBytes,
    WeightPackChecksumError,
    WeightPackGroupTensorStore,
    WeightPackStoreError,
    write_weight_pack,
)


def _group(index, storage_key, *, nbytes=4, logical_name="weight"):
    return WeightGroup(
        group_id=f"layer.{index}",
        order=index,
        tensors=(
            TensorRef(
                name=logical_name,
                storage_key=storage_key,
                shape=(nbytes,),
                dtype="uint8",
                storage_nbytes=nbytes,
            ),
        ),
    )


class WeightPackGroupTensorStoreTest(unittest.TestCase):
    def test_real_pipeline_prefetches_and_evicts_packed_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.lokahi-pack"
            packed = [
                ("packed.layer.0", (1).to_bytes(4, "little")),
                ("packed.layer.1", (2).to_bytes(4, "little")),
                ("packed.layer.2", (3).to_bytes(4, "little")),
            ]
            write_weight_pack(path, packed, alignment=64)
            groups = tuple(
                _group(index, f"packed.layer.{index}") for index in range(3)
            )
            spec = ModelSpec.from_groups("packed-tiny", groups)
            evicted = []
            residency = ResidencyManager(
                8,
                on_evict=lambda key, value: evicted.append(key),
            )
            store = WeightPackGroupTensorStore(path)
            scheduler = PrefetchScheduler(store, residency)
            pipeline = DenseLayerPipeline(
                spec,
                ExecutionPlan.dense(spec),
                scheduler,
            )

            seen = []

            def execute(state, group, weights):
                value = weights["weight"]
                self.assertIsInstance(value, ImmutableWeightBytes)
                self.assertEqual(value.nbytes, 4)
                view = value.view
                try:
                    self.assertTrue(view.readonly)
                finally:
                    view.release()
                seen.append(group.group_id)
                return state + int.from_bytes(bytes(value), "little")

            try:
                result = pipeline.run(0, execute, phase="prefill")
            finally:
                pipeline.close()
                store.close()

            self.assertEqual(result, 6)
            self.assertEqual(seen, ["layer.0", "layer.1", "layer.2"])
            scheduler_snapshot = scheduler.snapshot()
            self.assertEqual(scheduler_snapshot.cold_misses, 1)
            self.assertEqual(
                scheduler_snapshot.prefetch_ready_hits
                + scheduler_snapshot.prefetch_waits,
                2,
            )
            self.assertEqual(scheduler_snapshot.bytes_loaded, 12)
            residency_snapshot = residency.snapshot()
            self.assertEqual(residency_snapshot.used_bytes, 8)
            self.assertEqual(set(residency_snapshot.resident_keys), {"layer.1", "layer.2"})
            self.assertIn("layer.0", evicted)

    def test_checksum_failure_releases_scheduler_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checksum.lokahi-pack"
            write_weight_pack(path, [("packed.weight", b"good")], alignment=64)
            store = WeightPackGroupTensorStore(path)
            record = store.pack.tensor_info("packed.weight")
            with path.open("r+b") as output:
                output.seek(record.offset)
                output.write(b"X")

            group = _group(0, "packed.weight")
            residency = ResidencyManager(4)
            scheduler = PrefetchScheduler(store, residency)
            try:
                with self.assertRaises(WeightPackChecksumError):
                    scheduler.acquire(group, phase="decode")
            finally:
                scheduler.close()
                store.close()

            snapshot = residency.snapshot()
            self.assertEqual(snapshot.used_bytes, 0)
            self.assertEqual(snapshot.loading_keys, ())
            self.assertEqual(snapshot.resident_keys, ())

    def test_size_mismatch_releases_scheduler_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "size.lokahi-pack"
            write_weight_pack(path, [("packed.weight", b"abc")], alignment=64)
            store = WeightPackGroupTensorStore(path)
            group = _group(0, "packed.weight", nbytes=4)
            residency = ResidencyManager(4)
            scheduler = PrefetchScheduler(store, residency)
            try:
                with self.assertRaisesRegex(WeightPackStoreError, "TensorRef.nbytes"):
                    scheduler.acquire(group, phase="decode")
            finally:
                scheduler.close()
                store.close()

            snapshot = residency.snapshot()
            self.assertEqual(snapshot.used_bytes, 0)
            self.assertEqual(snapshot.loading_keys, ())

    def test_duplicate_storage_keys_fail_before_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.lokahi-pack"
            write_weight_pack(path, [("shared", b"abcd")], alignment=64)
            decoder_calls = []

            def decoder(tensor_ref, payload):
                decoder_calls.append((tensor_ref, payload))
                return payload

            store = WeightPackGroupTensorStore(path, decoder=decoder)
            group = WeightGroup(
                group_id="duplicates",
                order=0,
                tensors=(
                    TensorRef("first", (4,), "uint8", "shared"),
                    TensorRef("second", (4,), "uint8", "shared"),
                ),
            )
            with self.assertRaisesRegex(WeightPackStoreError, "duplicate storage keys"):
                store.load_group(group)
            self.assertEqual(decoder_calls, [])
            store.close()

    def test_custom_decoder_receives_read_only_exact_bytes_and_close_is_final(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decoder.lokahi-pack"
            payload = b"decoder input"
            write_weight_pack(path, [("packed", payload)], alignment=64)
            observations = []

            def decoder(tensor_ref, data):
                observations.append(
                    (
                        tensor_ref.key,
                        isinstance(data, bytes),
                        memoryview(data).readonly,
                        hashlib.sha256(data).hexdigest(),
                    )
                )
                return ImmutableWeightBytes(data)

            store = WeightPackGroupTensorStore(path, decoder=decoder)
            group = _group(0, "packed", nbytes=len(payload), logical_name="decoded")
            loaded = store.load_group(group)
            self.assertEqual(bytes(loaded["decoded"]), payload)
            self.assertEqual(
                observations,
                [("packed", True, True, hashlib.sha256(payload).hexdigest())],
            )
            store.close()
            self.assertTrue(store.closed)
            with self.assertRaisesRegex(WeightPackStoreError, "closed"):
                store.load_group(group)


if __name__ == "__main__":
    unittest.main()
