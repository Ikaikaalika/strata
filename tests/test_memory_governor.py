import threading
import unittest

from ollm.backends.mlx_governor import (
    detect_mlx_hardware,
    mlx_capabilities,
    suggested_residency_budget,
)
from ollm.core import ExecutionPlan, ModelSpec, TensorRef, WeightGroup
from ollm.scheduling import (
    BudgetExceededError,
    DenseLayerPipeline,
    PrefetchScheduler,
    ResidencyManager,
)


class FakeTensor:
    def __init__(self, value, nbytes=4):
        self.value = value
        self.nbytes = nbytes


class ControlledStore:
    def __init__(self):
        self.loads = []
        self.started = {}
        self.allowed = {}
        self._lock = threading.Lock()

    def control(self, group_id):
        self.started[group_id] = threading.Event()
        self.allowed[group_id] = threading.Event()

    def load_group(self, group):
        with self._lock:
            self.loads.append(group.group_id)
        if group.group_id in self.started:
            self.started[group.group_id].set()
            if not self.allowed[group.group_id].wait(timeout=2):
                raise TimeoutError(f"test never released {group.group_id}")
        return {group.tensors[0].name: FakeTensor(group.order + 1)}


def group(index, nbytes=4):
    return WeightGroup(
        group_id=f"layer.{index}",
        order=index,
        tensors=(
            TensorRef(
                name="weight",
                shape=(1,),
                dtype="int8",
                storage_nbytes=nbytes,
            ),
        ),
    )


class RuntimeContractTest(unittest.TestCase):
    def test_dense_plan_declares_exact_next_layer_prefetch(self):
        spec = ModelSpec.from_groups("tiny", [group(2), group(0), group(1)])
        plan = ExecutionPlan.dense(spec, prefetch_distance=1)

        self.assertEqual([step.group_id for step in plan.steps], ["layer.0", "layer.1", "layer.2"])
        self.assertEqual(plan.steps[0].prefetch_group_ids, ("layer.1",))
        self.assertEqual(plan.steps[1].prefetch_group_ids, ("layer.2",))
        self.assertEqual(plan.steps[2].prefetch_group_ids, ())
        self.assertEqual(spec.total_weight_bytes, 12)

    def test_residency_never_evicts_a_pinned_group(self):
        evicted = []
        residency = ResidencyManager(8, on_evict=lambda key, value: evicted.append(key))
        self.assertTrue(residency.reserve("a", 4))
        residency.commit("a", {"value": "a"})
        residency.pin("a")
        self.assertTrue(residency.reserve("b", 4))
        residency.commit("b", {"value": "b"})

        self.assertTrue(residency.reserve("c", 4))
        residency.commit("c", {"value": "c"})

        snapshot = residency.snapshot()
        self.assertIn("a", snapshot.resident_keys)
        self.assertIn("c", snapshot.resident_keys)
        self.assertNotIn("b", snapshot.resident_keys)
        self.assertEqual(evicted, ["b"])
        self.assertLessEqual(snapshot.used_bytes, snapshot.budget_bytes)
        residency.release("a")

    def test_residency_budget_shrink_evicts_oldest_unpinned_groups(self):
        evicted = []
        residency = ResidencyManager(
            12,
            on_evict=lambda key, value: evicted.append(key),
        )
        for key in ("a", "b", "c"):
            self.assertTrue(residency.reserve(key, 4))
            residency.commit(key, {"value": key})
        residency.pin("a")
        residency.release("a")

        removed = residency.resize_budget(8)

        self.assertEqual(removed, ("b",))
        self.assertEqual(evicted, ["b"])
        snapshot = residency.snapshot()
        self.assertEqual(snapshot.budget_bytes, 8)
        self.assertEqual(snapshot.resident_keys, ("a", "c"))

    def test_residency_budget_shrink_refuses_to_evict_pinned_groups(self):
        evicted = []
        residency = ResidencyManager(
            8,
            on_evict=lambda key, value: evicted.append(key),
        )
        for key in ("a", "b"):
            self.assertTrue(residency.reserve(key, 4))
            residency.commit(key, {"value": key})
            residency.pin(key)

        before = residency.snapshot()
        with self.assertRaisesRegex(BudgetExceededError, "pinned"):
            residency.resize_budget(4)
        after = residency.snapshot()

        self.assertEqual(after, before)
        self.assertEqual(evicted, [])
        residency.release("a")
        residency.release("b")

    def test_pipeline_starts_next_load_before_current_compute(self):
        groups = tuple(group(index) for index in range(3))
        spec = ModelSpec("tiny", groups)
        plan = ExecutionPlan.dense(spec)
        store = ControlledStore()
        store.control("layer.1")
        residency = ResidencyManager(8)
        scheduler = PrefetchScheduler(store, residency)
        pipeline = DenseLayerPipeline(spec, plan, scheduler)

        def execute_layer(state, current_group, weights):
            if current_group.group_id == "layer.0":
                self.assertTrue(store.started["layer.1"].wait(timeout=1))
                store.allowed["layer.1"].set()
            return state + weights["weight"].value

        try:
            result = pipeline.run(0, execute_layer, phase="decode")
        finally:
            pipeline.close()

        self.assertEqual(result, 6)
        self.assertEqual(store.loads, ["layer.0", "layer.1", "layer.2"])
        scheduler_snapshot = scheduler.snapshot()
        self.assertEqual(scheduler_snapshot.cold_misses, 1)
        self.assertEqual(
            scheduler_snapshot.prefetch_ready_hits + scheduler_snapshot.prefetch_waits,
            2,
        )
        self.assertLessEqual(
            residency.snapshot().used_bytes,
            residency.snapshot().budget_bytes,
        )

    def test_single_group_budget_skips_double_buffering_but_still_runs(self):
        groups = (group(0), group(1))
        spec = ModelSpec("tiny", groups)
        scheduler = PrefetchScheduler(ControlledStore(), ResidencyManager(4))
        pipeline = DenseLayerPipeline(spec, ExecutionPlan.dense(spec), scheduler)
        try:
            result = pipeline.run(
                0,
                lambda state, current_group, weights: state + weights["weight"].value,
                phase="prefill",
            )
        finally:
            pipeline.close()

        self.assertEqual(result, 3)
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.prefetch_skips, 1)
        self.assertEqual(snapshot.cold_misses, 2)


class MLXGovernorProfileTest(unittest.TestCase):
    def test_profile_and_capabilities_match_local_mlx_runtime(self):
        profile = detect_mlx_hardware()
        capabilities = mlx_capabilities()

        self.assertGreater(profile.unified_memory_bytes, 0)
        self.assertGreater(profile.recommended_working_set_bytes, 0)
        self.assertGreater(suggested_residency_budget(profile), 0)
        self.assertTrue(capabilities.supports_weight_paging)
        self.assertTrue(capabilities.supports_async_prefetch)
        self.assertTrue(capabilities.supports_memory_counters)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
