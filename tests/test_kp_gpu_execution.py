from __future__ import annotations

import unittest

from kp_export.parallel import plan_cpu_affinity_assignments
from scripts.extract_keypoints import _resolve_execution_mode


class GpuExecutionPlannerTests(unittest.TestCase):
    def test_auto_prefers_gpu_single_for_tasks_gpu(self) -> None:
        self.assertEqual(
            _resolve_execution_mode("auto", mp_backend="tasks", mp_tasks_delegate="gpu"),
            "gpu_single",
        )

    def test_auto_uses_cpu_pool_for_non_gpu_configs(self) -> None:
        self.assertEqual(
            _resolve_execution_mode("auto", mp_backend="tasks", mp_tasks_delegate="cpu"),
            "cpu_pool",
        )
        self.assertEqual(
            _resolve_execution_mode("auto", mp_backend="solutions", mp_tasks_delegate="gpu"),
            "cpu_pool",
        )

    def test_gpu_single_requires_tasks_gpu(self) -> None:
        with self.assertRaises(SystemExit):
            _resolve_execution_mode("gpu_single", mp_backend="tasks", mp_tasks_delegate="cpu")

    def test_cpu_pool_is_rejected_for_tasks_gpu(self) -> None:
        with self.assertRaises(SystemExit):
            _resolve_execution_mode("cpu_pool", mp_backend="tasks", mp_tasks_delegate="gpu")


class CpuAffinityPlanningTests(unittest.TestCase):
    def test_balances_assignments_across_workers(self) -> None:
        assignments = plan_cpu_affinity_assignments(list(range(8)), 6)
        self.assertEqual(len(assignments), 6)
        sizes = [len(item) for item in assignments]
        self.assertEqual(sorted(cpu for item in assignments for cpu in item), list(range(8)))
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_caps_assignments_to_available_cpus(self) -> None:
        assignments = plan_cpu_affinity_assignments(list(range(4)), 12)
        self.assertEqual(len(assignments), 4)
        self.assertTrue(all(len(item) == 1 for item in assignments))
