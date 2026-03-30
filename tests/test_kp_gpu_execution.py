from __future__ import annotations

import unittest

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
