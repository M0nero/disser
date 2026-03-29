from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESS_DIR = REPO_ROOT / "kp_export" / "process"
RUNNER_PATH = PROCESS_DIR / "pipeline" / "runner.py"


class ProcessLayoutTests(unittest.TestCase):
    def test_legacy_flat_modules_removed(self) -> None:
        disallowed = {
            "assign.py",
            "decode.py",
            "detect.py",
            "detectors.py",
            "orchestrator.py",
            "process_assignment.py",
            "process_constants.py",
            "process_geometry.py",
            "process_init.py",
            "process_metrics.py",
            "process_models.py",
            "process_occlusion.py",
            "process_pose.py",
            "process_second_pass.py",
            "recover.py",
            "reporting.py",
            "runtime.py",
            "summary.py",
        }
        present = {path.name for path in PROCESS_DIR.glob("*.py")}
        self.assertFalse(disallowed & present, f"legacy flat modules remain: {sorted(disallowed & present)}")

    def test_public_api_is_minimal(self) -> None:
        module = ast.parse((PROCESS_DIR / "__init__.py").read_text(encoding="utf-8"))
        exports = []
        for node in module.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        exports = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
        self.assertEqual(exports, ["process_video", "process_task"])

    def test_runner_has_no_nested_functions(self) -> None:
        module = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                nested = [child.name for child in ast.walk(node) if isinstance(child, ast.FunctionDef) and child is not node]
                self.assertFalse(nested, f"{node.name} contains nested functions: {nested}")
