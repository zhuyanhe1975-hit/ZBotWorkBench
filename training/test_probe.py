import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from training import probe


class NativeRuntimeProbeTests(unittest.TestCase):
    def probe_with_native_result(self, result=None, error=None):
        versions = {"mjlab": "1.3.0", "rsl-rl-lib": "5.2.0",
                    "torch": "2.9.1+cu128", "warp-lang": "1.12.1"}
        with patch.object(probe.importlib.metadata, "version", side_effect=versions.__getitem__), \
             patch.object(probe.importlib.util, "find_spec", return_value=SimpleNamespace(origin=None)), \
             patch.object(probe.subprocess, "run", return_value=result, side_effect=error):
            return probe.probe()

    def test_blocked_native_dll_does_not_report_available(self):
        result = self.probe_with_native_result(SimpleNamespace(
            returncode=1, stderr="ImportError: DLL blocked by code integrity policy"))
        self.assertFalse(result["available"])
        self.assertIn("code integrity policy", result["error"])

    def test_native_probe_timeout_does_not_report_available(self):
        result = self.probe_with_native_result(error=subprocess.TimeoutExpired("python", 15))
        self.assertFalse(result["available"])
        self.assertIn("timed out", result["error"])

    def test_loadable_native_runtime_reports_available(self):
        result = self.probe_with_native_result(SimpleNamespace(returncode=0, stderr=""))
        self.assertTrue(result["available"])
        self.assertTrue(result["cudaBuild"])


if __name__ == "__main__":
    unittest.main()
