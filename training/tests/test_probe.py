import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("training_probe", Path(__file__).parents[1] / "probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class CudaBuildTests(unittest.TestCase):
    def test_cuda_constant_does_not_execute_module(self):
        self.assertTrue(probe.cuda_build_from_source("raise RuntimeError('must not execute')\ncuda: str = '12.8'", "2.9.1"))

    def test_cpu_constant(self):
        self.assertFalse(probe.cuda_build_from_source("cuda = None", "2.9.1"))

    def test_version_fallback(self):
        self.assertFalse(probe.cuda_build_from_source("", "2.9.1+cpu"))
        self.assertTrue(probe.cuda_build_from_source("", "2.9.1+cu128"))
        self.assertIsNone(probe.cuda_build_from_source("", "2.9.1"))


if __name__ == "__main__":
    unittest.main()
