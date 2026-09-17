import tempfile
import unittest
from pathlib import Path

from proctor.inference import missing_models, select_device
from proctor.state import SessionStore


class DeploymentTests(unittest.TestCase):
    def test_cpu_device_is_explicit(self):
        self.assertEqual(select_device("cpu"), "cpu")

    def test_missing_model_diagnostics_are_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = missing_models(Path(directory))
            self.assertTrue(any("yolov8n.pt" in item for item in missing))
            self.assertTrue(any("best_20.pt" in item for item in missing))

    def test_sessions_are_owned_and_isolated(self):
        store = SessionStore()
        first = store.start("one", "alice")
        second = store.start("two", "bob")
        first.latest_result["faces"] = 1
        self.assertIs(store.get_owned("one", "alice"), first)
        self.assertIsNone(store.get_owned("one", "bob"))
        self.assertNotEqual(store.get_owned("two", "bob").latest_result, first.latest_result)


if __name__ == "__main__":
    unittest.main()
