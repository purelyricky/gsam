"""End-to-end GSAM and ACE runs against the offline mock LLM provider."""

import json
import os
import tempfile
import unittest

from tests import helpers  # noqa: F401  (forces the offline embedding path)
from gsam.gsam import GSAM
from ace import ACE
from utils import initialize_clients


class StubProcessor:
    """Minimal data-processor interface used by the orchestrators."""

    def answer_is_correct(self, predicted, ground_truth, **kwargs):
        return str(predicted).strip().lower() == str(ground_truth).strip().lower()

    def evaluate_accuracy(self, predictions, targets):
        hits = sum(self.answer_is_correct(p, t) for p, t in zip(predictions, targets))
        return hits / len(targets) if targets else 0.0


def samples(n=4):
    return [
        {"context": f"context {i}", "question": f"question {i}", "target": "mock" if i % 2 == 0 else "other",
         "others": {"task": "stub"}}
        for i in range(n)
    ]


class TestMockProvider(unittest.TestCase):
    def test_initialize_clients_mock_needs_no_key(self):
        gen, ref, cur = initialize_clients("mock")
        reply = gen.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(json.loads(reply.choices[0].message.content)["final_answer"], "mock")

    def test_unknown_provider_rejected(self):
        with self.assertRaises(ValueError):
            initialize_clients("nope")


class TestGSAMEndToEnd(unittest.TestCase):
    def _gsam(self):
        return GSAM(api_provider="mock", generator_model="m", reflector_model="m", curator_model="m")

    def _config(self, tmp, **extra):
        cfg = {"task_name": "stub", "save_dir": tmp, "num_epochs": 1, "max_num_rounds": 1,
               "curator_frequency": 1, "eval_steps": 2, "save_steps": 2, "test_workers": 2,
               "online_eval_frequency": 2}
        cfg.update(extra)
        return cfg

    def test_eval_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._gsam().run("eval_only", test_samples=samples(), data_processor=StubProcessor(),
                                   config=self._config(tmp))
            self.assertAlmostEqual(out["test_results"]["accuracy"], 0.5)

    def test_offline_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._gsam().run("offline", train_samples=samples(), val_samples=samples(2),
                                   test_samples=samples(2), data_processor=StubProcessor(),
                                   config=self._config(tmp))
            self.assertIn("training_results", out)
            self.assertIn("final_test_results", out)

    def test_online_training_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._gsam().run("online", test_samples=samples(), data_processor=StubProcessor(),
                                   config=self._config(tmp))
            self.assertIn("online_test_results", out)
            run_dirs = [d for d in os.listdir(tmp) if d.startswith("gsam_run_")]
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue(os.path.exists(os.path.join(tmp, run_dirs[0], "final_results.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, run_dirs[0], "tracking.jsonl")))

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            self._gsam().run("bogus")


class TestACEEndToEnd(unittest.TestCase):
    def _ace(self):
        return ACE(api_provider="mock", generator_model="m", reflector_model="m", curator_model="m",
                   max_tokens=256)

    def _config(self, tmp):
        return {"task_name": "stub", "save_dir": tmp, "test_workers": 2, "num_epochs": 1, "max_num_rounds": 1,
                "curator_frequency": 1, "eval_steps": 2, "save_steps": 2, "online_eval_frequency": 2}

    def test_eval_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._ace().run("eval_only", test_samples=samples(), data_processor=StubProcessor(),
                                  config=self._config(tmp))
            self.assertIn("test_results", out)

    def test_offline_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._ace().run("offline", train_samples=samples(), val_samples=samples(2),
                                  test_samples=samples(2), data_processor=StubProcessor(),
                                  config=self._config(tmp))
            self.assertIn("training_results", out)

    def test_online_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._ace().run("online", test_samples=samples(), data_processor=StubProcessor(),
                                  config=self._config(tmp))
            self.assertIn("online_test_results", out)


class TestGSAMAblations(unittest.TestCase):
    def test_ablation_flags_construct(self):
        for flags in ({"no_failure_cascades": True}, {"untyped_edges": True}, {"embedding_only_retrieval": True}):
            g = GSAM(api_provider="mock", generator_model="m", reflector_model="m", curator_model="m", **flags)
            self.assertIsNotNone(g.graph_retriever)

    def test_taxonomy_initialises_concepts(self):
        path = os.path.join(os.path.dirname(__file__), "..", "eval", "finance", "data", "xbrl_taxonomy.json")
        if not os.path.exists(path):
            self.skipTest("taxonomy file not present")
        g = GSAM(api_provider="mock", generator_model="m", reflector_model="m", curator_model="m",
                 taxonomy_path=path)
        self.assertGreater(len(g.entity_name_map), 0)


if __name__ == "__main__":
    unittest.main()
