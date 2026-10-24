"""Guards the module split of the orchestrators (agents, training mixins)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ace.ace import ACE  # noqa: E402
from ace.training_modes import OfflineTrainingMixin  # noqa: E402
from ace.online_training import OnlineTrainingMixin  # noqa: E402
from ace.sample_training import SampleTrainingMixin  # noqa: E402
from gsam import gsam as gsam_module  # noqa: E402
from gsam.core import agents  # noqa: E402
from gsam.sample_training import SampleTrainingMixin as GSAMSampleMixin  # noqa: E402


class TestModuleStructure(unittest.TestCase):
    def test_ace_composes_training_mixins(self):
        self.assertTrue(issubclass(ACE, OfflineTrainingMixin))
        self.assertTrue(issubclass(ACE, OnlineTrainingMixin))
        self.assertTrue(issubclass(ACE, SampleTrainingMixin))
        for name in ("_offline_train", "_online_train_and_test", "_train_single_sample", "run", "test"):
            self.assertTrue(callable(getattr(ACE, name)), name)

    def test_gsam_composes_sample_mixin(self):
        self.assertTrue(issubclass(gsam_module.GSAM, GSAMSampleMixin))
        self.assertTrue(callable(gsam_module.GSAM._train_single_sample))

    def test_gsam_reexports_agents(self):
        for name in ("GSAMGenerator", "GSAMReflector", "GSAMCurator"):
            self.assertIs(getattr(gsam_module, name), getattr(agents, name))

    def test_retrieval_wrapper_replaces_playbook_with_graph_context(self):
        class FakeRetriever:
            def retrieve(self, query, context=""):
                return "GRAPH-CTX", ["S:0001"]

        class FakeGen:
            def generate(self, **kwargs):
                return kwargs

        wrapper = agents._RetrievalAwareGenerator(FakeGen(), FakeRetriever())
        out = wrapper.generate(question="q", playbook="stale", context="c")
        self.assertEqual(out["graph_context"], "GRAPH-CTX")
        self.assertNotIn("playbook", out)


if __name__ == "__main__":
    unittest.main()
