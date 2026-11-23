"""Offline checks for the packaged algorithms and the read-only resource boundary."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from specalgo._support.graph_relations import apply_model_output
from specalgo._support.graph_store import GraphStore
from specalgo.attribution import graph_edit
from specalgo.common import cached_input, digest, read_json, work_root, write_json
from specalgo.contracts import Algorithm1Request, Algorithm2Request
from specalgo.data import evaluator_class


class PortabilityTests(unittest.TestCase):
    def test_request_contracts(self):
        request = Algorithm1Request(
            operation="retrieve", query="Find an applicable rule."
        )
        self.assertEqual(request.top_k, 5)
        request = Algorithm2Request(
            case_ids=["example"],
            operations=["remove_content", "remove_edge", "rewire_edge", "restore"],
        )
        self.assertEqual(len(request.operations), 4)
        with self.assertRaises(ValueError):
            Algorithm1Request(operation="retrieve", top_k=0)

    def test_embedded_scorer(self):
        evaluator, module = evaluator_class()
        self.assertTrue(module.JUDGE_CONTRACT.strip())
        self.assertTrue(module.SPEC_BENCH_INSTRUCTIONS.strip())
        gold = {"case_id": "example", "evidence": {}}
        item = {"gold_spec_id": "spec", "case_id": "example", "evidence": {}}
        self.assertEqual(evaluator({"spec": gold}).fixed_gold(item), gold)
        with self.assertRaises(ValueError):
            evaluator({"spec": gold}).fixed_gold({**item, "case_id": "other"})

    def test_graph_operations_preserve_the_input(self):
        store = GraphStore()
        for node_id, kind in [
            ("rule", "Rule"),
            ("first", "Value"),
            ("second", "Value"),
        ]:
            store.add_node(node_id, kind, node_id)
        edge = store.add_edge("rule", "PROTECTS", "first")
        self.assertEqual(
            apply_model_output(
                store, [], {}, {}, batch_number=0, allowed_spec_ids=set()
            ),
            [],
        )
        graph = store.data()
        original = digest(graph)
        cut, audit = graph_edit(graph, "remove_edge", edge)
        self.assertEqual(cut["edges"], [])
        rewired, _ = graph_edit(graph, "rewire_edge", edge, target="second")
        self.assertEqual(rewired["edges"][0]["target"], "second")
        for modified in (cut, rewired):
            restored, _ = graph_edit(
                modified,
                "restore",
                edge,
                patch={"edge": audit["before"], "original_index": 0},
            )
            self.assertEqual(digest(restored), original)
        self.assertEqual(digest(graph), original)

    def test_read_through_cache_keeps_resources_read_only(self):
        with TemporaryDirectory() as directory:
            resources = Path(directory, "inputs")
            work = Path(directory, "outputs")
            resources.mkdir()
            source = resources / "cache.json"
            source.write_text("{}", encoding="utf-8")
            with patch.dict(
                "os.environ",
                SPECALGO_RESOURCES=str(resources),
                SPECALGO_WORK_DIR=str(work),
            ):
                output = work / source.name
                self.assertEqual(read_json(cached_input(output)), {})
                write_json(output, {"count": 1})
                self.assertEqual(read_json(cached_input(output)), {"count": 1})
                with self.assertRaises(ValueError):
                    write_json(source, {})
                self.assertEqual(source.read_bytes(), b"{}")
            with patch.dict(
                "os.environ",
                SPECALGO_RESOURCES=str(resources),
                SPECALGO_WORK_DIR=str(resources / "work"),
            ):
                with self.assertRaises(ValueError):
                    work_root()


if __name__ == "__main__":
    unittest.main()
