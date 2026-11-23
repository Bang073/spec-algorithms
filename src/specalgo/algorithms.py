"""Public entry points for alignment, value representation and path interventions."""

from pathlib import Path

from .common import digest, read_json, read_rows, resource_root
from .contracts import Algorithm1Request, Algorithm2Request
from .data import Dataset
from .experiments import answer_and_score, messages
from .graph import GraphEngine, SpecIndex
from .models import Embedder, ExperimentModels


class SpecAlignmentValueAlgorithm:
    algorithm_id = "spec-semantic-alignment-value-v2"

    def __init__(
        self,
        data=None,
        graph=None,
        model=None,
        embedder=None,
        *,
        observations=None,
        selection=None,
    ):
        self.data = data if data is not None else Dataset()
        self.records = self.data.load()
        self.observations = Path(observations) if observations is not None else None
        self.selection = Path(selection) if selection is not None else None
        self.embedder = embedder if embedder is not None else Embedder()
        self.model = model if model is not None else ExperimentModels()
        if graph is None:
            snapshot = read_json(resource_root() / "artifacts/graph/base.json")
            if snapshot["meta"]["train_hash"] != digest(self.records):
                raise ValueError(
                    "Graph and training records belong to different snapshots"
                )
            index = SpecIndex(self.records, self.embedder, graph=snapshot)
            graph = GraphEngine(self.records, snapshot, index, gate=self.model)
        self.graph = graph

    def _represent(self, case_id):
        records = {row["case_id"]: row for row in self.records}
        selected = list(records)
        if self.selection is not None:
            selection = read_json(self.selection)
            selected = selection["selected_case_ids"]
            if (
                not isinstance(selected, list)
                or not selected
                or len(set(selected)) != len(selected)
                or set(selected) - records.keys()
            ):
                raise ValueError(
                    "Selection must contain unique training case IDs from this snapshot"
                )
            if "graph_hash" in selection and selection["graph_hash"] != digest(
                self.graph.graph
            ):
                raise ValueError("Selection and graph belong to different snapshots")
            if "train_hash" in selection and selection["train_hash"] != digest(
                self.records
            ):
                raise ValueError(
                    "Selection and training records belong to different snapshots"
                )
        if case_id is not None:
            if case_id not in selected:
                raise ValueError(
                    "Value representation requires a selected training case"
                )
            selected = [case_id]
        path = (
            self.observations
            or resource_root() / "artifacts/train_observations/results.jsonl"
        )
        observations = {}
        if self.observations is not None or path.is_file():
            for row in read_rows(path):
                if row["phase"] != "graph_spec" or row["query_type"] != "risk":
                    continue
                key = row["case_id"]
                if key not in records or key in observations:
                    raise ValueError("Unknown or duplicate training observation")
                observations[key] = row
        rows = self.graph.values(
            [records[key] for key in selected], observations, self.embedder, self.model
        )
        return {
            "schema_version": "2.0",
            "algorithm_id": self.algorithm_id,
            "results": rows,
        }

    def run(self, request):
        request = Algorithm1Request.model_validate(request)
        if request.operation == "represent":
            return self._represent(request.case_id)
        if not request.query:
            raise ValueError("A query is required for retrieval and alignment")
        if request.operation == "retrieve":
            return {
                "algorithm_id": self.algorithm_id,
                **self.graph.retrieve(request.query, request.top_k),
            }
        context = self.graph.context(
            request.query, gate=self.model, top_k=request.top_k
        )
        if request.case_id is None:
            generated = self.model.generate(
                messages(request.query, context["contexts"]),
                role="target",
                max_tokens=600,
            )
            return {
                "schema_version": "1.0",
                "algorithm_id": self.algorithm_id,
                "query": request.query,
                "retrieval": context,
                "answer": generated["answer"],
                "target_call_id": generated["call_id"],
                "score": None,
            }
        cases = read_rows(resource_root() / "data/test_private/cases.jsonl")
        matches = [
            c
            for c in cases
            if c["case_id"] == request.case_id and c["query_type"] == request.query_type
        ]
        if len(matches) != 1:
            raise ValueError(
                "Scored alignment requires one evaluation case matching case_id and query_type"
            )
        case = {**matches[0], "query": request.query}
        row = answer_and_score(
            self.data.scoring_record(request.case_id),
            case,
            context,
            self.model,
            "graph_spec",
            data=self.data,
        )
        return {"algorithm_id": self.algorithm_id, **row}


class PathInfluenceAlgorithm:
    algorithm_id = "path-intervention-influence-v2"

    def __init__(self, alignment_algorithm=None):
        self.dependencies = (
            alignment_algorithm
            if alignment_algorithm is not None
            else SpecAlignmentValueAlgorithm()
        )

    def run(self, request):
        from .attribution import PathAttributor

        request = Algorithm2Request.model_validate(request)
        operations = {"restore" if op == "repair" else op for op in request.operations}
        if operations != {"remove_content", "remove_edge", "rewire_edge", "restore"}:
            raise ValueError(
                "Supply remove_content, remove_edge, rewire_edge and restore"
            )
        alignment = self.dependencies
        evaluator = PathAttributor(
            alignment.data.test_records(),
            alignment.graph,
            alignment.model,
            data=alignment.data,
        )
        cases = read_rows(resource_root() / "data/test_private/cases.jsonl")
        return {
            "algorithm_id": self.algorithm_id,
            **evaluator.run(cases, request.case_ids),
        }
