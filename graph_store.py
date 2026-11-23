"""Storage and export for typed specification graphs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx


def sha256(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    return f"{prefix}:{sha256(list(parts))[:length]}"


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def rule_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    spec = record.get("spec", record)
    result = []
    for field, kind in (
        ("safety_specifications", "Safety"),
        ("behavioral_specifications", "Behavioral"),
    ):
        for row in spec.get(field, []) or []:
            if not isinstance(row, dict) or not row.get("rule_id"):
                continue
            result.append(
                {
                    "rule_id": row["rule_id"],
                    "text": str(row.get("text", "")),
                    "rule_kind": kind,
                    "source_field": row.get("source_field"),
                    "source_text": row.get("source_text"),
                }
            )
    return result


class GraphStore:
    def __init__(self, data: dict[str, Any] | None = None):
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.meta: dict[str, Any] = {}
        if data:
            self.meta = dict(data.get("meta", {}))
            for node in data.get("nodes", []):
                self.nodes[node["id"]] = dict(node)
            for edge in data.get("edges", []):
                self.edges[edge["id"]] = dict(edge)

    def add_node(self, node_id: str, node_type: str, label: str, **attrs: Any) -> str:
        record = {"id": node_id, "type": node_type, "label": label, **attrs}
        old = self.nodes.get(node_id)
        if old is not None:
            comparable_old = {key: old.get(key) for key in record}
            if comparable_old != record:
                raise ValueError(f"Conflicting node identity: {node_id}")
        else:
            self.nodes[node_id] = record
        return node_id

    def add_edge(self, source: str, edge_type: str, target: str, **attrs: Any) -> str:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Dangling graph edge: {source} -> {target}")
        edge_id = stable_id("Edge", source, edge_type, target, attrs)
        self.edges[edge_id] = {
            "id": edge_id,
            "source": source,
            "target": target,
            "type": edge_type,
            **attrs,
        }
        return edge_id

    def data(self, node_types: set[str] | None = None) -> dict[str, Any]:
        nodes = [
            node
            for node in self.nodes.values()
            if node_types is None or node.get("type") in node_types
        ]
        ids = {node["id"] for node in nodes}
        edges = [
            edge
            for edge in self.edges.values()
            if edge["source"] in ids and edge["target"] in ids
        ]
        return {"meta": dict(self.meta), "nodes": nodes, "edges": edges}

    def save(self, path: Path, node_types: set[str] | None = None) -> None:
        write_json(path, self.data(node_types))

    @classmethod
    def from_path(cls, path: Path) -> "GraphStore":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def _graphml_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, bool)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def export_graphml(path: Path, data: dict[str, Any]) -> None:
    graph = nx.MultiDiGraph()
    for node in data.get("nodes", []):
        graph.add_node(
            node["id"],
            **{
                key: _graphml_scalar(value)
                for key, value in node.items()
                if key != "id"
            },
        )
    for edge in data.get("edges", []):
        graph.add_edge(
            edge["source"],
            edge["target"],
            key=edge["id"],
            relation_id=edge["id"],
            **{
                key: _graphml_scalar(value)
                for key, value in edge.items()
                if key not in {"id", "source", "target"}
            },
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path, edge_id_from_attribute="relation_id")
    restored = nx.read_graphml(path, force_multigraph=True)
    if len(restored.nodes) != len(
        data.get("nodes", [])
    ) or restored.number_of_edges() != len(data.get("edges", [])):
        raise ValueError(f"GraphML round-trip changed node/edge counts: {path}")
