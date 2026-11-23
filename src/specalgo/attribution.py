"""Immutable graph interventions and paired contributions."""

import copy
import json
from .common import work_root, digest, safe_component, write_json, write_rows
from .experiments import run_row, cached_context, PROTOCOL_FINGERPRINT

from ._support.path_effects import (
    PathRecord,
    remove_content,
    diagnostic_set_contributions,
)


class PathAttributor:
    def __init__(self, records, graph, model, data=None):
        self.records = records
        self.graph = graph
        self.model = model
        self.data = data

    def run(self, cases, case_ids):
        return run_attribution(
            self.records,
            cases,
            self.graph,
            self.model,
            case_ids=case_ids,
            data=self.data,
        )


def graph_edit(original, operation, edge_id, *, target=None, patch=None):
    """Every operation starts from its explicit immutable input graph."""
    graph = copy.deepcopy(original)
    nodes = {n["id"]: n for n in graph["nodes"]}
    edge = next((e for e in graph["edges"] if e["id"] == edge_id), None)
    if operation == "repair":
        operation = "restore"  # Deprecated input alias.
    if edge is None and operation != "restore":
        raise ValueError("Unknown edge")
    before = copy.deepcopy(edge)
    if operation == "remove_edge":
        graph["edges"] = [e for e in graph["edges"] if e["id"] != edge_id]
    elif operation == "rewire_edge":
        if (
            target not in nodes
            or nodes[target]["type"] != nodes[edge["target"]]["type"]
        ):
            raise ValueError("Rewiring requires compatible node types")
        if target == edge["target"]:
            raise ValueError("Rewire target must differ")
        edge["target"] = target
    elif operation == "restore":
        if patch is None:
            raise ValueError("Explicit repair patch required")
        if patch.get("edge"):
            fixed = patch["edge"]
            if (
                fixed["id"] != edge_id
                or fixed["source"] not in nodes
                or fixed["target"] not in nodes
            ):
                raise ValueError("Invalid restore edge")
            if edge is not None:
                if (
                    nodes[fixed["source"]]["type"] != nodes[edge["source"]]["type"]
                    or nodes[fixed["target"]]["type"] != nodes[edge["target"]]["type"]
                ):
                    raise ValueError("Incompatible restore")
                graph["edges"] = [
                    copy.deepcopy(fixed) if e["id"] == edge_id else e
                    for e in graph["edges"]
                ]
            else:
                graph["edges"].insert(patch["original_index"], copy.deepcopy(fixed))
        else:
            raise ValueError("Restore requires original edge and position")
    else:
        raise ValueError("Unknown graph edit")
    return graph, {
        "operation": operation,
        "edge_id": edge_id,
        "before": before,
        "after": next((e for e in graph["edges"] if e["id"] == edge_id), None),
        "original_graph_hash": digest(original),
        "edited_graph_hash": digest(graph),
        "nodes_unchanged": digest(original["nodes"]) == digest(graph["nodes"]),
        "patch": patch,
    }


def content_edit(context, path_id):
    paths = []
    for c in context["contexts"]:
        payload = {
            k: v
            for k, v in c.items()
            if k not in {"path_id", "node_ids", "edge_ids", "score", "source_refs"}
        }
        # The L3 values are interpretive rule content and must also disappear
        # with the selected path's rules, rather than leaking through a side key.
        payload["higher_level_rules"] = {
            "rules": payload.pop("higher_level_rules", []),
            "l3_values": payload.pop("l3_values", []),
        }
        paths.append(
            PathRecord(
                c["path_id"],
                tuple(c["node_ids"]),
                tuple(c["edge_ids"]),
                json.dumps(payload, ensure_ascii=False),
                c["score"],
                tuple(c["source_refs"]),
            )
        )
    modified = remove_content(paths, {path_id}, "remove_path")
    contexts = []
    original = {c["path_id"]: c for c in context["contexts"]}
    for p in modified:
        data = json.loads(p.text)
        # Preserve original evidence, including JSON key order. Reconstructing
        # untouched evidence from canonical JSON would change the
        # actual model prompt even when no operational Spec was supplied.
        kept = copy.deepcopy(original[p.path_id])
        if "operational_spec" not in data:
            kept.pop("operational_spec", None)
        if "higher_level_rules" not in data:
            kept["higher_level_rules"] = []
            kept["l3_values"] = []
        contexts.append(kept)
    return {
        **copy.deepcopy(context),
        "contexts": contexts,
        "context_hash": digest(contexts),
        "actually_supplied_l2_ids": [
            c["rule_id"] for c in contexts if c.get("operational_spec")
        ],
        "content_intervention": {
            "path_id": path_id,
            "operation": "remove_content",
            "preserve_evidence": True,
        },
    }


def source_cases(graph, rule_id):
    specs = {
        e["source"]
        for e in graph["edges"]
        if e["type"] == "CLAIMS_RULE" and e["target"] == rule_id
    }
    return sorted({n["case_id"] for n in graph["nodes"] if n["id"] in specs})


def run_attribution(records, cases, engine, model, case_ids=None, data=None):
    from .freeze import verify_freeze

    frozen = verify_freeze(engine, model)
    original = copy.deepcopy(engine.graph)
    original_hash = digest(original)
    by_id = {r["case_id"]: r for r in records}
    out = (
        work_root()
        / "artifacts"
        / (
            "attribution"
            if case_ids is None
            else "attribution_queries/" + digest(sorted(case_ids))[:20]
        )
    )
    selected = case_ids or sorted(
        {c["case_id"] for c in cases if c["query_type"] == "benign"}
    )
    if len(set(selected)) != len(selected) or set(selected) - set(by_id):
        raise ValueError("Unknown/duplicate attribution case IDs")
    chosen = [c for c in cases if c["case_id"] in selected]
    for cid in selected:
        if {c["query_type"] for c in chosen if c["case_id"] == cid} != {
            "risk",
            "benign",
        }:
            raise ValueError("Each attribution case needs risk and benign questions")
    all_rows = []
    audits = []
    for case in chosen:
        record = by_id[case["case_id"]]
        full_ctx = cached_context(engine, case["query"], model)
        path = next(
            (
                c
                for c in full_ctx["contexts"]
                if c.get("operational_spec") and c["edge_ids"]
            ),
            None,
        )
        if path is None:
            audits.append(
                {
                    "test_id": case["test_id"],
                    "case_id": case["case_id"],
                    "status": "not_applicable",
                    "reason": "no_supplied_L2_path",
                }
            )
            write_rows(out / "audits.partial.jsonl", audits)
            continue
        edge_id = next(
            eid
            for eid in path["edge_ids"]
            if any(
                e["id"] == eid and e["type"] == "PROTECTS" for e in original["edges"]
            )
        )
        edge = next(e for e in original["edges"] if e["id"] == edge_id)
        detached, cut = graph_edit(original, "remove_edge", edge_id)
        # Fault injection is chosen solely from graph/ranking, never answers or gold.
        existing = {
            e["target"]
            for e in original["edges"]
            if e["type"] == "PROTECTS" and e["source"] == edge["source"]
        }
        alternatives = [
            v
            for v in reversed(engine.index.rank(case["query"], "l3"))
            if v["node_id"] not in existing
        ]
        if not alternatives:
            raise ValueError("No compatible alternative L3")
        broken, rewire = graph_edit(
            original, "rewire_edge", edge_id, target=alternatives[0]["node_id"]
        )
        restored, restoration = graph_edit(
            broken,
            "restore",
            edge_id,
            patch={"edge": edge, "original_index": original["edges"].index(edge)},
        )
        if digest(restored) != original_hash:
            raise AssertionError("Restoration differs from original")
        removed = content_edit(full_ctx, path["path_id"])
        phases = [
            ("full", full_ctx, None),
            ("remove_content", removed, removed["content_intervention"]),
            (
                "remove_edge",
                cached_context(engine, case["query"], model, graph=detached),
                cut,
            ),
            (
                "rewire_edge",
                cached_context(engine, case["query"], model, graph=broken),
                rewire,
            ),
            (
                "restore",
                cached_context(engine, case["query"], model, graph=restored),
                restoration,
            ),
        ]
        local = {}
        for phase, ctx, audit in phases:
            p = (
                out
                / "rows"
                / (
                    safe_component(case["test_id"].replace(":", "_"))
                    + "_"
                    + phase
                    + ".json"
                )
            )
            row = run_row(
                record, case, ctx, model, phase, p, fresh=phase == "restore", data=data
            )
            row.update(
                original_graph_hash=original_hash,
                intervention=audit,
                freeze_id=frozen["freeze_id"],
                target_l2_id=path["rule_id"],
                target_path_id=path["path_id"],
                target_edge_id=edge_id,
                training_source_case_ids=source_cases(original, path["rule_id"]),
            )
            write_json(p, row)
            all_rows.append(row)
            local[phase] = row
        if local["full"]["input_hash"] != local["restore"]["input_hash"]:
            raise AssertionError("Restored input differs")
        if local["full"]["input_hash"] == local["remove_content"]["input_hash"]:
            raise AssertionError("Content intervention did not change model input")
        if digest(original) != original_hash or digest(engine.graph) != original_hash:
            raise AssertionError("Original graph mutated")
        audit = {
            "test_id": case["test_id"],
            "case_id": case["case_id"],
            "status": "executed",
            "target_path_id": path["path_id"],
            "target_l2_id": path["rule_id"],
            "target_edge_id": edge_id,
            "training_source_case_ids": source_cases(original, path["rule_id"]),
            "original_graph_unchanged": True,
            "restore_same_graph_and_input": True,
            "operations": {
                op: {
                    "input_changed": local[op]["input_hash"]
                    != local["full"]["input_hash"],
                    "supplied_l2_ids": local[op]["retrieval"][
                        "actually_supplied_l2_ids"
                    ],
                    "target_still_returned": path["rule_id"]
                    in local[op]["retrieval"]["actually_supplied_l2_ids"],
                    "edited_graph_hash": local[op]["graph_hash"],
                }
                for op in ["remove_content", "remove_edge", "rewire_edge"]
            },
            "replay": {
                "original_graph_hash": original_hash,
                "edge": edge,
                "rewire_target": alternatives[0]["node_id"],
            },
        }
        audits.append(audit)
        write_rows(out / "audits.partial.jsonl", audits)
        write_json(
            out / "progress.json",
            {
                "done_questions": len(audits),
                "planned_questions": len(chosen),
                "rows": len(all_rows),
                "status": "running",
            },
        )
        print(
            json.dumps(
                {
                    "event": "attribution",
                    "done_questions": len(audits),
                    "planned_questions": len(chosen),
                    "test_id": case["test_id"],
                }
            ),
            flush=True,
        )
    full = [r for r in all_rows if r["phase"] == "full"]
    restored = [r for r in all_rows if r["phase"] == "restore"]
    comparisons = [
        {
            "operation": op,
            "contributions": diagnostic_set_contributions(
                full, [r for r in all_rows if r["phase"] == op], restore_rows=restored
            ),
        }
        for op in ["remove_content", "remove_edge", "rewire_edge"]
    ]
    per_rule = []
    for rid in sorted({a["target_l2_id"] for a in audits if a["status"] == "executed"}):
        part = [r for r in all_rows if r["target_l2_id"] == rid]
        per_rule.append(
            {
                "l2_id": rid,
                "training_source_case_ids": source_cases(original, rid),
                "test_ids": sorted({r["test_id"] for r in part}),
                "comparisons": [
                    {
                        "operation": op,
                        "contributions": diagnostic_set_contributions(
                            [r for r in part if r["phase"] == "full"],
                            [r for r in part if r["phase"] == op],
                            restore_rows=[r for r in part if r["phase"] == "restore"],
                        ),
                    }
                    for op in ["remove_content", "remove_edge", "rewire_edge"]
                ],
            }
        )
    counts = {
        cid: sum(a["case_id"] == cid and a["status"] == "executed" for a in audits)
        for cid in selected
    }
    paired = sum(v == 2 for v in counts.values())
    scored = sum(r["score"]["grading_status"] == "scored" for r in all_rows)
    summary = {
        "schema_version": "2.0",
        "algorithm": "L2-path-intervention-influence-v2",
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "freeze_id": frozen["freeze_id"],
        "graph_hash": original_hash,
        "selected_case_ids": selected,
        "selected_cases": len(selected),
        "paired_cases_executed": paired,
        "questions": len(chosen),
        "not_applicable": sum(a["status"] == "not_applicable" for a in audits),
        "runs": len(all_rows),
        "scored": scored,
        "complete": paired >= 100 and scored == len(all_rows),
        "is_mock": False,
        "unit": "inference-time L2 rules and graph edges, no training-gradient claim",
        "comparisons": comparisons,
        "per_rule": per_rule,
        "interpretation": "An L2 effect traces its supporting training L0 set, not independent causal effects of every supporting L0.",
    }
    write_rows(out / "results.jsonl", all_rows)
    write_rows(out / "audits.jsonl", audits)
    write_json(out / "summary.json", summary)
    write_json(
        out / "progress.json",
        {
            "done_questions": len(audits),
            "planned_questions": len(chosen),
            "rows": len(all_rows),
            "status": "complete",
        },
    )
    return summary
