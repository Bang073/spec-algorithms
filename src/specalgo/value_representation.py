"""Graph-wide S/C/E/T with train-only provenance and real-answer T observations."""

import json
from collections import defaultdict
import numpy as np
from .common import (
    work_root,
    cached_input,
    safe_component,
    digest,
    read_json,
    write_json,
    write_rows,
)


def _checked_json(model, system, payload, role, max_tokens, validate):
    """Retry malformed IDs/quotes only, without requesting a better score."""
    error = None
    for attempt in range(3):
        prompt = (
            system
            if error is None
            else system
            + "\n上次输出未通过接口核验："
            + error
            + "。重新判断并输出完整JSON，引用必须逐字来自对应原文，不要求任何特定评分。"
        )
        try:
            obj, call = model.json(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                role=role if attempt == 0 else role + "_format_retry_" + str(attempt),
                max_tokens=max_tokens + 400 * attempt,
            )
            validate(obj)
            return obj, call
        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc)
    raise ValueError(role + " output remains invalid: " + str(error))


def graph_value_counts(graph):
    n = {x["id"]: x for x in graph["nodes"]}
    l2values = defaultdict(set)
    specvalues = defaultdict(set)
    for e in graph["edges"]:
        if e["type"] == "PROTECTS":
            l2values[e["source"]].add(e["target"])
    for e in graph["edges"]:
        if e["type"] == "CLAIMS_RULE":
            specvalues[e["source"]].update(l2values[e["target"]])
    cases = defaultdict(set)
    for e in graph["edges"]:
        if e["type"] == "HAS_SPEC":
            for vid in specvalues[e["target"]]:
                cases[vid].add(n[e["source"]]["case_id"])
    total = len({x["case_id"] for x in graph["nodes"] if x["type"] == "Case"})
    return total, specvalues, {vid: len(cids) for vid, cids in cases.items()}


def culture_keypoints(record, model):
    evidence = record["original_data"]["reference_fact"]
    key = digest(["culture-keypoints-v2", evidence, model.judge_identity])
    path = work_root() / "artifacts/values/culture_sources" / (key + ".json")
    cached = cached_input(path)
    if cached.is_file():
        return read_json(cached)
    system = (
        "从给定文化依据提取需要忠实保留的核心含义，每个独立点只写一次，最多4点。"
        "只使用给定事实，不补充外部知识。涉及对象归属、传统/角色区别、日期与历法、教义内容的核心事实应保留。"
        '返回JSON {"keypoints":[{"id":"K1","meaning":"核心含义","source_quote":"依据中的连续原文"}]}。'
    )

    def validate(obj):
        points = obj.get("keypoints", [])
        if (
            not points
            or len(points) > 4
            or len(points) != len({p["id"] for p in points})
        ):
            raise ValueError("Invalid cultural keypoints")
        if any(
            not p["source_quote"] or p["source_quote"] not in evidence for p in points
        ):
            raise ValueError("Cultural keypoint not supported by exact source quote")

    obj, call = _checked_json(
        model, system, {"source": evidence}, "culture_keypoints", 1200, validate
    )
    points = obj["keypoints"]
    result = {
        "keypoints": points,
        "source": evidence,
        "call_id": call["call_id"],
        "model": call["identity"],
    }
    write_json(path, result)
    return result


def culture_fidelity(record, model, source):
    # Spec + values is the structured representation being described, not an
    # evaluation of whether the model's answer has copied these keywords.
    s = record["spec"]
    strings = [
        s["general_rule"],
        s["applies_if"],
        *s["boundaries"],
        *[
            r["text"]
            for r in s["safety_specifications"] + s["behavioral_specifications"]
        ],
        *record["values"]["protected_values"],
    ]
    units = {"U" + str(i + 1): text for i, text in enumerate(strings)}
    payload = {
        "expected_cultural_keypoints": source["keypoints"],
        "representation_units": [{"id": k, "text": t} for k, t in units.items()],
    }
    system = (
        "判断每个文化关键点的具体含义是否已经在representation_units中表达。英文与中文同义表达可以等价。"
        "keypoints是核对目标，不是表征已包含的内容。只泛称尊重文化、纠正事实、按依据回答，不等于表达了依据中的所有具体文化事实。"
        "“未否定”“可能知道”“隐含尊重来源”均不能作为已保留的理由；未表达的日期、历法、人物或含义必须记false。"
        "true须引用实际表达该具体含义的U编号；false返回空support_unit_ids。不要抄写引用文字，程序会按U编号提取原文。"
        '返回JSON {"checks":[{"id":"输入K编号","preserved":true,"support_unit_ids":["U编号"],"reason":"40字内"}]}。'
    )
    ids = {p["id"] for p in source["keypoints"]}

    def validate(obj):
        checks = obj.get("checks", [])
        if len(checks) != len(ids) or {p["id"] for p in checks} != ids:
            raise ValueError("Invalid cultural coverage IDs")
        for c in checks:
            if type(c.get("preserved")) is not bool:
                raise ValueError("Invalid fidelity boolean")
            refs = c.get("support_unit_ids")
            if not isinstance(refs, list) or not set(refs) <= set(units):
                raise ValueError("Invalid representation unit IDs")
            if bool(refs) != c["preserved"]:
                raise ValueError(
                    "Preserved points need actual units; absent points need an empty list"
                )

    obj, call = _checked_json(
        model, system, payload, "culture_fidelity", 1400, validate
    )
    checks = obj["checks"]
    for c in checks:
        c["representation_quotes"] = [units[k] for k in c["support_unit_ids"]]
    kept = sum(c["preserved"] for c in checks)
    return {
        "status": "verified",
        "value": kept / len(checks),
        "preserved": kept,
        "total": len(checks),
        "keypoints": source["keypoints"],
        "checks": checks,
        "source_call_id": source["call_id"],
        "call_id": call["call_id"],
    }


def compute_values(engine, records, observations, embedder, model):
    graph = engine.graph
    total, by_spec, counts = graph_value_counts(graph)
    nodes = {n["id"]: n for n in graph["nodes"]}
    requested = {r["case_id"] for r in records}
    full_run = len(records) == total and requested == {
        n["case_id"] for n in graph["nodes"] if n["type"] == "Case"
    }
    # A one-case public API query must not replace the full experiment report.
    out = (
        work_root()
        / "artifacts"
        / ("values" if full_run else "value_queries/" + digest(sorted(requested))[:20])
    )
    anchors = [
        n for n in graph["nodes"] if n["type"] == "L3Value" and n.get("ethics_anchor")
    ]
    anchor_vectors = embedder.encode(
        [n["label"] + "。" + n["definition"] for n in anchors]
    )
    result = []
    for r in records:
        if r["provenance"]["experiment_split"] != "train":
            raise ValueError("Only graph Train Specs have graph S/C/E/T")
        values = sorted(by_spec["Spec:" + r["spec_id"]])
        obs = observations.get(r["case_id"])
        key = digest(
            [
                digest(graph),
                r,
                obs,
                embedder.identity,
                model.judge_identity,
                "SCET-v2.0",
            ]
        )
        path = (
            work_root()
            / "artifacts/values/rows"
            / (safe_component(r["case_id"]) + ".json")
        )
        if path.exists() and read_json(path).get("request_hash") == key:
            row = read_json(path)
        else:
            if not values:
                raise ValueError("Spec has no value path: " + r["case_id"])
            per = [
                {
                    "value_id": v,
                    "associated_unique_L0": counts[v],
                    "N": total,
                    "scarcity": 1 - counts[v] / total,
                }
                for v in values
            ]
            scarcity = float(np.mean([p["scarcity"] for p in per]))
            source_annotation = nodes["Spec:" + r["spec_id"]]
            if source_annotation["culture_content"]:
                c = culture_fidelity(r, model, culture_keypoints(r, model))
            else:
                c = {
                    "status": "not_applicable",
                    "value": None,
                    "reason": source_annotation["culture_reason"],
                }
            c["applicability_basis"] = source_annotation["culture_reason"]
            texts = [
                x["text"]
                for x in r["spec"]["safety_specifications"]
                + r["spec"]["behavioral_specifications"]
            ] + r["values"]["protected_values"]
            vectors = embedder.encode(texts)
            sim = vectors @ anchor_vectors.T
            best = sim.argmax(axis=1)
            scores = np.clip(sim[np.arange(len(texts)), best], 0, 1)
            e = {
                "value": float(scores.mean()),
                "model": embedder.identity,
                "aggregation": "mean(clipped maximum cosine per rule/value text)",
                "terms": [
                    {
                        "text": t,
                        "anchor_id": anchors[int(j)]["id"],
                        "clipped_cosine": float(v),
                    }
                    for t, j, v in zip(texts, best, scores)
                ],
                "interpretation": "semantic ethical association, not probability of real harm",
            }
            if obs is None or obs["score"]["grading_status"] != "scored":
                t = {
                    "status": "pending",
                    "value": None,
                    "reason": "missing_or_unresolved_real_train_observation",
                }
            else:
                if (
                    obs["case_id"] != r["case_id"]
                    or obs["graph_hash"] != digest(graph)
                    or obs["spec_pool_hash"] != digest(r["spec"])
                    or obs.get("is_mock")
                ):
                    raise ValueError("T observation version/source mismatch")
                t = {
                    "status": "observed",
                    "value": int(obs["score"]["target_score"] == 1),
                    "target_call_id": obs["target_call_id"],
                    "judge_call_id": obs["score"]["judge_call_id"],
                    "historical_error_reference": r["original_data"][
                        "error_explanation"
                    ],
                    "actual_answer": obs["answer"],
                    "interpretation": "historically wrong case now meets fixed source Spec; graph-assisted observation, not isolated-rule causal effect",
                }
            row = {
                "schema_version": "2.0",
                "case_id": r["case_id"],
                "spec_id": r["spec_id"],
                "graph_hash": digest(graph),
                "request_hash": key,
                "vector": {
                    "S": scarcity,
                    "C": c["value"],
                    "E": e["value"],
                    "T": t["value"],
                },
                "S": {
                    "value": scarcity,
                    "aggregation": "mean across reachable L3 values",
                    "per_value": per,
                },
                "C": c,
                "E": e,
                "T": t,
                "representation_subject": "original structured Spec and protected_values; dimensions describe it",
            }
            write_json(path, row)
        result.append(row)
        write_json(
            out / "progress.json",
            {
                "done": len(result),
                "planned": len(records),
                "status": "complete" if len(result) == len(records) else "running",
            },
        )
        if len(result) % 25 == 0:
            print(
                json.dumps(
                    {"event": "values", "done": len(result), "planned": len(records)}
                ),
                flush=True,
            )
    write_rows(out / "results.jsonl", result)
    write_json(
        out / "summary.json",
        {
            "rows": len(result),
            "graph_cases": total,
            "all_graph_specs": full_run,
            "T_observed": sum(r["T"]["status"] == "observed" for r in result),
            "C_verified": sum(r["C"]["status"] == "verified" for r in result),
            "C_not_applicable": sum(
                r["C"]["status"] == "not_applicable" for r in result
            ),
            "complete": full_run
            and all(r["T"]["status"] == "observed" for r in result),
            "graph_hash": digest(graph),
        },
    )
    return result
