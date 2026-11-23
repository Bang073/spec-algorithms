"""Validate approved proposals and apply their relations to a graph."""

from __future__ import annotations

from typing import Any
from .graph_store import GraphStore, rule_rows, stable_id


RELATIONS = {
    "equivalent",
    "existing_upper",
    "shared_requirement",
    "topic_only",
    "conflict",
}


QUALITY_EXCLUDED = {"quarantined", "quality_hold"}


def _record_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["spec_id"]: row for row in records}


def is_quality_eligible(record: dict[str, Any]) -> bool:
    """Return whether a Spec may participate in semantic construction.

    Isolated records remain available in the provenance graph, but they must not
    enter retrieval, model prompts, or claims.  Keep this predicate in the
    engine so build and update cannot accidentally drift apart.
    """
    return str(record.get("quality_status", "")).strip().lower() not in QUALITY_EXCLUDED


def _covered_rule_ids(record: dict[str, Any]) -> set[str]:
    return {row["rule_id"] for row in rule_rows(record)}


def _review_approved_rule_ids(review: dict[str, Any]) -> set[str]:
    if not isinstance(review, dict):
        return set()
    values = review.get("approved_rule_ids", [])
    if isinstance(values, list) and values:
        return {str(value) for value in values}
    # Also accept a detailed reviewer response if a user edits the prompt later.
    return {
        str(row.get("proposal_id"))
        for row in review.get("proposal_reviews", [])
        if isinstance(row, dict) and row.get("valid") and row.get("proposal_id")
    }


def _review_approved_claims(review: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(review, dict):
        return []
    if isinstance(review.get("approved_existing_claims"), list):
        return [
            row for row in review["approved_existing_claims"] if isinstance(row, dict)
        ]
    result = []
    for row in review.get("existing_claim_reviews", []):
        if isinstance(row, dict) and row.get("valid"):
            result.append(row)
    return result


def _review_approved_pairs(review: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(review, dict):
        return []
    if isinstance(review.get("approved_pair_decisions"), list):
        return [
            row for row in review["approved_pair_decisions"] if isinstance(row, dict)
        ]
    return [
        row
        for row in review.get("pair_reviews", [])
        if isinstance(row, dict) and row.get("valid")
    ]


def _claim_key(claim: dict[str, Any]) -> tuple[str, str, tuple[str, ...]] | None:
    if not isinstance(claim, dict):
        return None
    upper_rule_id = str(claim.get("upper_rule_id", "")).strip()
    spec_id = str(claim.get("spec_id", "")).strip()
    covered = claim.get("covered_rule_ids")
    if not upper_rule_id or not spec_id or not isinstance(covered, list) or not covered:
        return None
    return upper_rule_id, spec_id, tuple(sorted({str(value) for value in covered}))


def _pair_key(pair: dict[str, Any]) -> tuple[str, str, str] | None:
    if not isinstance(pair, dict):
        return None
    left = str(pair.get("left_spec_id", "")).strip()
    right = str(pair.get("right_spec_id", "")).strip()
    relation = str(pair.get("relation", "")).strip()
    if not left or not right or not relation:
        return None
    return left, right, relation


def _proposal_claims(
    proposal: dict[str, Any],
) -> dict[tuple[str, str, tuple[str, ...]], dict[str, Any]]:
    values = proposal.get("existing_claims", []) if isinstance(proposal, dict) else []
    return {
        key: claim
        for claim in values
        if isinstance(claim, dict)
        for key in [_claim_key(claim)]
        if key is not None
    }


def _proposal_pairs(
    proposal: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    values = proposal.get("pair_decisions", []) if isinstance(proposal, dict) else []
    return {
        key: pair
        for pair in values
        if isinstance(pair, dict)
        for key in [_pair_key(pair)]
        if key is not None
    }


def _valid_new_claims(
    graph: GraphStore,
    claims: Any,
    by_id: dict[str, dict[str, Any]],
    allowed_spec_ids: set[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate claims before the independent-base-fact gate is evaluated."""
    if not isinstance(claims, list):
        return [], [{"spec_id": "", "reason": "claims_not_a_list"}]
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            rejected.append({"spec_id": "", "reason": "claim_not_an_object"})
            continue
        sid = str(claim.get("spec_id", "")).strip()
        record = by_id.get(sid)
        if not sid or record is None:
            rejected.append({"spec_id": sid, "reason": "unknown_identity"})
            continue
        if allowed_spec_ids is not None and sid not in allowed_spec_ids:
            rejected.append({"spec_id": sid, "reason": "spec_not_in_batch_payload"})
            continue
        if not is_quality_eligible(record):
            rejected.append({"spec_id": sid, "reason": "quality_isolated"})
            continue
        covered = claim.get("covered_rule_ids")
        if not isinstance(covered, list) or not covered:
            rejected.append({"spec_id": sid, "reason": "unknown_or_empty_source_rule"})
            continue
        covered = [str(value) for value in covered]
        if not set(covered).issubset(_covered_rule_ids(record)):
            rejected.append({"spec_id": sid, "reason": "unknown_or_empty_source_rule"})
            continue
        key = (sid, tuple(sorted(set(covered))))
        if key in seen:
            continue
        seen.add(key)
        if f"Spec:{sid}" not in graph.nodes:
            rejected.append({"spec_id": sid, "reason": "missing_spec_node"})
            continue
        valid.append({**claim, "spec_id": sid, "covered_rule_ids": covered})
    return valid, rejected


def _existing_claim_edge(
    graph: GraphStore, source: str, target: str, covered_rule_ids: list[str]
) -> str | None:
    wanted = set(covered_rule_ids)
    for edge in graph.edges.values():
        if (
            edge.get("type") == "CLAIMS_RULE"
            and edge.get("source") == source
            and edge.get("target") == target
            and set(str(value) for value in edge.get("covered_rule_ids", [])) == wanted
        ):
            return str(edge["id"])
    return None


def apply_model_output(
    graph: GraphStore,
    records: list[dict[str, Any]],
    proposal: dict[str, Any],
    review: dict[str, Any],
    batch_number: int,
    allowed_spec_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    by_id = _record_by_id(records)
    events: list[dict[str, Any]] = []
    if not isinstance(proposal, dict):
        proposal = {}
    proposals = proposal.get("new_rules", proposal.get("new_upper_rules", []))
    if not isinstance(proposals, list):
        proposals = []
    proposals_by_id = {
        str(row.get("proposal_id")): row
        for row in proposals
        if isinstance(row, dict) and row.get("proposal_id")
    }
    approved_rules = _review_approved_rule_ids(review)
    for proposal_id, rule in proposals_by_id.items():
        if proposal_id not in approved_rules:
            events.append(
                {
                    "batch": batch_number,
                    "type": "rejected_rule",
                    "proposal_id": proposal_id,
                    "reason": "review_not_approved",
                }
            )
            continue
        claims, rejected_claims = _valid_new_claims(
            graph, rule.get("claims", []), by_id, allowed_spec_ids
        )
        for rejected in rejected_claims:
            events.append(
                {
                    "batch": batch_number,
                    "type": "rejected_claim",
                    "proposal_id": proposal_id,
                    **rejected,
                }
            )
        claim_records = [by_id[claim["spec_id"]] for claim in claims]
        base_facts = {
            str(row.get("base_fact_id")).strip()
            for row in claim_records
            if str(row.get("base_fact_id", "")).strip()
        }
        if len(base_facts) < 2:
            events.append(
                {
                    "batch": batch_number,
                    "type": "rejected_rule",
                    "proposal_id": proposal_id,
                    "reason": "fewer_than_two_independent_base_facts",
                }
            )
            continue
        upper_id = stable_id(
            "G0Rule",
            rule.get("label", ""),
            rule.get("requirement", ""),
            rule.get("applies_if", ""),
        )
        existing = graph.nodes.get(upper_id)
        created_upper = existing is None
        if existing is not None and existing.get("type") != "G0Rule":
            events.append(
                {
                    "batch": batch_number,
                    "type": "rejected_rule",
                    "proposal_id": proposal_id,
                    "reason": "conflicting_rule_identity",
                }
            )
            continue
        if created_upper:
            graph.add_node(
                upper_id,
                "G0Rule",
                str(rule.get("label") or rule.get("requirement") or proposal_id),
                rule_id=upper_id,
                requirement=str(rule.get("requirement", "")),
                actor=str(rule.get("actor", "assistant")),
                modality=str(rule.get("modality", "mixed")),
                rule_kind=str(rule.get("rule_kind", "Mixed")),
                applies_if=str(rule.get("applies_if", "")),
                boundaries=[str(value) for value in rule.get("boundaries", [])],
                quality_status="automatically_reviewed_candidate",
                human_approved=False,
                support_spec_ids=list(
                    dict.fromkeys(row["spec_id"] for row in claim_records)
                ),
                enforcement_scope="union_of_claimed_spec_scopes",
            )
        accepted_claims = 0
        for claim in claims:
            sid = str(claim.get("spec_id"))
            record = by_id.get(sid)
            if record is None:
                continue
            covered = [str(value) for value in claim.get("covered_rule_ids", [])]
            spec = (
                record.get("spec", {})
                if isinstance(record.get("spec", {}), dict)
                else {}
            )
            edge_id = _existing_claim_edge(graph, f"Spec:{sid}", upper_id, covered)
            if edge_id is None:
                edge_id = graph.add_edge(
                    f"Spec:{sid}",
                    "CLAIMS_RULE",
                    upper_id,
                    covered_rule_ids=covered,
                    reason=str(claim.get("reason", "")),
                    status="automatically_reviewed_candidate",
                    applies_if=spec.get("applies_if", ""),
                    boundaries=spec.get("exceptions_or_boundaries", []),
                    batch=batch_number,
                )
            domain_id = record.get("domain_id")
            if domain_id not in graph.nodes:
                domain_id = next(
                    (
                        node_id
                        for node_id, node in graph.nodes.items()
                        if node.get("type") == "Domain"
                        and node.get("label") == record.get("domain_name")
                    ),
                    domain_id,
                )
            if domain_id in graph.nodes:
                graph.add_edge(
                    upper_id,
                    "APPLIES_TO_DOMAIN",
                    domain_id,
                    supporting_spec_ids=[sid],
                    status="candidate",
                )
            accepted_claims += 1
            events.append(
                {
                    "batch": batch_number,
                    "type": "claimed_new_upper",
                    "upper_rule_id": upper_id,
                    "spec_id": sid,
                    "edge_id": edge_id,
                    "covered_rule_ids": covered,
                }
            )
        if accepted_claims == 0:
            if created_upper:
                graph.nodes.pop(upper_id, None)
            events.append(
                {
                    "batch": batch_number,
                    "type": "rejected_rule",
                    "proposal_id": proposal_id,
                    "reason": "no_valid_claims",
                }
            )
        else:
            events.append(
                {
                    "batch": batch_number,
                    "type": "created_or_reused_upper",
                    "upper_rule_id": upper_id,
                    "proposal_id": proposal_id,
                    "accepted_claims": accepted_claims,
                    "status": "created" if created_upper else "reused",
                }
            )

    proposed_claims = _proposal_claims(proposal)
    approved_claim_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    for approved_claim in _review_approved_claims(review):
        key = _claim_key(approved_claim)
        if key is None or key not in proposed_claims:
            events.append(
                {
                    "batch": batch_number,
                    "type": "rejected_existing_claim",
                    "upper_rule_id": str(approved_claim.get("upper_rule_id", "")),
                    "spec_id": str(approved_claim.get("spec_id", "")),
                    "reason": "review_not_in_proposal",
                }
            )
            continue
        if key in approved_claim_keys:
            continue
        approved_claim_keys.add(key)
        # Apply the proposer version.  Reviewer output is an approval signal,
        # not a second source of rule text or covered IDs.
        claim = proposed_claims[key]
        uid = str(claim.get("upper_rule_id"))
        sid = str(claim.get("spec_id"))
        record = by_id.get(sid)
        if (
            uid not in graph.nodes
            or graph.nodes[uid].get("type") != "G0Rule"
            or record is None
            or not is_quality_eligible(record)
            or (allowed_spec_ids is not None and sid not in allowed_spec_ids)
        ):
            reason = (
                "quality_isolated"
                if record is not None and not is_quality_eligible(record)
                else "spec_not_in_batch_payload"
                if allowed_spec_ids is not None and sid not in allowed_spec_ids
                else "unknown_identity"
            )
            events.append(
                {
                    "batch": batch_number,
                    "type": "rejected_existing_claim",
                    "upper_rule_id": uid,
                    "spec_id": sid,
                    "reason": reason,
                }
            )
            continue
        covered = [str(value) for value in claim.get("covered_rule_ids", [])]
        if not covered or not set(covered).issubset(_covered_rule_ids(record)):
            events.append(
                {
                    "batch": batch_number,
                    "type": "rejected_existing_claim",
                    "upper_rule_id": uid,
                    "spec_id": sid,
                    "reason": "unknown_source_rule",
                }
            )
            continue
        edge_id = _existing_claim_edge(graph, f"Spec:{sid}", uid, covered)
        if edge_id is None:
            edge_id = graph.add_edge(
                f"Spec:{sid}",
                "CLAIMS_RULE",
                uid,
                covered_rule_ids=covered,
                reason=str(claim.get("reason", "")),
                status="automatically_reviewed_candidate",
                batch=batch_number,
            )
        domain_id = record.get("domain_id")
        if domain_id not in graph.nodes:
            domain_id = next(
                (
                    node_id
                    for node_id, node in graph.nodes.items()
                    if node.get("type") == "Domain"
                    and node.get("label") == record.get("domain_name")
                ),
                domain_id,
            )
        if domain_id in graph.nodes:
            graph.add_edge(
                uid,
                "APPLIES_TO_DOMAIN",
                domain_id,
                supporting_spec_ids=[sid],
                status="candidate",
            )
        events.append(
            {
                "batch": batch_number,
                "type": "claimed_existing_upper",
                "upper_rule_id": uid,
                "spec_id": sid,
                "edge_id": edge_id,
            }
        )

    proposed_pairs = _proposal_pairs(proposal)
    approved_pair_keys: set[tuple[str, str, str]] = set()
    for approved_pair in _review_approved_pairs(review):
        key = _pair_key(approved_pair)
        if key is None or key not in proposed_pairs:
            events.append(
                {
                    "batch": batch_number,
                    "type": "rejected_pair",
                    "pair": approved_pair,
                    "reason": "review_not_in_proposal",
                }
            )
            continue
        if key in approved_pair_keys:
            continue
        approved_pair_keys.add(key)
        # Preserve the proposer's relation and rationale exactly.
        pair = proposed_pairs[key]
        left = str(pair.get("left_spec_id"))
        right = str(pair.get("right_spec_id"))
        relation = str(pair.get("relation"))
        if (
            left not in by_id
            or right not in by_id
            or relation not in RELATIONS
            or not is_quality_eligible(by_id[left])
            or not is_quality_eligible(by_id[right])
            or (
                allowed_spec_ids is not None
                and (left not in allowed_spec_ids or right not in allowed_spec_ids)
            )
        ):
            reason = (
                "quality_isolated"
                if (
                    left in by_id
                    and not is_quality_eligible(by_id[left])
                    or right in by_id
                    and not is_quality_eligible(by_id[right])
                )
                else "spec_not_in_batch_payload"
                if allowed_spec_ids is not None
                and (left not in allowed_spec_ids or right not in allowed_spec_ids)
                else "unknown_identity_or_relation"
            )
            events.append(
                {
                    "batch": batch_number,
                    "type": "rejected_pair",
                    "pair": pair,
                    "reason": reason,
                }
            )
            continue
        edge_id = None
        if relation in {"equivalent", "conflict"}:
            edge_id = graph.add_edge(
                f"Spec:{left}",
                relation.upper(),
                f"Spec:{right}",
                reason=str(pair.get("reason", "")),
                status="candidate",
            )
        events.append(
            {"batch": batch_number, "type": "pair_decision", **pair, "edge_id": edge_id}
        )
    return events
