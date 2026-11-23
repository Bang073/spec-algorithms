"""Validate the private input snapshot used for a path intervention."""

from pathlib import Path

from .common import digest, read_json, resource_root
from .graph import RETRIEVAL_CONFIG


def verify_freeze(engine, model):
    root = resource_root()
    manifest = read_json(root / "artifacts/freeze/manifest.json")
    freeze_id = manifest["freeze_id"]
    if digest({k: v for k, v in manifest.items() if k != "freeze_id"}) != freeze_id:
        raise ValueError("The resource freeze manifest has changed")

    files = manifest["files"]
    required = {
        "data/train.jsonl",
        "data/splits/manifest.json",
        "artifacts/graph/base.json",
    }
    if not required.issubset(files):
        raise ValueError(
            "The freeze must cover the training records, split manifest and graph"
        )
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Invalid path in the freeze manifest")
        if relative.parts[0] in {"scripts", "src"}:
            # Older resource packs also recorded the private build's source hashes.
            # This distribution reports its own executable fingerprint separately.
            continue
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or digest(path.read_bytes()) != expected
        ):
            raise ValueError(f"Frozen resource missing or changed: {name}")

    expected = {
        "graph_hash": digest(engine.graph),
        "train_hash": engine.index.identity["train_hash"],
        "retrieval": RETRIEVAL_CONFIG,
        "target_model": model.identity,
        "judge_model": model.judge_identity,
        "embedding": engine.index.identity["embedding"],
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"Active configuration differs from the resource freeze: {field}"
            )
    if engine.graph["meta"]["train_hash"] != expected["train_hash"]:
        raise ValueError("Graph and training split differ")
    return manifest
