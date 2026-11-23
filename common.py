"""Portable resource paths, content hashes and atomic output helpers."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


def resource_root():
    return (
        Path(os.environ.get("SPECALGO_RESOURCES", "resources")).expanduser().resolve()
    )


def work_root():
    work = Path(os.environ.get("SPECALGO_WORK_DIR", "runs")).expanduser().resolve()
    assets = resource_root()
    if work.is_relative_to(assets) or assets.is_relative_to(work):
        raise ValueError("Choose separate directories for resources and run outputs.")
    return work


def writable(path):
    resolved = Path(path).expanduser().resolve()
    if resolved.is_relative_to(resource_root()):
        raise ValueError(
            "The resource directory is read-only. Choose a different output path."
        )
    return resolved


def cached_input(path):
    """Prefer completed work and fall back to a matching cache supplied with the inputs."""
    current = Path(path).resolve()
    work = work_root()
    if current.is_file() or not current.is_relative_to(work):
        return current
    return resource_root() / current.relative_to(work)


def safe_component(value):
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\0" in value
    ):
        raise ValueError("Invalid identifier for an output filename")
    return value


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    return hashlib.sha256(value).hexdigest()


def protocol_fingerprint():
    package = Path(__file__).resolve().parent
    return digest(
        {
            p.relative_to(package).as_posix(): digest(p.read_bytes())
            for p in package.rglob("*")
            if p.suffix in {".py", ".txt"}
        }
    )


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


@contextmanager
def atomic_output(path, *, binary=False):
    target = writable(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(
        mode="wb" if binary else "w",
        encoding=None if binary else "utf-8",
        dir=target.parent,
        prefix=".writing-",
        delete=False,
    )
    scratch = Path(stream.name)
    try:
        with stream:
            yield stream
        scratch.replace(target)
    finally:
        scratch.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_output(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_rows(path, rows):
    with atomic_output(path) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_array(path, value):
    import numpy as np

    with atomic_output(path, binary=True) as stream:
        np.save(stream, value, allow_pickle=False)
