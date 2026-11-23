"""Offline-only MLX target/Judge and Qwen3 embedding adapters."""

import gc
import json
import os
import time
import uuid
from pathlib import Path
import numpy as np
from .common import (
    work_root,
    writable,
    cached_input,
    digest,
    read_json,
    write_json,
    write_array,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def local_model(repo_or_path):
    repo_or_path = os.fspath(repo_or_path)
    p = Path(repo_or_path).expanduser()
    if p.is_dir():
        if not (p / "config.json").is_file():
            raise FileNotFoundError(p / "config.json")
        return p.resolve()
    root = (
        Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
        / "hub"
        / ("models--" + repo_or_path.replace("/", "--"))
    )
    ref = root / "refs/main"
    if ref.exists():
        p = root / "snapshots" / ref.read_text().strip()
    else:
        snapshots = sorted((root / "snapshots").glob("*"))
        if not snapshots:
            raise FileNotFoundError(f"Local model unavailable: {repo_or_path}")
        p = snapshots[-1]
    if not (p / "config.json").exists():
        raise FileNotFoundError(p)
    return p


def parse_json(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            return json.loads(text[start : end + 1])
        raise


class LocalMLX:
    def __init__(
        self, model="mlx-community/Qwen3-8B-8bit", cache_dir=None, thinking=False
    ):
        self.path = local_model(model)
        self.model_id = (
            self.path.name if Path(model).expanduser().is_dir() else str(model)
        )
        self.thinking = thinking
        self.cache_dir = writable(
            cache_dir or work_root() / "artifacts/cache/generation"
        )
        self.model = None
        self.tokenizer = None
        self.identity = {
            "backend": "mlx-lm",
            "model": self.model_id,
            "revision": self.path.name,
            "config_hash": digest((self.path / "config.json").read_bytes()),
            "temperature": 0.0,
            "enable_thinking": thinking,
        }

    def load(self):
        if self.model is None:
            from mlx_lm import load

            self.model, self.tokenizer = load(str(self.path))

    def generate(self, messages, *, role="target", max_tokens=600, fresh=False):
        if self.thinking:
            max_tokens = max(max_tokens, 5000)
        key = digest(
            {
                "identity": self.identity,
                "messages": messages,
                "role": role,
                "max_tokens": max_tokens,
            }
        )
        path = self.cache_dir / (key + ".json")
        cached = cached_input(path)
        if cached.is_file() and not fresh:
            result = read_json(cached)
            return {**result, "cache_hit": True}
        self.load()
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(42)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        started = time.time()
        chunks = []
        last = None
        for last in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
        ):
            chunks.append(last.text)
        result = {
            "call_id": key[:20] + "-" + uuid.uuid4().hex[:12],
            "cache_key": key,
            "identity": self.identity,
            "role": role,
            "messages": messages,
            "answer": "".join(chunks).strip(),
            "seconds": round(time.time() - started, 3),
            "max_tokens": max_tokens,
            "generation_tokens": last.generation_tokens if last else 0,
            "finish_reason": last.finish_reason if last else "empty",
            "is_mock": False,
            "cache_hit": False,
        }
        write_json(path, result)
        write_json(
            work_root() / "artifacts/calls" / (result["call_id"] + ".json"), result
        )
        print(
            json.dumps(
                {
                    "event": "model_call",
                    "role": role,
                    "call_id": key[:12],
                    "seconds": result["seconds"],
                    "tokens": result["generation_tokens"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return result

    def json(self, messages, *, role, max_tokens=1200):
        result = self.generate(messages, role=role, max_tokens=max_tokens)
        return parse_json(result["answer"]), result


class ExperimentModels:
    """8B target, separate local 35B reviewer; no external inference services."""

    def __init__(
        self,
        target="mlx-community/Qwen3-8B-8bit",
        judge="mlx-community/Qwen3.6-35B-A3B-6bit",
        judge_thinking=False,
    ):
        self.target = LocalMLX(target)
        self.judge = LocalMLX(judge, thinking=judge_thinking)
        self.reviewer = LocalMLX(judge, thinking=False)
        self.identity = self.target.identity
        self.judge_identity = self.judge.identity

    def generate(self, messages, *, role="target", max_tokens=600, fresh=False):
        backend = (
            self.target
            if role == "target"
            else (self.judge if role == "judge" else self.reviewer)
        )
        # Batch grading uses the frozen, calibrated direct-JSON configuration.
        # Both reviewer roles share the same local weights in memory.
        if backend in (self.judge, self.reviewer) and backend.model is None:
            other = self.reviewer if backend is self.judge else self.judge
            if other.model is not None:
                backend.model, backend.tokenizer = other.model, other.tokenizer
        return backend.generate(messages, role=role, max_tokens=max_tokens, fresh=fresh)

    def json(self, messages, *, role, max_tokens=1200):
        result = self.generate(messages, role=role, max_tokens=max_tokens)
        return parse_json(result["answer"]), result


class Embedder:
    """Last-token pooling of the local Qwen3-Embedding hidden state, L2 normalized."""

    def __init__(self, model="Qwen/Qwen3-Embedding-0.6B", cache_dir=None):
        self.path = local_model(model)
        self.model_id = (
            self.path.name if Path(model).expanduser().is_dir() else str(model)
        )
        self.cache_dir = writable(
            cache_dir or work_root() / "artifacts/cache/embeddings"
        )
        self.model = None
        self.tokenizer = None
        self.identity = {
            "model": self.model_id,
            "revision": self.path.name,
            "pooling": "last_token_l2",
            "max_tokens": 1024,
            "backend": "mlx",
        }

    def load(self):
        if self.model is None:
            import mlx.core as mx
            from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model
            from transformers import AutoTokenizer

            # HF embedding checkpoint stores the backbone without a model. prefix.
            # Load all backbone weights strictly, rather than ignoring missing keys.
            self.model = Qwen3Model(
                ModelArgs.from_dict(read_json(self.path / "config.json"))
            )
            weights = {}
            for p in sorted(self.path.glob("*.safetensors")):
                weights.update(mx.load(p))
            self.model.load_weights(list(weights.items()), strict=True)
            mx.eval(self.model.parameters())
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.path), local_files_only=True
            )

    def encode(self, texts):
        vectors = []
        for i, text in enumerate(texts):
            path = self.cache_dir / (digest([self.identity, text]) + ".npy")
            cached = cached_input(path)
            if cached.is_file():
                value = np.load(cached, allow_pickle=False)
            else:
                import mlx.core as mx

                self.load()
                tokens = self.tokenizer.encode(text, add_special_tokens=False)[:1024]
                if not tokens:
                    tokens = [self.tokenizer.eos_token_id]
                hidden = self.model(mx.array([tokens]))[:, -1, :].astype(mx.float32)
                hidden = hidden / mx.maximum(
                    mx.linalg.norm(hidden, axis=-1, keepdims=True), 1e-12
                )
                mx.eval(hidden)
                value = np.array(hidden[0])
                write_array(path, value)
                mx.clear_cache()
            vectors.append(value)
            if len(texts) > 50 and (i + 1) % 100 == 0:
                print(
                    json.dumps(
                        {"event": "embedding", "done": i + 1, "total": len(texts)}
                    ),
                    flush=True,
                )
        return np.stack(vectors)

    def close(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
