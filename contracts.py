"""Validated request, record and algorithm interfaces."""

from typing import Any, Literal, Protocol
from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"


class Rule(BaseModel):
    rule_id: str
    text: str


class OriginalData(BaseModel):
    prompt: str
    historical_wrong_answer: str
    error_explanation: str
    reference_fact: str
    source: str


class Spec(BaseModel):
    applies_if: str
    general_rule: str
    boundaries: list[str] = Field(default_factory=list)
    safety_specifications: list[Rule]
    behavioral_specifications: list[Rule]


class Record(Contract):
    case_id: str
    spec_id: str
    original_data: OriginalData
    spec: Spec
    values: dict[str, Any]
    provenance: dict[str, Any]


class DataModule(Protocol):
    def prepare(self) -> list[dict]: ...
    def load(self) -> list[dict]: ...
    def score(self, case: dict, answer: str, model: Any) -> dict: ...


class GraphModule(Protocol):
    graph: dict  # Frozen baseline snapshot; callers edit only independent copies.

    def retrieve(
        self, query: str, top_k: int = 5, graph: dict | None = None
    ) -> dict: ...
    def context(
        self, query: str, graph: dict | None = None, gate: Any = None, top_k: int = 5
    ) -> dict: ...
    def values(
        self,
        records: list[dict],
        observations: dict,
        embedder: Any,
        model: Any,
        cases: list[dict],
    ) -> list[dict]: ...


class AttributionModule(Protocol):
    def run(self, cases: list[dict], case_ids: list[str]) -> dict: ...


class Algorithm1Request(Contract):
    operation: Literal["retrieve", "align", "represent"]
    query: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    case_id: str | None = None
    query_type: Literal["risk", "benign"] = "risk"


class Algorithm2Request(Contract):
    case_ids: list[str] = Field(min_length=1)
    operations: list[
        Literal["remove_content", "remove_edge", "rewire_edge", "restore", "repair"]
    ]
