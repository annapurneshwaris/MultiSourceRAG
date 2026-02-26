"""Pydantic request/response schemas for the FastAPI backend."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question")
    config: str = Field("DBW", description="Source config: D, B, W, DB, DW, BW, DBW, BM25")
    router_type: str | None = Field(None, description="Override router: heuristic, adaptive, llm_zeroshot")
    top_k: int = Field(10, description="Number of final results")
    generate: bool = Field(True, description="Whether to generate LLM answer")


class ChunkResponse(BaseModel):
    chunk_id: str
    source_type: str
    source_id: str
    source_url: str
    feature_area: str
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    config: str
    active_sources: list[str]
    source_boosts: dict[str, float]
    metadata_hints: dict
    retrieved_count: int
    reranked_chunks: list[ChunkResponse]
    answer: str
    citations: dict[str, list[str]]
    timing: dict[str, float]
    router_stats: dict


class CompareRequest(BaseModel):
    query: str = Field(..., description="User question")
    configs: list[str] = Field(["D", "DB", "DBW"], description="Configs to compare")
    router_type: str | None = Field(None)
    top_k: int = Field(10)


class AnnotationRequest(BaseModel):
    query_id: str
    config: str
    annotator_id: str
    rci: int = Field(..., ge=0, le=2)
    as_: int = Field(..., ge=0, le=2, alias="as")
    vm: int = Field(..., ge=0, le=2)
    root_cause_category: str = "unknown"
    notes: str = ""

    class Config:
        populate_by_name = True
