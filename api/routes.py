"""FastAPI route definitions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import (
    QueryRequest, QueryResponse, CompareRequest, AnnotationRequest,
)

router = APIRouter()

_NOT_READY = "Pipeline not initialized"
_EVAL_NOT_READY = "Evaluation service not initialized"

# Services are injected via app state (set in app.py lifespan)
_query_service = None
_eval_service = None
_stats_service = None


def set_services(query_svc, eval_svc, stats_svc):
    global _query_service, _eval_service, _stats_service
    _query_service = query_svc
    _eval_service = eval_svc
    _stats_service = stats_svc


def _require_pipeline():
    if _query_service is None:
        raise HTTPException(503, _NOT_READY)
    return _query_service


def _require_eval():
    if _eval_service is None:
        raise HTTPException(503, _EVAL_NOT_READY)
    return _eval_service


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Process a query through the HeteroRAG pipeline."""
    svc = _require_pipeline()
    return svc.query(
        query=req.query,
        config=req.config,
        router_type=req.router_type,
        top_k=req.top_k,
        generate=req.generate,
    )


@router.post("/query/compare")
async def compare(req: CompareRequest):
    """Compare the same query across multiple configs."""
    svc = _require_pipeline()
    results = svc.compare(
        query=req.query,
        configs=req.configs,
        router_type=req.router_type,
        top_k=req.top_k,
    )
    return {"query": req.query, "configs": req.configs, "results": results}


@router.post("/retrieve")
async def retrieve(req: QueryRequest):
    """Retrieve chunks without generating an answer."""
    svc = _require_pipeline()
    return svc.query(
        query=req.query,
        config=req.config,
        router_type=req.router_type,
        top_k=req.top_k,
        generate=False,
    )


@router.post("/route")
async def route(req: QueryRequest):
    """Get source routing boosts for a query (no retrieval)."""
    svc = _require_pipeline()
    pipeline = svc.pipeline
    query_embedding = pipeline._embedder.embed_query(req.query)
    boosts = pipeline._router.predict(req.query, query_embedding)
    return {"query": req.query, "source_boosts": boosts}


@router.get("/stats")
async def stats():
    """Get system statistics."""
    if _stats_service is None:
        return {}
    return _stats_service.get_all_stats()


@router.get("/configs")
async def configs():
    """List available experiment configurations."""
    return {
        "source_configs": ["D", "B", "W", "DB", "DW", "BW", "DBW"],
        "baseline_configs": ["BM25", "Naive", "PageIndex"],
        "router_types": ["heuristic", "adaptive", "llm_zeroshot"],
    }


@router.get("/router/state")
async def router_state():
    """Get adaptive router state."""
    svc = _require_pipeline()
    pipeline = svc.pipeline
    if hasattr(pipeline._router, "stats"):
        return pipeline._router.stats
    return {"type": pipeline._router_type}


@router.get("/router/history")
async def router_history():
    """Get adaptive router learning history."""
    import json
    import os
    history_path = os.path.join("data", "models", "adaptive_router", "history.jsonl")
    if not os.path.exists(history_path):
        return {"entries": []}
    entries = []
    with open(history_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return {"entries": entries[-100:]}  # Last 100 entries


@router.post("/evaluate")
async def evaluate(req: QueryRequest):
    """Run LLM judge evaluation on a single query result."""
    svc = _require_pipeline()
    result = svc.query(
        query=req.query,
        config=req.config,
        router_type=req.router_type,
        top_k=req.top_k,
        generate=True,
    )
    # Auto-score if eval service available
    if _eval_service and result.get("answer"):
        judge_result = _eval_service.judge_single(result)
        result["judge_scores"] = judge_result
    return result


@router.post("/annotate")
async def annotate(req: AnnotationRequest):
    """Submit a human annotation."""
    svc = _require_eval()
    return svc.add_annotation(
        query_id=req.query_id,
        config=req.config,
        annotator_id=req.annotator_id,
        rci=req.rci,
        as_=req.as_,
        vm=req.vm,
        root_cause_category=req.root_cause_category,
        notes=req.notes,
    )


@router.get("/annotations")
async def annotations():
    """Get all annotations."""
    if _eval_service is None:
        return []
    return _eval_service.get_all_annotations()
