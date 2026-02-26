"""FastAPI route definitions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import (
    QueryRequest, QueryResponse, CompareRequest, AnnotationRequest,
)

router = APIRouter()

# Services are injected via app state (set in app.py lifespan)
_query_service = None
_eval_service = None
_stats_service = None


def set_services(query_svc, eval_svc, stats_svc):
    global _query_service, _eval_service, _stats_service
    _query_service = query_svc
    _eval_service = eval_svc
    _stats_service = stats_svc


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Process a query through the HeteroRAG pipeline."""
    if _query_service is None:
        raise HTTPException(503, "Pipeline not initialized")

    result = _query_service.query(
        query=req.query,
        config=req.config,
        router_type=req.router_type,
        top_k=req.top_k,
        generate=req.generate,
    )
    return result


@router.post("/query/compare")
async def compare(req: CompareRequest):
    """Compare the same query across multiple configs."""
    if _query_service is None:
        raise HTTPException(503, "Pipeline not initialized")

    results = _query_service.compare(
        query=req.query,
        configs=req.configs,
        router_type=req.router_type,
        top_k=req.top_k,
    )
    return {"query": req.query, "configs": req.configs, "results": results}


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
        "baseline_configs": ["BM25"],
        "router_types": ["heuristic", "adaptive", "llm_zeroshot"],
    }


@router.get("/router/state")
async def router_state():
    """Get adaptive router state."""
    if _query_service is None:
        raise HTTPException(503, "Pipeline not initialized")
    pipeline = _query_service.pipeline
    if hasattr(pipeline._router, "stats"):
        return pipeline._router.stats
    return {"type": pipeline._router_type}


@router.post("/annotate")
async def annotate(req: AnnotationRequest):
    """Submit a human annotation."""
    if _eval_service is None:
        raise HTTPException(503, "Evaluation service not initialized")

    result = _eval_service.add_annotation(
        query_id=req.query_id,
        config=req.config,
        annotator_id=req.annotator_id,
        rci=req.rci,
        as_=req.as_,
        vm=req.vm,
        root_cause_category=req.root_cause_category,
        notes=req.notes,
    )
    return result


@router.get("/annotations")
async def annotations():
    """Get all annotations."""
    if _eval_service is None:
        return []
    return _eval_service.get_all_annotations()
