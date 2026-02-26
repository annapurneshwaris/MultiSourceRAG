"""FastAPI application with lifespan management.

Usage:
    uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router, set_services
from api.services.query_service import QueryService
from api.services.evaluation_service import EvaluationService
from api.services.stats_service import StatsService

# Global services
query_service = QueryService()
eval_service = EvaluationService()
stats_service = StatsService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup, save router on shutdown."""
    print("Starting HeteroRAG API...")

    # Initialize pipeline (loads embedder, indices, router)
    try:
        query_service.initialize()
        print("Pipeline initialized successfully.")
    except Exception as e:
        print(f"WARNING: Pipeline initialization failed: {e}")
        print("API will start but /query endpoints will return 503.")

    # Wire services to routes
    set_services(query_service, eval_service, stats_service)

    yield

    # Shutdown: save router state
    print("Shutting down HeteroRAG API...")
    try:
        pipeline = query_service.pipeline
        if hasattr(pipeline._router, "save_state"):
            pipeline._router.save_state("models/adaptive_router")
            print("Router state saved.")
    except Exception:
        pass


app = FastAPI(
    title="HeteroRAG API",
    description="Multi-Source Heterogeneous RAG for VS Code Technical Support",
    version="0.4.0",
    lifespan=lifespan,
)

# CORS (allow Gradio frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
