"""FastAPI gateway.  uvicorn recsys.serving.api:app --host 0.0.0.0 --port 8080"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from ..config import load_config
from ..utils import get_logger
from .schemas import (
    FeedbackRequest,
    FeedbackResponse,
    RecommendRequest,
    RecommendResponse,
    SimilarItem,
)

log = get_logger("api")


def create_app(recommender=None) -> FastAPI:
    """Pass a Recommender to inject one (tests); otherwise it is loaded at startup."""
    registry = CollectorRegistry()
    stage_h = Histogram("recsys_stage_latency_ms", "Per-stage latency (ms)", ["stage"], registry=registry,
                        buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000))  # fmt: skip
    req_c = Counter(
        "recsys_requests_total", "Requests", ["variant", "strategy", "cache_hit"], registry=registry
    )
    state = {"rec": recommender}

    @asynccontextmanager
    async def lifespan(app):
        if state["rec"] is None:
            from .assets import load_recommender

            state["rec"] = load_recommender(load_config())
        yield

    app = FastAPI(title="Two-stage Recommender API", version="1.0.0", lifespan=lifespan,
                  description="Retrieval (two-tower + FAISS) -> ranking (DCN-V2 / Wide&Deep on Triton) with cold-start fallbacks and Redis caching.")  # fmt: skip

    def rec():
        if state["rec"] is None:
            raise HTTPException(503, "recommender not loaded")
        return state["rec"]

    def _recommend(req: RecommendRequest):
        try:
            out = rec().recommend(
                req.user_id, req.k, req.model_variant, req.onboarding_genres, req.use_cache
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        m = out["meta"]
        req_c.labels(req.model_variant, m["strategy"], str(m["cache_hit"]).lower()).inc()
        for s, v in m["latency_ms"].items():
            stage_h.labels(s).observe(v)
        return out

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend_post(req: RecommendRequest):
        return _recommend(req)

    @app.get("/recommend", response_model=RecommendResponse)
    def recommend_get(
        user_id: int,
        k: int = Query(10, ge=1, le=100),
        model_variant: str = "dcn_v2",
        genres: str | None = None,
    ):
        gs = [g for g in (genres or "").split(",") if g] or None
        if model_variant not in ("wide_deep", "dcn_v2"):
            raise HTTPException(422, "model_variant must be wide_deep or dcn_v2")
        return _recommend(
            RecommendRequest(
                user_id=user_id, k=k, model_variant=model_variant, onboarding_genres=gs
            )
        )

    @app.get("/similar-items", response_model=list[SimilarItem])
    def similar(item_id: int, k: int = Query(10, ge=1, le=100)):
        try:
            return rec().similar_items(item_id, k)
        except KeyError as e:
            raise HTTPException(404, f"unknown item_id {item_id}") from e

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(fb: FeedbackRequest):
        return rec().feedback(fb.user_id, fb.item_id, fb.event, fb.rating, fb.position, fb.variant)

    @app.get("/health")
    def health():
        r = state["rec"]
        if r is None:
            return {"status": "starting"}
        ready = r.backend.ready()
        return {"status": "ok" if ready else "degraded", "model_backend": getattr(r.backend, "name", "?"), "model_backend_ready": ready,
                "cache": r.cache.backend_name, "cache_healthy": r.cache.healthy(), "n_items": r.table.n_items}  # fmt: skip

    @app.get("/cache/stats")
    def cache_stats():
        return rec().cache.stats()

    @app.get("/genres")
    def genres():
        return rec().genre_options()

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
