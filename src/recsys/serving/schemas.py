"""Pydantic request/response models (drive the OpenAPI docs at /docs)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="External MovieLens userId. Unknown ids are treated as new (cold-start) users.")
    k: int = Field(10, ge=1, le=100)
    model_variant: Literal["wide_deep", "dcn_v2"] = "dcn_v2"
    onboarding_genres: list[str] | None = Field(None, description="Genres picked at signup (used for brand-new users).")
    use_cache: bool = True


class ItemOut(BaseModel):
    item_id: int
    title: str
    genres: str
    score: float
    retrieval_score: float | None = None
    ranker_score: float | None = None


class RecommendMeta(BaseModel):
    strategy: str
    reason_code: str
    cache_hit: bool
    backend: str
    latency_ms: dict[str, float]


class RecommendResponse(BaseModel):
    user_id: int
    k: int
    variant: str
    items: list[ItemOut]
    meta: RecommendMeta


class SimilarItem(BaseModel):
    item_id: int
    title: str
    genres: str
    similarity: float
    is_cold_item: bool


class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int
    event: Literal["impression", "click", "rating"]
    rating: float | None = Field(None, ge=0.5, le=5.0)
    position: int | None = Field(None, ge=0)
    variant: str | None = None


class FeedbackResponse(BaseModel):
    logged: bool
    cache_entries_invalidated: int
