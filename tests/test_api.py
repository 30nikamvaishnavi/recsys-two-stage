import pytest

pytest_plugins = ("test_pipeline",)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("prometheus_client")
from fastapi.testclient import TestClient  # noqa: E402
from recsys.serving.api import create_app  # noqa: E402
from test_pipeline import ext_id, make_rec, warm_user  # noqa: E402


def client(stack):
    return TestClient(create_app(make_rec(stack)))


def test_health_recommend_feedback_metrics(stack):
    c = client(stack)
    assert c.get("/health").json()["status"] == "ok"
    u = ext_id(stack, warm_user(stack))
    r = c.post("/recommend", json={"user_id": u, "k": 5, "model_variant": "wide_deep"})
    assert (
        r.status_code == 200
        and len(r.json()["items"]) == 5
        and r.json()["meta"]["cache_hit"] is False
    )
    assert (
        c.post("/recommend", json={"user_id": u, "k": 5, "model_variant": "wide_deep"}).json()[
            "meta"
        ]["cache_hit"]
        is True
    )
    item = r.json()["items"][0]["item_id"]
    fb = c.post(
        "/feedback",
        json={
            "user_id": u,
            "item_id": item,
            "event": "click",
            "position": 0,
            "variant": "wide_deep",
        },
    )
    assert fb.status_code == 200 and fb.json()["cache_entries_invalidated"] >= 1
    m = c.get("/metrics").text
    assert "recsys_requests_total" in m and "recsys_stage_latency_ms_bucket" in m
    assert c.get("/cache/stats").json()["hits"] >= 1


def test_validation_errors_and_similar(stack):
    c = client(stack)
    assert c.post("/recommend", json={"user_id": 1, "k": 0}).status_code == 422
    assert c.post("/recommend", json={"user_id": 1, "model_variant": "xyz"}).status_code == 422
    assert c.get("/similar-items", params={"item_id": -5}).status_code == 404
    assert c.get("/recommend", params={"user_id": 10**9, "genres": "Comedy"}).status_code == 200
    assert c.get("/openapi.json").status_code == 200
