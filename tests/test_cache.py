import time

from recsys.cache.redis_cache import InMemoryBackend, RecCache, make_cache


def mk(ttl=900):
    return RecCache(InMemoryBackend(), ttl=ttl)


def test_set_get_and_hit_rate():
    c = mk()
    assert c.get_recs("dcn_v2", 1, 10) is None
    c.set_recs("dcn_v2", 1, 10, {"items": [1, 2]})
    assert c.get_recs("dcn_v2", 1, 10) == {"items": [1, 2]}
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1 and s["hit_rate"] == 0.5


def test_ttl_expiry():
    c = mk(ttl=1)
    c.set_recs("v", 1, 5, {"x": 1})
    assert c.get_recs("v", 1, 5) is not None
    time.sleep(1.1)
    assert c.get_recs("v", 1, 5) is None


def test_invalidate_user_removes_all_variants_and_ks_only_for_that_user():
    c = mk()
    for v in ("wide_deep", "dcn_v2"):
        for k in (5, 10):
            c.set_recs(v, 7, k, {"u": 7})
    c.set_recs("dcn_v2", 8, 10, {"u": 8})
    assert c.invalidate_user(7) == 5  # 4 recs + the index set
    assert c.get_recs("dcn_v2", 7, 10) is None and c.get_recs("wide_deep", 7, 5) is None
    assert c.get_recs("dcn_v2", 8, 10) == {"u": 8}


def test_lists_and_factory_fallback():
    c = mk()
    c.set_list("global", [1, 2, 3])
    assert c.get_list("global") == [1, 2, 3] and c.get_list("genre:Comedy") is None
    m = make_cache({"cache": {"redis_url": "redis://127.0.0.1:1/0"}})  # unreachable -> memory
    assert m.backend_name == "memory" and m.healthy()
