"""Redis cache for recommendations, popularity lists and cold-start lists.

Keys:  rec:{variant}:{user}:{k}   JSON, TTL        pop:global / pop:genre:{g}   JSON, longer TTL
       recidx:{user}              SET of rec keys   -> O(1) invalidation when the user logs a new event
       stats:hits / stats:misses  counters
Falls back to an in-process dict backend (same interface) when Redis is unreachable or in tests.
"""
from __future__ import annotations

import json
import os
import time

from ..utils import get_logger

log = get_logger("cache")


class InMemoryBackend:
    """Minimal subset of redis-py used by RecCache (get/set(ex)/delete/sadd/smembers/incr/flushdb)."""

    def __init__(self):
        self.kv: dict[str, tuple[str, float | None]] = {}
        self.sets: dict[str, set] = {}

    def _alive(self, k):
        v = self.kv.get(k)
        if v and v[1] is not None and v[1] < time.time():
            del self.kv[k]
            return None
        return v

    def get(self, k):
        v = self._alive(k)
        return v[0] if v else None

    def set(self, k, val, ex=None):
        self.kv[k] = (str(val), time.time() + ex if ex else None)

    def delete(self, *keys):
        n = 0
        for k in keys:
            n += (self.kv.pop(k, None) is not None) + (self.sets.pop(k, None) is not None)
        return n

    def sadd(self, k, *v):
        self.sets.setdefault(k, set()).update(v)

    def smembers(self, k):
        return set(self.sets.get(k, set()))

    def expire(self, k, s):  # sets keep no TTL in the fake
        return True

    def incr(self, k):
        v = int(self.get(k) or 0) + 1
        self.set(k, v)
        return v

    def ping(self):
        return True

    def flushdb(self):
        self.kv.clear()
        self.sets.clear()


class RecCache:
    def __init__(self, backend, ttl: int = 900, pop_ttl: int = 3600, name: str = "memory"):
        self.r, self.ttl, self.pop_ttl, self.backend_name = backend, ttl, pop_ttl, name

    @staticmethod
    def rec_key(variant: str, user: int, k: int) -> str:
        return f"rec:{variant}:{user}:{k}"

    def get_recs(self, variant: str, user: int, k: int):
        raw = self.r.get(self.rec_key(variant, user, k))
        self.r.incr("stats:hits" if raw else "stats:misses")
        return json.loads(raw) if raw else None

    def set_recs(self, variant: str, user: int, k: int, payload: dict) -> None:
        key = self.rec_key(variant, user, k)
        self.r.set(key, json.dumps(payload), ex=self.ttl)
        self.r.sadd(f"recidx:{user}", key)

    def invalidate_user(self, user: int) -> int:
        keys = [k.decode() if isinstance(k, bytes) else k for k in self.r.smembers(f"recidx:{user}")]
        n = self.r.delete(*keys, f"recidx:{user}") if keys else 0
        return int(n)

    def get_list(self, name: str):
        raw = self.r.get(f"pop:{name}")
        return json.loads(raw) if raw else None

    def set_list(self, name: str, items: list) -> None:
        self.r.set(f"pop:{name}", json.dumps(items), ex=self.pop_ttl)

    def stats(self) -> dict:
        h, m = int(self.r.get("stats:hits") or 0), int(self.r.get("stats:misses") or 0)
        return {"hits": h, "misses": m, "hit_rate": h / (h + m) if h + m else 0.0, "backend": self.backend_name}

    def healthy(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False


def make_cache(cfg=None, url: str | None = None, force_memory: bool = False) -> RecCache:
    c = (cfg or {}).get("cache", {}) if cfg else {}
    ttl, pttl = c.get("ttl_seconds", 900), c.get("pop_ttl_seconds", 3600)
    url = url or os.environ.get("REDIS_URL") or c.get("redis_url")
    if not force_memory and url:
        try:
            import redis

            r = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=1.0)
            r.ping()
            return RecCache(r, ttl, pttl, "redis")
        except Exception as e:
            log.warning("Redis unavailable (%s) -> in-memory cache", e)
    return RecCache(InMemoryBackend(), ttl, pttl, "memory")
