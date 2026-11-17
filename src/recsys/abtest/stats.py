"""Statistics for the simulated online experiment."""

from __future__ import annotations

import hashlib
import math

import numpy as np


def bucket(user_id: int, salt: str, n_buckets: int = 2) -> int:
    """Deterministic, uniform, salt-isolated assignment (same user -> same arm across requests)."""
    h = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
    return int(h[:12], 16) % n_buckets


def two_proportion_ztest(c1: int, n1: int, c2: int, n2: int) -> dict:
    """Pooled two-sided z-test for H0: p1 == p2."""
    p1, p2 = c1 / n1, c2 / n2
    p = (c1 + c2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    return {
        "p1": p1,
        "p2": p2,
        "diff": p2 - p1,
        "z": z,
        "p_value": math.erfc(abs(z) / math.sqrt(2)),
    }


def bootstrap_ci(
    x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """User-level bootstrap of mean(b) - mean(a) (arms contain different users -> independent resampling)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    da = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(1)
    db = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(1)
    d = db - da
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": float(b.mean() - a.mean()), "ci_low": float(np.quantile(d, alpha / 2)), "ci_high": float(np.quantile(d, 1 - alpha / 2)),
            "p_value_bootstrap": float(min(p, 1.0))}  # fmt: skip


def min_detectable_effect(
    p0: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Approx. smallest absolute CTR lift detectable with n impressions per arm."""
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return z * math.sqrt(2 * p0 * (1 - p0) / max(n_per_arm, 1))


def aa_false_positive_rate(
    per_user: np.ndarray, n_splits: int = 300, alpha: float = 0.05, seed: int = 0
) -> float:
    """Randomly split ONE arm in two and test: the rejection rate should be ~alpha (sanity check)."""
    from scipy import stats

    rng = np.random.default_rng(seed)
    per_user = np.asarray(per_user, float)
    rej = 0
    for _ in range(n_splits):
        perm = rng.permutation(len(per_user))
        h = len(per_user) // 2
        rej += (
            stats.ttest_ind(per_user[perm[:h]], per_user[perm[h:]], equal_var=False).pvalue < alpha
        )
    return rej / n_splits


def decide(ctr: dict, ndcg: dict, alpha: float = 0.05) -> dict:
    """Ship B only if CTR lift is significantly positive AND NDCG@10 is not significantly worse."""
    if ctr["ci_low"] > 0 and ndcg["ci_high"] >= 0:  # NDCG not significantly worse
        d, why = "SHIP_B", "CTR lift CI is entirely above 0 and NDCG@10 does not regress."
    elif ctr["ci_high"] < 0:
        d, why = "KEEP_A", "B has significantly lower CTR."
    else:
        d, why = (
            "INCONCLUSIVE",
            "CTR difference CI includes 0; collect more traffic (see MDE) or keep A.",
        )
    return {"decision": d, "reason": why, "alpha": alpha}
