import numpy as np
import pandas as pd
from recsys.features.pandas_features import user_state
from recsys.serving.all_states import user_state_all


def _inter() -> pd.DataFrame:
    rows = [  # user_idx, item_idx, rating, ts
        (1, 1, 5.0, 10), (1, 2, 4.0, 20), (1, 3, 2.0, 30), (1, 4, 5.0, 150),  # u1: has a target
        (2, 5, 4.5, 15), (2, 6, 4.0, 25), (2, 7, 3.0, 35),  # u2: history only, no target
        (3, 8, 5.0, 200),  # u3: first activity AFTER the cutoff -> unknown at serving time
    ]  # fmt: skip
    df = pd.DataFrame(rows, columns=["user_idx", "item_idx", "rating", "ts"])
    df["is_pos"] = df["rating"] >= 4.0
    return df


def test_all_users_with_history_are_present_even_without_targets():
    inter, cutoff, end, n = _inter(), 100, 300, 4
    windowed = user_state(inter, cutoff, end, n)
    everyone = user_state_all(inter, cutoff, n)
    assert set(windowed["user_idx"]) == {1, 3}  # the old behaviour: only users with targets
    assert set(everyone["user_idx"]) == {1, 2}  # all users that have history before the cutoff
    assert "targets" not in everyone.columns


def test_all_users_state_matches_windowed_state_for_shared_users():
    inter, cutoff, n = _inter(), 100, 4
    a = user_state(inter, cutoff, 300, n).set_index("user_idx").loc[[1]]
    b = user_state_all(inter, cutoff, n).set_index("user_idx").loc[[1]]
    for col in ("n_all", "n_pos", "avg_rating", "days_since_last"):
        assert np.isclose(float(a[col].iloc[0]), float(b[col].iloc[0])), col
    assert list(a["hist_items"].iloc[0]) == list(b["hist_items"].iloc[0])
    assert list(b["hist_items"].iloc[0]) == [1, 2, 0, 0]  # oldest -> newest, zero padded at the END


def test_history_is_cut_to_the_most_recent_positives():
    inter = pd.DataFrame(
        {
            "user_idx": [1] * 6,
            "item_idx": [1, 2, 3, 4, 5, 6],
            "rating": [5.0] * 6,
            "ts": [1, 2, 3, 4, 5, 6],
        }
    )
    inter["is_pos"] = True
    out = user_state_all(inter, 100, 3)
    assert list(out["hist_items"].iloc[0]) == [4, 5, 6]
    assert int(out["n_pos"].iloc[0]) == 6
