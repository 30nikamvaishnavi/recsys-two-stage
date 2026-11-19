"""One-off patch: serve from states of ALL users instead of only users with test-window positives.

Run from the repo root:  python scripts/apply_serving_state_fix.py
It edits src/recsys/serving/export.py and assets.py by pattern (so it works whether or not
`make format` already re-wrapped the lines) and refuses to change anything if a pattern is missing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORT = ROOT / "src/recsys/serving/export.py"
ASSETS = ROOT / "src/recsys/serving/assets.py"

EXPORT_RE = re.compile(
    r"(?P<indent>[ \t]*)pd\.read_parquet\(\s*Path\(path\(cfg, \"features\"\)\)"
    r"\s*/\s*\"user_state_test\.parquet\"\s*\)"
    r"\.to_parquet\(\s*sdir\s*/\s*\"user_states\.parquet\",\s*index=False\s*\)\n"
)
ASSETS_OLD = 'pd.read_parquet(s / "user_states.parquet")'
ASSETS_NEW = (
    'pd.read_parquet(s / ("user_states_all.parquet" if (s / "user_states_all.parquet").exists() '
    'else "user_states.parquet"))'
)


def main() -> int:
    export_src, assets_src = EXPORT.read_text(), ASSETS.read_text()
    if "user_states_all" in export_src or "user_states_all" in assets_src:
        print("already patched; nothing to do")
        return 0
    matches = list(EXPORT_RE.finditer(export_src))
    if len(matches) != 1 or assets_src.count(ASSETS_OLD) != 1:
        print("pattern not found exactly once; NOT changing anything. Send the lines to Claude:")
        print("  grep -n 'user_state' src/recsys/serving/export.py src/recsys/serving/assets.py")
        return 1
    m = matches[0]
    ind = m.group("indent")
    extra = (
        f"{ind}from .all_states import user_state_all\n\n"
        f"{ind}everyone = user_state_all(\n"
        f'{ind}    fs.interactions(["user_idx", "item_idx", "rating", "ts", "is_pos"]),\n'
        f'{ind}    fs.cutoff("test"),\n'
        f"{ind}    fs.N,\n"
        f"{ind})\n"
        f'{ind}everyone.to_parquet(sdir / "user_states_all.parquet", index=False)\n'
    )
    export_src = export_src[: m.end()] + extra + export_src[m.end() :]
    EXPORT.write_text(export_src)
    ASSETS.write_text(assets_src.replace(ASSETS_OLD, ASSETS_NEW))
    print("patched export.py and assets.py; now run: make format && make lint && make test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
