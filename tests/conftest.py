import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from helpers import build_world  # noqa: E402


@pytest.fixture(scope="session")
def world():
    return build_world()
