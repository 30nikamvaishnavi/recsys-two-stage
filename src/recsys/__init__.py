"""Two-stage personalized recommendation engine (retrieval + ranking)."""

import os

# TFRS 0.7.x needs Keras 2 semantics; harmless for TF 2.15 and protects against TF>=2.16.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

__version__ = "1.0.0"
