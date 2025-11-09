from __future__ import annotations

from creditrisk.config import TestingConfig

# Prevent pytest from trying to collect TestingConfig as a test case.
TestingConfig.__test__ = False  # type: ignore[attr-defined]
