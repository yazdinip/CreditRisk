"""Core package for the Home Credit default risk MLOps project."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("creditrisk")
except PackageNotFoundError:  # pragma: no cover - package metadata unavailable in dev installs
    __version__ = "0.0.0"

__all__ = ["__version__"]
