"""CLI entry-point for ingesting raw datasets into the workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlopen

from creditrisk.config import Config, IngestionSourceConfig

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and validate raw data artifacts.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the experiment configuration YAML.",
    )
    return parser.parse_args()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _copy_filesystem(src: Path, dest: Path) -> None:
    if src.resolve() == dest.resolve():
        LOGGER.debug("Source and destination are the same (%s); skipping copy.", src)
        return
    _ensure_parent(dest)
    shutil.copy2(src, dest)


def _download_http(uri: str, dest: Path) -> None:
    _ensure_parent(dest)
    with urlopen(uri) as response, open(dest, "wb") as fh:
        shutil.copyfileobj(response, fh)


def _md5(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _record_result(
    source: IngestionSourceConfig,
    dest: Path,
    status: str,
    message: str | None = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "name": source.name,
        "output_path": str(dest),
        "status": status,
    }
    if message:
        payload["message"] = message
    if dest.exists() and dest.is_file():
        payload["size_bytes"] = dest.stat().st_size
        payload["checksum"] = _md5(dest)
    return payload


def _handle_source(source: IngestionSourceConfig, fail_on_missing: bool) -> Dict[str, object]:
    dest = source.output_path
    if source.type.lower() in {"existing", "validate"}:
        if not dest.exists():
            message = f"Expected file {dest} does not exist."
            if fail_on_missing:
                raise FileNotFoundError(message)
            LOGGER.warning(message)
            return _record_result(source, dest, status="missing", message=message)
        LOGGER.info("Validated presence of %s", dest)
        return _record_result(source, dest, status="validated")

    if not source.uri:
        raise ValueError(f"Source '{source.name}' requires a 'uri' field.")

    if source.type.lower() in {"filesystem", "local"}:
        src = Path(source.uri).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"Source file {src} does not exist for '{source.name}'.")
        LOGGER.info("Copying %s -> %s", src, dest)
        _copy_filesystem(src, dest)
    elif source.type.lower() in {"http", "https", "url"}:
        LOGGER.info("Downloading %s -> %s", source.uri, dest)
        _download_http(source.uri, dest)
    else:
        raise ValueError(f"Unsupported ingestion source type '{source.type}'.")

    return _record_result(source, dest, status="fetched")


def ingest_sources(config: Config) -> List[Dict[str, object]]:
    if not config.ingestion.enabled:
        LOGGER.info("Ingestion disabled via configuration; skipping.")
        return []

    results: List[Dict[str, object]] = []
    for source in config.ingestion.sources:
        try:
            result = _handle_source(source, config.ingestion.fail_on_missing)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Failed to ingest source '%s'", source.name)
            if config.ingestion.fail_on_missing:
                raise
            result = _record_result(source, source.output_path, status="failed", message=str(exc))
        results.append(result)
    return results


def write_ingestion_report(results: List[Dict[str, object]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"artifacts": results}
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Ingestion report written to %s", report_path)


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    results = ingest_sources(config)
    write_ingestion_report(results, config.paths.ingestion_report)


if __name__ == "__main__":
    main()
