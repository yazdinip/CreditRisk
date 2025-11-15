"""CLI entry-point for ingesting raw datasets into the workspace."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse
from urllib.request import urlopen


from creditrisk.config import Config, IngestionSourceConfig

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_TEMP_DIR_PREFIX = "creditrisk-ingest-"


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


def _temporary_file(dest: Path, suffix: str = "") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=_TEMP_DIR_PREFIX, dir=dest.parent)
    handle.close()
    return Path(handle.name)


def _materialize_download(temp_path: Path, dest: Path, force_decompress: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _is_archive(path: Path) -> bool:
        if not path.exists():
            return False
        if zipfile.is_zipfile(path) or tarfile.is_tarfile(path):
            return True
        return path.suffix in {".gz", ".bz2", ".xz"}

    should_decompress = force_decompress or _is_archive(temp_path)
    if should_decompress:
        _extract_single_artifact(temp_path, dest)
    else:
        shutil.move(temp_path, dest)


def _extract_single_artifact(archive: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            if len(members) != 1:
                raise ValueError(
                    f"Archive {archive} contains {len(members)} files; "
                    "specify 'decompress: false' if you want to keep the archive."
                )
            member = members[0]
            extracted_path = Path(zf.extract(member, path=archive.parent))
        final_path = archive.parent / member
        shutil.move(final_path, dest)
        archive.unlink(missing_ok=True)
        return

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            if len(members) != 1:
                raise ValueError(
                    f"Archive {archive} contains {len(members)} files; "
                    "specify 'decompress: false' if you want to keep the archive."
                )
            member = members[0]
            tf.extract(member, path=archive.parent)
            extracted_path = archive.parent / member.name
        shutil.move(extracted_path, dest)
        archive.unlink(missing_ok=True)
        return

    if archive.suffix == ".gz" and not tarfile.is_tarfile(archive):
        with gzip.open(archive, "rb") as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
        archive.unlink(missing_ok=True)
        return

    shutil.move(archive, dest)


def _enforce_checksum(path: Path, expected_checksum: str) -> None:
    actual = _md5(path)
    if actual.lower() != expected_checksum.lower():
        raise ValueError(
            f"Checksum mismatch for {path}. Expected {expected_checksum}, observed {actual}."
        )


def _cleanup_temp_dir(path: Path) -> None:
    if path.name.startswith(_TEMP_DIR_PREFIX):
        shutil.rmtree(path, ignore_errors=True)


def _download_kaggle_artifact(source: IngestionSourceConfig) -> Path:
    try:
        from kaggle import KaggleApi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Kaggle support requires the 'kaggle' package. "
            "Install it via 'pip install kaggle' and configure your API credentials."
        ) from exc

    options = source.options
    dataset = options.get("dataset")
    competition = options.get("competition")
    reference = options.get("ref")
    if reference and not dataset and not competition:
        if reference.startswith("competition:"):
            competition = reference.split(":", 1)[1]
        else:
            dataset = reference
    if source.uri and not dataset and not competition:
        # Allow shorthand `uri: owner/dataset` or `uri: competition/<name>`
        if source.uri.startswith("competition/"):
            competition = source.uri.split("/", 1)[1]
        else:
            dataset = source.uri
    if dataset and competition:
        raise ValueError(f"Ingestion source '{source.name}' cannot specify both dataset and competition.")
    if not dataset and not competition:
        raise ValueError(
            f"Ingestion source '{source.name}' of type '{source.type}' requires "
            "an 'options.dataset', 'options.competition', or 'uri'."
        )

    file_name = options.get("file") or options.get("filename") or source.output_path.name
    force = bool(options.get("force", True))
    quiet = bool(options.get("quiet", True))

    source.output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=_TEMP_DIR_PREFIX, dir=source.output_path.parent))

    api = KaggleApi()
    api.authenticate()
    try:
        if dataset:
            LOGGER.info("Downloading %s from Kaggle dataset %s", file_name, dataset)
            downloaded = Path(
                api.dataset_download_file(
                    dataset, file_name, path=str(temp_dir), force=force, quiet=quiet
                )
            )
        else:
            LOGGER.info("Downloading %s from Kaggle competition %s", file_name, competition)
            downloaded = Path(
                api.competition_download_file(
                    competition, file_name, path=str(temp_dir), force=force, quiet=quiet
                )
            )
    except Exception:  # pragma: no cover - Kaggle API failure
        _cleanup_temp_dir(temp_dir)
        raise

    if not downloaded.exists():
        # Kaggle saves `<filename>.zip` when the API returns archives.
        zipped = temp_dir / f"{file_name}.zip"
        if zipped.exists():
            downloaded = zipped
        else:
            candidate = temp_dir / file_name
            if candidate.exists():
                downloaded = candidate
            else:
                _cleanup_temp_dir(temp_dir)
                raise FileNotFoundError(
                    f"Kaggle download for '{source.name}' did not produce {file_name}."
                )

    # Caller is responsible for moving/decompressing; we keep the temp dir for cleanup later.
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    return downloaded


def _download_s3_object(source: IngestionSourceConfig) -> Path:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "S3 ingestion requires the 'boto3' package. Install it via 'pip install boto3'."
        ) from exc

    options = source.options
    bucket = options.get("bucket")
    key = options.get("key")
    if source.uri:
        parsed = urlparse(source.uri)
        if parsed.scheme == "s3":
            bucket = bucket or parsed.netloc
            key = key or parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(
            f"Ingestion source '{source.name}' of type '{source.type}' requires "
            "a bucket/key pair either via 'options' or the 'uri' field (s3://bucket/key)."
        )

    session_kwargs = {}
    profile = options.get("profile") or options.get("aws_profile")
    if profile:
        session_kwargs["profile_name"] = profile
    region = options.get("region")

    session = boto3.Session(**session_kwargs)
    client_kwargs = {}
    if region:
        client_kwargs["region_name"] = region
    if options.get("endpoint_url"):
        client_kwargs["endpoint_url"] = options["endpoint_url"]
    if options.get("aws_access_key_id"):
        client_kwargs["aws_access_key_id"] = options["aws_access_key_id"]
        client_kwargs["aws_secret_access_key"] = options.get("aws_secret_access_key")
        client_kwargs["aws_session_token"] = options.get("aws_session_token")
    client = session.client("s3", **client_kwargs)

    temp_path = _temporary_file(source.output_path, suffix=".s3")
    extra_args = {}
    if options.get("version_id"):
        extra_args["VersionId"] = options["version_id"]

    LOGGER.info("Downloading s3://%s/%s", bucket, key)
    download_kwargs = {"ExtraArgs": extra_args} if extra_args else {}
    client.download_file(bucket, key, str(temp_path), **download_kwargs)
    return temp_path


def _download_azure_blob(source: IngestionSourceConfig) -> Path:
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Azure Blob ingestion requires 'azure-storage-blob'. "
            "Install it via 'pip install azure-storage-blob'."
        ) from exc

    options = source.options
    container = options.get("container")
    blob_name = options.get("blob") or options.get("key")
    if source.uri:
        parsed = urlparse(source.uri)
        if parsed.scheme in {"azure", "wasbs", "abfs"}:
            container = container or parsed.netloc
            blob_name = blob_name or parsed.path.lstrip("/")
    if not container or not blob_name:
        raise ValueError(
            f"Ingestion source '{source.name}' of type '{source.type}' must supply "
            "'options.container' and 'options.blob' (or a URI like azure://container/blob)."
        )

    connection_string = options.get("connection_string") or os.getenv(
        "AZURE_STORAGE_CONNECTION_STRING"
    )
    credential = options.get("credential") or options.get("sas_token") or os.getenv(
        "AZURE_STORAGE_SAS_TOKEN"
    )
    account_url = options.get("account_url")

    if connection_string:
        service = BlobServiceClient.from_connection_string(connection_string)
    else:
        if not account_url:
            account_name = options.get("account_name")
            if not account_name:
                raise ValueError(
                    f"Azure source '{source.name}' requires either 'connection_string' "
                    "or ('account_url' / 'account_name')."
                )
            account_url = f"https://{account_name}.blob.core.windows.net"
        service = BlobServiceClient(account_url=account_url, credential=credential)

    blob_client = service.get_blob_client(container=container, blob=blob_name)
    temp_path = _temporary_file(source.output_path, suffix=".azure")
    LOGGER.info("Downloading azure://%s/%s", container, blob_name)
    stream = blob_client.download_blob()
    with open(temp_path, "wb") as fh:
        stream.readinto(fh)  # type: ignore[arg-type]
    return temp_path


def _pull_via_dvc(source: IngestionSourceConfig) -> None:
    target = source.uri or source.options.get("target") or str(source.output_path)
    LOGGER.info("Pulling %s via DVC", target)
    try:
        subprocess.run(
            ["dvc", "pull", target],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"DVC pull failed for target '{target}'. stderr: {exc.stderr}"
        ) from exc


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
        "source_type": source.type,
    }
    if message:
        payload["message"] = message
    if source.uri:
        payload["uri"] = source.uri
    if source.checksum:
        payload["expected_checksum"] = source.checksum
    if dest.exists() and dest.is_file():
        payload["size_bytes"] = dest.stat().st_size
        payload["checksum"] = _md5(dest)
    return payload


def _handle_source(source: IngestionSourceConfig, fail_on_missing: bool) -> Dict[str, object]:
    dest = source.output_path
    source_type = source.type.lower()

    if (
        dest.exists()
        and dest.is_file()
        and source.skip_if_exists
        and source_type not in {"existing", "validate"}
    ):
        LOGGER.info(
            "Destination %s already exists for source '%s'; skipping fetch.",
            dest,
            source.name,
        )
        if source.checksum:
            _enforce_checksum(dest, source.checksum)
        return _record_result(source, dest, status="cached")

    if source_type in {"existing", "validate"}:
        if not dest.exists():
            message = f"Expected file {dest} does not exist."
            if fail_on_missing:
                raise FileNotFoundError(message)
            LOGGER.warning(message)
            return _record_result(source, dest, status="missing", message=message)
        LOGGER.info("Validated presence of %s", dest)
        if source.checksum:
            _enforce_checksum(dest, source.checksum)
        return _record_result(source, dest, status="validated")

    downloaded_path: Path | None = None
    cleanup_dir: Path | None = None

    if source_type in {"filesystem", "local"}:
        if not source.uri:
            raise ValueError(f"Filesystem source '{source.name}' requires a 'uri' path.")
        src = Path(source.uri).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"Source file {src} does not exist for '{source.name}'.")
        LOGGER.info("Copying %s -> %s", src, dest)
        _copy_filesystem(src, dest)
    elif source_type in {"http", "https", "url"}:
        if not source.uri:
            raise ValueError(f"HTTP source '{source.name}' requires a 'uri'.")
        LOGGER.info("Downloading %s -> %s", source.uri, dest)
        _download_http(source.uri, dest)
    elif source_type in {"kaggle", "kaggle_dataset", "kaggle_competition"}:
        downloaded_path = _download_kaggle_artifact(source)
        cleanup_dir = downloaded_path.parent
    elif source_type in {"s3", "aws_s3"}:
        downloaded_path = _download_s3_object(source)
    elif source_type in {"azure", "azure_blob", "blob"}:
        downloaded_path = _download_azure_blob(source)
    elif source_type in {"dvc", "dvc_remote"}:
        _pull_via_dvc(source)
    else:
        raise ValueError(f"Unsupported ingestion source type '{source.type}'.")

    try:
        if downloaded_path:
            _materialize_download(downloaded_path, dest, force_decompress=source.decompress)
    finally:
        if cleanup_dir:
            _cleanup_temp_dir(cleanup_dir)

    if not dest.exists():
        message = f"Ingestion for source '{source.name}' did not produce {dest}."
        if fail_on_missing:
            raise FileNotFoundError(message)
        LOGGER.warning(message)
        return _record_result(source, dest, status="missing", message=message)

    if source.checksum:
        _enforce_checksum(dest, source.checksum)

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
