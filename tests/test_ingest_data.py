from pathlib import Path

from creditrisk.config import IngestionConfig, IngestionSourceConfig
from creditrisk.pipelines.ingest_data import ingest_sources, write_ingestion_report
from .utils import build_test_config


def test_ingest_sources_validates_existing_file(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "application_train.csv"
    raw_file.write_text("SK_ID_CURR,TARGET\n1,0\n", encoding="utf-8")

    ingestion_cfg = IngestionConfig(
        enabled=True,
        fail_on_missing=True,
        sources=[
            IngestionSourceConfig(
                name="application_train",
                type="existing",
                output_path=raw_file,
            )
        ],
    )

    config = build_test_config(
        tmp_path,
        ["feature_1"],
        ingestion_config=ingestion_cfg,
    )

    results = ingest_sources(config)
    assert results[0]["status"] == "validated"

    report_path = Path(config.paths.ingestion_report)
    write_ingestion_report(results, report_path)
    assert report_path.exists()


def test_ingest_sources_copies_local_file(tmp_path):
    source_dir = tmp_path / "downloads"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_file = source_dir / "bureau.csv"
    source_file.write_text("SK_ID_CURR,CREDIT_ACTIVE\n1,Active\n", encoding="utf-8")

    dest_file = tmp_path / "data" / "raw" / "bureau.csv"

    ingestion_cfg = IngestionConfig(
        enabled=True,
        fail_on_missing=True,
        sources=[
            IngestionSourceConfig(
                name="bureau",
                type="filesystem",
                uri=str(source_file),
                output_path=dest_file,
            )
        ],
    )

    config = build_test_config(
        tmp_path,
        ["feature_1"],
        ingestion_config=ingestion_cfg,
    )

    results = ingest_sources(config)
    assert dest_file.exists()
    assert dest_file.read_text(encoding="utf-8") == source_file.read_text(encoding="utf-8")
    assert results[0]["status"] == "fetched"
