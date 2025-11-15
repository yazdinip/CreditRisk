"""Generate a deployment manifest summarizing artifacts and deployment hooks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from creditrisk.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize deployable artifacts for serving.")
    parser.add_argument(
        "--config",
        default="configs/creditrisk_pd.yaml",
        help="Path to the experiment configuration YAML.",
    )
    parser.add_argument(
        "--output",
        default="reports/deploy_manifest.json",
        help="Where to write the manifest JSON.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_manifest(config: Config, *, manifest_path: Path, config_path: Path) -> Dict[str, Any]:
    registry_transition = _read_json(config.paths.reports_dir / "registry_transition.json")
    promotion_candidate = _read_json(config.paths.registry_report)

    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "model_path": str(config.paths.model_path),
        "reports_dir": str(config.paths.reports_dir),
        "serving": {
            "api_entrypoint": "uvicorn creditrisk.serve.api:get_app --host 0.0.0.0 --port 8080",
            "batch_entrypoint": "python -m creditrisk.pipelines.batch_predict --help",
        },
        "docker": {
            "api_dockerfile": "Dockerfile.api",
            "batch_dockerfile": "Dockerfile.batch",
            "ghcr_repository": "ghcr.io/<owner>/creditrisk-{api|batch}",
        },
        "ci_cd": {
            "workflow": ".github/workflows/cd.yaml",
            "ecs_secrets": [
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_REGION",
                "ECS_CLUSTER_NAME",
                "ECS_SERVICE_NAME",
                "PREDICT_ENDPOINT_URL",
            ],
        },
        "registry": registry_transition or promotion_candidate,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    build_manifest(config, manifest_path=Path(args.output), config_path=Path(args.config))


if __name__ == "__main__":
    main()
