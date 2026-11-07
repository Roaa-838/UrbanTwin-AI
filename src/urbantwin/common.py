"""Shared configuration and provenance utilities."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        result = yaml.safe_load(f)
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return result


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata(kind: str, config: dict, **extra: Any) -> dict:
    return {
        "data_kind": kind,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        **extra,
    }


def require_real_provenance(meta: dict) -> None:
    if meta.get("data_kind") != "real":
        raise ValueError("Research analysis requires real provenance. Synthetic test data rejected.")
