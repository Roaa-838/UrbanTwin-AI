from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from urbantwin.common import (
    CANONICAL_CONFIG_HASH_SCHEME,
    ConfigurationHashError,
    canonical_config_sha256,
    configuration_hash_record,
    load_yaml,
    sha256_file,
    validate_configuration_hash,
)

_CONFIG_TEXT = """\
schema_version: test/v1
benchmark:
  seed: 42
  alpha: 0.25
  enabled: true
sites:
  - cairo-a
  - cairo-b
"""


def _write_bytes(path: Path, text: str, newline: bytes) -> None:
    path.write_bytes(text.replace("\n", newline.decode("ascii")).encode("utf-8"))


def test_canonical_hash_is_identical_for_lf_and_crlf(tmp_path: Path) -> None:
    lf_path = tmp_path / "lf.yaml"
    crlf_path = tmp_path / "crlf.yaml"
    _write_bytes(lf_path, _CONFIG_TEXT, b"\n")
    _write_bytes(crlf_path, _CONFIG_TEXT, b"\r\n")

    lf_config = load_yaml(lf_path)
    crlf_config = load_yaml(crlf_path)

    assert lf_config == crlf_config
    assert sha256_file(lf_path) != sha256_file(crlf_path)
    assert canonical_config_sha256(lf_config) == canonical_config_sha256(crlf_config)


def test_canonical_hash_changes_with_scientific_values() -> None:
    original = {"benchmark": {"seed": 42, "alpha": 0.25}, "sites": ["cairo-a"]}
    modified = {"benchmark": {"seed": 43, "alpha": 0.25}, "sites": ["cairo-a"]}

    assert canonical_config_sha256(original) == (
        "465df2e9018c299f435e443cd675c87275904855f4d4ebad8535f4a5f47a628d"
    )
    assert canonical_config_sha256(original) != canonical_config_sha256(modified)


def test_legacy_hash_contract_remains_strictly_byte_bound(tmp_path: Path) -> None:
    config_path = tmp_path / "observed.yaml"
    _write_bytes(config_path, _CONFIG_TEXT, b"\n")
    legacy_record = {"config_sha256": sha256_file(config_path)}

    verified = validate_configuration_hash(config_path, legacy_record, mode="legacy_raw")
    assert verified == {"mode": "legacy_raw", "raw_checkout_matches": True}

    _write_bytes(config_path, _CONFIG_TEXT, b"\r\n")
    with pytest.raises(ConfigurationHashError, match="raw SHA-256 mismatch"):
        validate_configuration_hash(config_path, legacy_record, mode="legacy_raw")

    backfilled = {
        **legacy_record,
        "config_hash_scheme": CANONICAL_CONFIG_HASH_SCHEME,
        "config_canonical_sha256": canonical_config_sha256(load_yaml(config_path)),
    }
    with pytest.raises(ConfigurationHashError, match="must not contain canonical"):
        validate_configuration_hash(config_path, backfilled, mode="legacy_raw")


def test_future_hash_contract_accepts_only_semantically_equivalent_checkout(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "observed.yaml"
    _write_bytes(config_path, _CONFIG_TEXT, b"\n")
    configuration = load_yaml(config_path)
    record = configuration_hash_record(config_path, configuration)
    with pytest.raises(ConfigurationHashError, match="snapshot does not match"):
        configuration_hash_record(
            config_path,
            {**configuration, "schema_version": "different/v1"},
        )

    _write_bytes(config_path, _CONFIG_TEXT, b"\r\n")
    verified = validate_configuration_hash(
        config_path,
        record,
        mode="canonical_v1",
        embedded_configuration=configuration,
    )
    assert verified == {"mode": "canonical_v1", "raw_checkout_matches": False}

    changed = _CONFIG_TEXT.replace("seed: 42", "seed: 43")
    _write_bytes(config_path, changed, b"\n")
    with pytest.raises(ConfigurationHashError, match="Canonical configuration SHA-256 mismatch"):
        validate_configuration_hash(
            config_path,
            record,
            mode="canonical_v1",
            embedded_configuration=configuration,
        )


def test_future_hash_contract_rejects_unknown_scheme_and_snapshot_mismatch(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "observed.yaml"
    _write_bytes(config_path, _CONFIG_TEXT, b"\n")
    configuration = load_yaml(config_path)
    record = configuration_hash_record(config_path, configuration)

    unknown = {**record, "config_hash_scheme": "unknown/v9"}
    with pytest.raises(ConfigurationHashError, match="Unknown or missing"):
        validate_configuration_hash(
            config_path,
            unknown,
            mode="canonical_v1",
            embedded_configuration=configuration,
        )

    mismatched_snapshot = {**configuration, "benchmark": {"seed": 99}}
    with pytest.raises(ConfigurationHashError, match="snapshot hash mismatch"):
        validate_configuration_hash(
            config_path,
            record,
            mode="canonical_v1",
            embedded_configuration=mismatched_snapshot,
        )


@pytest.mark.parametrize(
    "configuration",
    [
        {"when": date(2024, 10, 31)},
        {"value": float("nan")},
        {"value": float("inf")},
        {1: "non-string-key"},
        {"value": {"not", "json"}},
    ],
)
def test_canonical_hash_rejects_type_ambiguous_values(configuration: dict) -> None:
    with pytest.raises(ConfigurationHashError):
        canonical_config_sha256(configuration)


def test_strict_config_loading_rejects_duplicate_and_corrupted_yaml(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_bytes(b"seed: 1\nseed: 2\n")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        load_yaml(duplicate)

    corrupted = tmp_path / "corrupted.yaml"
    corrupted.write_bytes(b"benchmark: [unterminated\n")
    record = {
        "config_sha256": "0" * 64,
        "config_hash_scheme": CANONICAL_CONFIG_HASH_SCHEME,
        "config_canonical_sha256": "0" * 64,
    }
    with pytest.raises(ConfigurationHashError, match="cannot be parsed"):
        validate_configuration_hash(
            corrupted,
            record,
            mode="canonical_v1",
            embedded_configuration={"benchmark": {}},
        )

    with pytest.raises(ConfigurationHashError, match="file is missing"):
        validate_configuration_hash(
            tmp_path / "missing.yaml",
            record,
            mode="canonical_v1",
            embedded_configuration={"benchmark": {}},
        )


def test_clean_checkouts_pin_only_observed_config_to_lf() -> None:
    attributes = (Path(__file__).parents[1] / ".gitattributes").read_text(encoding="utf-8")
    rules = {
        line.strip()
        for line in attributes.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert rules == {"configs/observed_traffic.yaml text eol=lf"}
    assert all("experiments.yaml" not in rule for rule in rules)
