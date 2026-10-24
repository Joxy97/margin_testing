"""Compact preset provenance and bounded input recovery, without GPU work."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "fine_tuning"


def read(name):
    return json.loads((BUNDLE / name).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def test_selected_presets_and_per_solver_files_have_complete_consistent_provenance():
    selected, provenance = read("selected_settings.json"), read("provenance.json")
    assert len(selected["solvers"]) == 144
    assert len(selected["excluded_unchanged"]) == 14
    assert set(selected["solvers"]) == set(selected["schedule_by_solver"])
    assert len(list((BUNDLE / "solvers").glob("*.json"))) == 144
    for relative, expected in provenance["files"].items():
        assert hashlib.sha256((BUNDLE / relative).read_bytes()).hexdigest() == expected
    execution = {key: selected[key] for key in provenance["execution_payload_keys"]}
    assert digest(execution) == provenance["execution_settings_sha256"]
    for sid, parameters in selected["solvers"].items():
        item = read("solvers/" + sid + ".json")
        assert item["solver"] == sid and item["parameters"] == parameters
        assert item["parameters_sha256"] == digest(parameters)
        assert item["schedule"] == selected["schedule_by_solver"][sid]
        if sid in selected["excluded_unchanged"]:
            assert item["status"] == "excluded_unchanged"
            assert not item["selection"]["tuned"] and item["observations"] is None
            assert item["archive_parameter_changes"] == {}
        else:
            assert item["selection"]["tuned"] and item["observations"]["seeds"] == 24
            assert item["observations"]["selection_affected"]
            assert not item["observations"]["independent_final_confirmation"]


def test_frozen_source_manifest_and_all_seed_stages_are_bound():
    manifest, protocol = read("source_manifest.json"), read("protocol.json")
    assert len(manifest["source_files"]) == 100
    assert digest(manifest["source_files"]) == manifest["source_sha256"]
    for relative, expected in manifest["source_files"].items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT) and path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    assert manifest["selected_settings_sha256"] == hashlib.sha256((BUNDLE / "selected_settings.json").read_bytes()).hexdigest()
    seen = set(protocol["performance_profile_seeds_excluded"])
    for stages in protocol["seed_stages"].values():
        for stage, seeds in stages.items():
            assert len(seeds) == len(set(seeds)) and not seen.intersection(seeds)
            seen.update(seeds)
            if stage == "holdout":
                assert len(seeds) == 24
    assert protocol["budget_seconds"] == 20 and protocol["execution"]["tf32"] is False
    assert not protocol["deadline"]["hard_physical_wall_time_cap"]


def test_published_json_has_no_machine_paths_or_addresses():
    pattern = re.compile(r"(?:\b[A-Za-z]:[\\/]|/workspace/|/home/|file://|root@|GPU-[0-9a-f]{8})")
    addresses = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    def inspect(value, key=None):
        if key == "package_versions":
            return  # Four-part package versions are not network addresses.
        if isinstance(value, dict):
            for child, item in value.items():
                inspect(item, child)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
        elif isinstance(value, str):
            assert not addresses.search(value)
    for path in BUNDLE.rglob("*.json"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name
        inspect(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture
def staging(tmp_path):
    spec = importlib.util.spec_from_file_location("preset_stage_input", BUNDLE / "stage_input.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = b"2 3\n1 1 -1\n1 2 2\n2 2 -3\n"
    original = tmp_path / "original.npz"
    module.parseMatrix(payload).canonical().save(original)
    problem = dict(n_variables=2, raw_source_url="https://example.invalid/pinned",
                   raw_source_bytes=len(payload), raw_source_sha256=hashlib.sha256(payload).hexdigest(),
                   npz_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                   numeric_coo_sha256=module.canonical_fingerprint(original))
    return module, payload, problem


def test_single_input_recovery_is_checksum_checked_and_idempotent(tmp_path, staging):
    module, payload, problem = staging
    target = tmp_path / "recovered.npz"
    report = module.stage(target, problem, fetch=lambda url: payload)
    assert report["canonical_arrays_identical"] and report["status"] == "staged_verified"
    before = target.read_bytes()
    report = module.stage(target, problem, fetch=lambda url: pytest.fail("unexpected download"))
    assert report["status"] == "existing_verified" and target.read_bytes() == before
    assert not list(tmp_path.glob("gka5e-stage-*"))


def test_recovery_rejects_bad_payload_or_existing_different_input(tmp_path, staging):
    module, payload, problem = staging
    target = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match="pinned"):
        module.stage(target, problem, fetch=lambda url: payload + b"bad")
    assert not target.exists()
    module.parseMatrix(b"1 1\n1 1 -1\n").canonical().save(target)
    before = target.read_bytes()
    with pytest.raises(ValueError, match="Canonical"):
        module.stage(target, problem, fetch=lambda url: pytest.fail("unexpected download"))
    assert target.read_bytes() == before


def test_container_byte_differences_preserve_canonical_identity(tmp_path, staging):
    module, payload, problem = staging
    value = module.parseMatrix(payload).canonical()
    target = tmp_path / "uncompressed.npz"
    np.savez(target, n=value.n, rows=value.rows, cols=value.cols, values=value.values, offset=value.offset)
    report = module.verified(target, problem)
    assert report["canonical_arrays_identical"] and not report["npz_bytes_identical"]
