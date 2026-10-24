"""Recover only the checksum-verified gka5e input; never run a solver."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from qubo_benchmark.formats import parseMatrix
from tools.stage_continuous_inputs import canonical_fingerprint


def verified(path, problem):
    if canonical_fingerprint(path) != problem["numeric_coo_sha256"]:
        raise ValueError("Canonical input differs from the tested gka5e problem")
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "canonical_arrays_identical": True,
            "numeric_coo_sha256": problem["numeric_coo_sha256"],
            "current_npz_sha256": current,
            "tested_npz_sha256": problem["npz_sha256"],
            "npz_bytes_identical": current == problem["npz_sha256"]}


def stage(output, problem, fetch=None):
    output = Path(output)
    if output.exists():
        return dict(verified(output, problem), status="existing_verified")
    if fetch is None:
        if not problem["raw_source_url"].startswith("https://"):
            raise ValueError("The pinned public input must use HTTPS")
        with urlopen(problem["raw_source_url"], timeout=60) as response:
            payload = response.read(problem["raw_source_bytes"] + 1)
    else:
        payload = fetch(problem["raw_source_url"])
    if (len(payload) != problem["raw_source_bytes"] or
            hashlib.sha256(payload).hexdigest() != problem["raw_source_sha256"]):
        raise ValueError("Public source differs from the pinned bytes/checksum")
    canonical = parseMatrix(payload).canonical()
    if canonical.n != problem["n_variables"]:
        raise ValueError("Public source has the wrong variable count")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="gka5e-stage-", suffix=".npz", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        canonical.save(temporary)
        verified(temporary, problem)
        # Atomic no-clobber publication; a concurrent writer is accepted only
        # when its completed canonical arrays independently verify.
        try:
            os.link(temporary, output)
        except FileExistsError:
            return dict(verified(output, problem), status="existing_verified")
        return dict(verified(output, problem), status="staged_verified")
    finally:
        temporary.unlink()


def main(argv=None):
    protocol = json.loads((Path(__file__).with_name("protocol.json")).read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / protocol["problem"]["repository_npz"])
    args = parser.parse_args(argv)
    print(json.dumps(stage(args.output, protocol["problem"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
