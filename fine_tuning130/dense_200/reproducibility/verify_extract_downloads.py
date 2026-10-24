"""Verify and extract completed fine-tuned production campaign partitions."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "downloads"
EXPECTED = {
    "old_1": ("old_1_b0324ff6311a.tar.gz", "df9719e1d55b9b32fd2f96cb787c139de51eb803d86c9d7daba104ec2860a051", 20),
    "old_2": ("old_2_844f3d4ff6f4.tar.gz", "3fba20ab122f1e18065614e3f90f7c7ae04b8b8ad118a77bd98790081aa44347", 20),
    "old_3": ("old_3_3cf9acea735e.tar.gz", "f95e0ea0a2922ede6d1622a54ee5e85e22e9c763b72c1e462e20d84335f7e015", 20),
    "new_1": ("new_1_7fccafc5cb98.tar.gz", "e8f0f91c998efa237d3697f08f4fe09763158b91c0626b0eb842e25810d9025f", 10),
    "new_3": ("new_3_42b1b79e2f65.tar.gz", "e48f18f48f6e6b4a5df420c06ef35d00f4adb4f040eb0c95c69943f86ab63e44", 20),
    "new_4": ("new_4_dfcd6814bbf0.tar.gz", "c64d5712cada68230ccb3b9e04b9321c3fce8f88de0ca5304f43a4e6d5885e97", 20),
    "old_4": ("old_4_cb5709963e6a.tar.gz", "126e44fd9a0a9ab2740dabd1d9ae42cb9a4cfa68af64f1f5473073b358939d6e", 20),
}
RUNNER_SHA = "bd47562f2766082fc35ba5f4269c9f4703fbacf0fe0c9b68dedbbff1d88b8d16"
PLAN_SHA = "ea5cec7fbe41ff4b3dae9c0c90b5435423d0c6f49969966dc616e2f80870d66f"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_members(archive: tarfile.TarFile):
    members = archive.getmembers()
    seen = set()
    for member in members:
        path = PurePosixPath(member.name)
        assert not path.is_absolute() and ".." not in path.parts and member.name not in seen
        assert path.parts and path.parts[0] in {"results", "run.log", "run_fine_tuned_dense.py", "campaign_plan.json"}
        assert member.isfile() or member.isdir()
        seen.add(member.name)
    return members


def main():
    rows = []
    all_solvers = set()
    for label, (name, expected_sha, solver_count) in EXPECTED.items():
        archive_path = DOWNLOADS / "archives" / name
        assert archive_path.is_file() and sha(archive_path) == expected_sha
        destination = DOWNLOADS / "hosts" / label
        if not destination.exists():
            destination.mkdir(parents=True)
            with tarfile.open(archive_path, "r:gz") as archive:
                members = safe_members(archive)
                archive.extractall(destination, members=members, filter="data")
        assert sha(destination / "run_fine_tuned_dense.py") == RUNNER_SHA
        assert sha(destination / "campaign_plan.json") == PLAN_SHA
        completion = read(destination / "results" / "completion.json")
        identity = read(destination / "results" / "identity.json")
        assert completion["status"] == "complete" and completion["failed"] == 0
        assert completion["attempted"] == completion["successful"] == solver_count * 100
        assert completion["completed_solvers"] == solver_count
        assert identity["host"] == label and len(identity["solvers"]) == solver_count
        trial_files = sorted((destination / "results" / "trials").glob("*/*.json"))
        assert len(trial_files) == solver_count * 100
        per_solver = {}
        for path in trial_files:
            row = read(path)
            assert row["status"] == "complete" and row["identity"] == identity["digest"]
            assert row["solver"] == path.parent.name and row["seed"] == int(path.stem)
            assert len(row["energies"]) == 9
            per_solver.setdefault(row["solver"], set()).add(row["seed"])
        assert set(per_solver) == set(identity["solvers"])
        assert all(seeds == set(range(100)) for seeds in per_solver.values())
        assert all_solvers.isdisjoint(per_solver)
        all_solvers.update(per_solver)
        rows.append({
            "host_label": label,
            "archive": f"downloads/archives/{name}",
            "archive_sha256": expected_sha,
            "completion_sha256": sha(destination / "results" / "completion.json"),
            "identity": identity["digest"],
            "solver_count": solver_count,
            "trial_count": len(trial_files),
            "successful": completion["successful"],
            "failed": completion["failed"],
            "status": "PASS",
        })
    assert len(all_solvers) == 130
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "fine_tuned_dense_20261024",
        "scope": "Latest post-tuning 100-seed production run; parameter-search trials are excluded.",
        "partial_download": False,
        "completed_hosts_downloaded": 7,
        "pending_hosts": [],
        "solver_count_downloaded": 130,
        "trial_count_downloaded": 13000,
        "successful_trials": 13000,
        "failed_trials": 0,
        "runner_sha256": RUNNER_SHA,
        "campaign_plan_sha256": PLAN_SHA,
        "hosts": rows,
    }
    (ROOT / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
