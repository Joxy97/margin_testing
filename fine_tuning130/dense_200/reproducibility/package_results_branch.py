"""Package the validated fine_tuning130 campaign into the results branch layout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve()
SOURCE = HERE.parents[1]
WORKSPACE = HERE.parents[4]
GLOBAL_PLOTS = SOURCE.parents[1] / "PLots"


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ValueError(f"Missing source directory: {source}")
    shutil.copytree(source, destination)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def package(repository: Path) -> dict:
    target = repository / "fine_tuning130"
    if target.exists():
        raise ValueError(f"Refusing to overwrite existing package: {target}")
    dense = target / "dense_200"
    dense.mkdir(parents=True)

    compact = SOURCE / "plot_ready"
    for folder in ("aggregated", "raw", "progress", "postprocessed", "solver_settings"):
        copy_tree(compact / folder, dense / folder)
    for name in ("manifest.json", "state.json", "completion.json", "projection_validation.json"):
        copy_file(compact / name, dense / name)
    copy_tree(SOURCE / "plots", dense / "plots")
    for name in ("README.txt", "download_manifest.json", "FINE_TUNING130_DENSE_200_SOLVER_RANKING.txt"):
        copy_file(SOURCE / name, dense / name)
    copy_tree(SOURCE / "reproducibility", dense / "reproducibility")

    archive_target = dense / "downloads" / "archives"
    archive_target.mkdir(parents=True)
    for archive in sorted((SOURCE / "downloads" / "archives").glob("*.tar.gz")):
        copy_file(archive, archive_target / archive.name)
    hosts_target = dense / "downloads" / "hosts"
    for host in sorted(path for path in (SOURCE / "downloads" / "hosts").iterdir() if path.is_dir()):
        destination = hosts_target / host.name
        for name in ("run.log", "campaign_plan.json", "run_fine_tuned_dense.py"):
            copy_file(host / name, destination / name)
        for name in ("identity.json", "progress.json", "completion.json", "run_end.json"):
            copy_file(host / "results" / name, destination / "results" / name)

    tuning_source = WORKSPACE / "fine_tuning_publish_20261024" / "fine_tuning"
    tuning_target = dense / "reproducibility" / "fine_tuning_setup"
    for name in ("selected_settings.json", "source_manifest.json", "protocol.json",
                 "provenance.json", "environment.json", "README.md"):
        copy_file(tuning_source / name, tuning_target / name)

    plots_target = target / "PLots"
    plots_target.mkdir(parents=True)
    global_plots = sorted(GLOBAL_PLOTS.glob("200_dense_*_fine_tuning130_part*.png"))
    if len(global_plots) != 9:
        raise ValueError(f"Expected nine global figures, found {len(global_plots)}")
    for plot in global_plots:
        copy_file(plot, plots_target / plot.name)

    (dense / "README.txt").write_text(
        "fine_tuning130 — dense-200 post-tuning production campaign\n\n"
        "This is the latest production rerun of 130 tuned solvers on gka5e, not the earlier "
        "parameter-search trials. Every solver used seeds 0-99, a 20-second cap, and checkpoints "
        "at 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, and 20 seconds. All 13,000 runs succeeded.\n\n"
        "raw/, aggregated/, progress/, postprocessed/, manifest.json, state.json, and completion.json "
        "form the compact plot-ready result set. solver_settings/ records the exact production "
        "parameters. plots/gka5e/ contains nine runtime figures and nine publication-style figures.\n\n"
        "downloads/archives/ contains all seven checksum-verified source archives, including every "
        "per-seed production JSON. downloads/hosts/ keeps each host's run log and identity/completion "
        "records without duplicating the 26,000 extracted trial/claim files. download_manifest.json "
        "binds the archives and host completion records to the 13,000 successful runs.\n",
        encoding="utf-8",
    )

    (target / "README.txt").write_text(
        "fine_tuning130 results\n\n"
        "dense_200/ contains the complete compact 130-solver, 13,000-seed production results, "
        "solver settings, ranking, campaign plots, logs, archives, and reproducibility records.\n"
        "PLots/ contains the nine shareable six-panel figures, grouped by final solution quality.\n"
        "The source campaign used gka5e (200-variable dense QUBO), 100 seeds per solver, "
        "a 20-second cap, and nine solution-quality checkpoints.\n",
        encoding="utf-8",
    )

    files = sorted(path for path in target.rglob("*") if path.is_file())
    checksums = {
        str(path.relative_to(target)).replace("\\", "/"): sha256(path)
        for path in files
    }
    write_json(target / "SHA256SUMS.json", checksums)
    manifest = {
        "schema_version": 1,
        "status": "ready_to_publish",
        "layout": "fine_tuning130/dense_200",
        "campaign": "fine_tuned_dense_20261024",
        "solvers": 130,
        "trials": 13000,
        "successful_trials": 13000,
        "failed_trials": 0,
        "production_partitions": 7,
        "global_figures": 9,
        "campaign_figures": 18,
        "archives": 7,
        "files_excluding_package_indexes": len(files),
        "bytes_excluding_package_indexes": sum(path.stat().st_size for path in files),
        "checksum_index": "SHA256SUMS.json",
    }
    write_json(target / "PACKAGE_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.repository.resolve()), indent=2))


if __name__ == "__main__":
    main()
