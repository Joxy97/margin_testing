"""Supplement zero-observation timing pilots without changing numerical source.

Only run after initial tuning completes. A three-second pilot on the selected
physical parameters proposes a longer finite schedule. Compare old/new schedules
on one fresh twenty-second paired validation seed. Preserve both trials and the
initial configuration. This is further tuning, not independent assessment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys

import campaign


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(root / "repo"))
    parser.add_argument("--catalog", default=str(root / "catalog.json"))
    parser.add_argument("--density", choices=("sparse", "dense"), required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--pilot-seconds", type=float, default=3.)
    args = parser.parse_args(argv)
    campaign.bootstrap(args.repo)
    from qubo_benchmark.continuous_catalog import load_continuous_catalog, select_continuous_problem
    from qubo_benchmark.runtime.common import atomic, read, filehash, source_snapshot_hash, digest
    from qubo_benchmark.runtime.compact_metrics import CHECKPOINTS
    density_root = root / args.density
    original_report = read(density_root / "tuning" / "report.json")
    if not original_report.get("complete"):
        raise ValueError("Initial tuning must finish before schedule review")
    current_config_path = density_root / "selected_config.json"
    config = read(current_config_path)
    directory = density_root / "schedule_review"
    original_path = directory / "original_selected_config.json"
    if not original_path.exists():
        atomic(original_path, config)
    original = read(original_path)
    if config != original and not (density_root / "schedule_review.json").exists():
        raise ValueError("Selected config changed without a completed schedule-review record")
    entry, npz = select_continuous_problem(200, args.density, load_continuous_catalog(args.catalog))
    reference = entry["normalized_reference_objective_min"]
    affected = []
    for solver in original_report["selections"]:
        path = density_root / "tuning" / "trials" / f"pilot__{solver}__baseline__9000.json"
        pilot = read(path)["result"]
        if not pilot.get("iterations") or not pilot.get("iteration_elapsed"):
            affected.append(solver)
    identity = dict(source_snapshot=source_snapshot_hash(), input_sha256=filehash(npz),
                    original_configuration_sha256=digest(original), helper_sha256=filehash(__file__),
                    original_report_sha256=filehash(density_root / "tuning" / "report.json"),
                    devices=args.devices.split(","), density=args.density,
                    pilot_seed=9001, pilot_seconds=args.pilot_seconds,
                    validation_seed=2002, validation_seconds=20., affected=affected,
                    schedule="finite steps", selection="paired final energy then checkpoint mean; strict improvement only")

    def job(solver, candidate, parameters, phase, seed, budget, pilot=False):
        return dict(key=f"{phase}__{solver}__{candidate}__{seed}", candidate=candidate,
                    pilot=pilot, solver=solver, parameters=parameters, seed=seed,
                    cpu_threads=1, npz=str(Path(npz).resolve()), input_sha256=identity["input_sha256"],
                    budget=budget, checkpoints=list(CHECKPOINTS) if budget == 20. else [budget],
                    wall_schedule=False, reference=reference,
                    reference_type="OPTIMUM" if entry["reference_status"] in
                        ("published_proven_optimum", "independently_verified_optimum") else "BKS",
                    tolerance=0. if entry.get("integer_objective", True) else 1e-9,
                    stop_on_optimum=not pilot)

    report = dict(schema_version=1, identity=identity, original_config_sha256=digest(original),
                  production_started=False, solvers={}, limitation="One paired fresh validation seed enters schedule selection; not an independent performance estimate.")
    if not affected:
        report.update(selected_config_sha256=digest(config), complete=True)
        atomic(density_root / "schedule_review.json", report)
        print("No pilot fallbacks required schedule review", flush=True)
        return
    pool = campaign.TuningPool(args.repo, args.devices.split(","), directory, identity)
    try:
        pilots = [job(solver, "selected", dict(original["solvers"][solver], max_steps=16),
                      "pilot", 9001, args.pilot_seconds, True) for solver in affected]
        pilot_rows = pool.run(pilots, "schedule_pilot")
        proposals, validation = {}, []
        for task in pilots:
            solver = task["solver"]
            row = pilot_rows[task["key"]]
            old = deepcopy(original["solvers"][solver])
            count, duration = row.get("iterations", 0), row.get("iteration_elapsed", 0.)
            steps = max(1, min(20000, math.floor(18. * count / duration))) if count and duration > 0 else old["max_steps"]
            new = dict(old, max_steps=steps)
            proposals[solver] = dict(old_parameters=old, new_parameters=new,
                pilot_trial=f"schedule_review/trials/{task['key']}.json",
                pilot_iterations=count, pilot_capture_elapsed_s=duration,
                accepted=False, note="Longer timing pilot still observed no completed iteration" if not count else None)
            for label, params in (("old_schedule", old), ("new_schedule", new)):
                if label == "new_schedule" and new == old:
                    continue
                validation.append(job(solver, label, params, "holdout", 2002, 20.))
        values = pool.run(validation, "schedule_validation")
        config = deepcopy(original)
        for solver, proposal in proposals.items():
            old_key = f"holdout__{solver}__old_schedule__2002"
            new_key = f"holdout__{solver}__new_schedule__2002"
            old_result = values[old_key]
            new_result = values.get(new_key, old_result)
            old_score, new_score = campaign.quality([old_result], reference), campaign.quality([new_result], reference)
            accepted = proposal["new_parameters"] != proposal["old_parameters"] and new_score < old_score
            proposal.update(accepted=accepted,
                baseline_trial=f"schedule_review/trials/{old_key}.json",
                supporting_trial=f"schedule_review/trials/{new_key if new_key in values else old_key}.json",
                result_status=new_result.get("status"),
                result_energy=new_result.get("energies", [None])[-1],
                old_score=[value if math.isfinite(value) else None for value in old_score],
                new_score=[value if math.isfinite(value) else None for value in new_score])
            if accepted:
                config["solvers"][solver] = proposal["new_parameters"]
            report["solvers"][solver] = proposal
        report.update(selected_config_sha256=digest(config), complete=True,
                      total_trials=len(pilots)+len(validation),
                      errors=sum(row.get("status") != "complete" for row in list(pilot_rows.values())+list(values.values())),
                      accepted_count=sum(row["accepted"] for row in proposals.values()))
        # Publish the complete evidence record before updating the selected file.
        atomic(density_root / "schedule_review.json", report)
        atomic(current_config_path, config)
        print(f"Schedule review complete: {report['accepted_count']}/{len(affected)} changed; errors={report['errors']}; production not started", flush=True)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
