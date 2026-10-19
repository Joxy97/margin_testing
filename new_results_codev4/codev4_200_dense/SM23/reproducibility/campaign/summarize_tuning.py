"""Summarize retained tuning evidence without changing any frozen configuration.

Standard library only. Run after downloading sparse/tuning and dense/tuning:
    python summarize_tuning.py --root .
Writes tuning_summary.json and tuning_summary.csv in --output (default --root).
Validation rows entered final selection; these are descriptive tuning metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metrics(rows):
    usable = [row for row in rows if row["result"].get("status") == "complete"
              and row["result"].get("energies") and finite(row["result"]["energies"][-1])]
    energies = [row["result"]["energies"][-1] for row in usable]
    gaps = [100. * (row["result"]["energies"][-1] - row["job"]["reference"])
            / max(1., abs(row["job"]["reference"])) for row in usable]
    exact = sum(abs(row["result"]["energies"][-1] - row["job"]["reference"])
                <= row["job"].get("tolerance", 0.) for row in usable)
    near = sum(row["result"]["energies"][-1] <= row["job"]["reference"]
               + .01 * max(1., abs(row["job"]["reference"]))
               + row["job"].get("tolerance", 0.) for row in usable)
    reasons = Counter(row["result"].get("stop_reason", "unknown") for row in rows)
    total = len(rows)
    return dict(trials=total, successful=len(usable), failed=total-len(usable),
                seeds=sorted({row["job"]["seed"] for row in rows}),
                mean_energy=statistics.mean(energies) if energies else None,
                best_energy=min(energies) if energies else None,
                mean_gap_percent=statistics.mean(gaps) if gaps else None,
                exact_hits=exact, exact_probability=exact/total if total else None,
                within_one_percent_hits=near,
                within_one_percent_probability=near/total if total else None,
                natural_stop_ratio=reasons["natural_return"]/total if total else None,
                stop_reasons=dict(reasons),
                step_counts=sorted({row["job"]["parameters"]["max_steps"] for row in rows}),
                mean_solve_wall_s=statistics.mean(row["result"]["actual_solve_wall_s"]
                    for row in rows if finite(row["result"].get("actual_solve_wall_s")))
                    if any(finite(row["result"].get("actual_solve_wall_s")) for row in rows) else None)


def summarize_density(root, density, methods):
    directory = root / density / "tuning"
    if not directory.exists():
        return None
    report = read(directory / "report.json") if (directory / "report.json").exists() else None
    identity = read(directory / "identity.json") if (directory / "identity.json").exists() else {}
    timing = read(directory / "timing_steps.json") if (directory / "timing_steps.json").exists() else {}
    config_path = root / density / "selected_config.json"
    configuration = read(config_path) if config_path.exists() else {}
    review_path = root / density / "schedule_review.json"
    schedule_review = read(review_path) if review_path.exists() else {}
    supplemental = [read(path) for path in sorted((root / density / "schedule_review" / "trials").glob("*.json"))]
    trials, groups = [], defaultdict(list)
    for path in sorted((directory / "trials").glob("*.json")):
        row = read(path)
        job = row["job"]
        if job["solver"] not in methods:
            raise ValueError(f"Unexpected method outside the new 23: {path}")
        phase = job["key"].split("__", 1)[0]
        if phase not in ("pilot", "screen", "holdout"):
            raise ValueError(f"Unknown tuning phase: {path}")
        groups[phase, job["solver"], job["candidate"]].append(row)
        trials.append(row)
    failures = [dict(key=row["job"]["key"], status=row["result"].get("status"),
                     error=row["result"].get("error")) for row in trials
                if row["result"].get("status") != "complete"]
    rows, warnings = [], []
    for method in methods:
        selection = (report or {}).get("selections", {}).get(method, {})
        selected = selection.get("selected")
        baseline_rows = groups["holdout", method, "baseline"]
        selected_rows = groups["holdout", method, selected] if selected else []
        baseline, chosen = metrics(baseline_rows), metrics(selected_rows)
        pilot_rows = groups["pilot", method, "baseline"]
        pilot = pilot_rows[0]["result"] if pilot_rows else {}
        count = pilot.get("iterations", 0)
        duration = pilot.get("iteration_elapsed", 0.)
        fallback = bool(pilot_rows and (not count or not finite(duration) or duration <= 0.))
        final_steps = configuration.get("solvers", {}).get(method, {}).get("max_steps", timing.get(method))
        review = schedule_review.get("solvers", {}).get(method)
        schedule_metrics = None
        if review:
            old = [row for row in supplemental if row["job"]["solver"] == method
                   and row["job"]["candidate"] == "old_schedule"]
            new = [row for row in supplemental if row["job"]["solver"] == method
                   and row["job"]["candidate"] == "new_schedule"]
            schedule_metrics = dict(accepted=review["accepted"], old=metrics(old),
                                    new=metrics(new or old), old_max_steps=review["old_parameters"]["max_steps"],
                                    new_max_steps=review["new_parameters"]["max_steps"])
        notes = []
        if fallback:
            notes.append("Pilot observed no completed iteration; finite ceiling fell back to one step.")
        if final_steps == 1:
            notes.append("Selected production ceiling is only one step; this is a limited smoke-scale schedule.")
        if review and review["accepted"]:
            notes.append("Supplemental paired validation accepted a longer finite schedule; see separate seed2002 comparison.")
        if chosen["natural_stop_ratio"] == 1.:
            notes.append("Every selected validation solve returned naturally; subsequent checkpoints retained the incumbent.")
        if chosen["failed"]:
            notes.append("Selected validation includes missing or failed objective results.")
        if not selected:
            notes.append("Final selection report has not been downloaded or tuning is incomplete.")
        if notes:
            warnings.append(dict(solver=method, notes=notes))
        delta = (baseline["mean_gap_percent"] - chosen["mean_gap_percent"]
                 if baseline["mean_gap_percent"] is not None and chosen["mean_gap_percent"] is not None else None)
        rows.append(dict(solver=method, screened=selection.get("screened"), selected=selected,
                         reference=baseline_rows[0]["job"]["reference"] if baseline_rows else None,
                         reference_type=baseline_rows[0]["job"].get("reference_type") if baseline_rows else None,
                         calibrated_max_steps=timing.get(method), pilot_iterations=count,
                         final_max_steps=final_steps, schedule_review=schedule_metrics,
                         pilot_capture_elapsed_s=duration, pilot_natural_elapsed_s=pilot.get("natural_elapsed"),
                         pilot_used_fallback=fallback, baseline_validation=baseline,
                         selected_validation=chosen, improvement_gap_percentage_points=delta,
                         screening={candidate: metrics(groups["screen", method, candidate])
                                    for candidate in ("baseline", "explore", "refine")}, notes=notes))
    statuses = Counter(row["result"].get("status", "unknown") for row in trials)
    phases = Counter(row["job"]["key"].split("__", 1)[0] for row in trials)
    return dict(density=density, complete=bool(report and report.get("complete")),
                identity=identity, phase_counts=dict(phases), status_counts=dict(statuses),
                total_trials=len(trials), total_errors=len(failures), failures=failures,
                supplemental_trials=len(supplemental),
                supplemental_errors=sum(row["result"].get("status") != "complete" for row in supplemental),
                pilot_fallback_count=sum(row["pilot_used_fallback"] for row in rows),
                one_step_count=sum(row["final_max_steps"] == 1 for row in rows),
                selected_baseline_count=sum(row["selected"] == "baseline" for row in rows),
                warnings=warnings, solvers=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    candidates = read(args.root / "tuning_candidates.json")
    methods = tuple(candidates["solvers"])
    if len(methods) != 23:
        raise ValueError("Expected exactly 23 new solver IDs in the frozen candidate file")
    densities = [value for name in ("sparse", "dense")
                 if (value := summarize_density(args.root, name, methods)) is not None]
    summary = dict(schema_version=1, note="Descriptive tuning metrics. Validation was used in final selection; "
                   "the few validation seeds are not independent performance evidence. Sparse/dense use different GPU models.",
                   gap_definition="100*(energy-reference)/max(1,abs(reference)); lower is better",
                   exact_definition="abs(energy-reference)<=recorded objective tolerance",
                   within_one_percent_definition="energy<=reference+0.01*max(1,abs(reference))+tolerance",
                   probability_denominator="All requested available validation trial rows, including failures as non-hits",
                   total_trials=sum(value["total_trials"]+value["supplemental_trials"] for value in densities),
                   total_errors=sum(value["total_errors"]+value["supplemental_errors"] for value in densities), densities=densities)
    output = args.output or args.root
    output.mkdir(parents=True, exist_ok=True)
    (output / "tuning_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    table = []
    for density in densities:
        for row in density["solvers"]:
            value = dict(density=density["density"], solver=row["solver"], selected=row["selected"],
                         screened=row["screened"], calibrated_max_steps=row["calibrated_max_steps"],
                         final_max_steps=row["final_max_steps"],
                         pilot_iterations=row["pilot_iterations"], pilot_used_fallback=row["pilot_used_fallback"],
                         improvement_gap_percentage_points=row["improvement_gap_percentage_points"])
            for label in ("baseline", "selected"):
                data = row[label+"_validation"]
                for name in ("trials", "successful", "failed", "mean_energy", "mean_gap_percent",
                             "exact_hits", "exact_probability", "within_one_percent_hits",
                             "within_one_percent_probability", "natural_stop_ratio", "mean_solve_wall_s"):
                    value[label+"_"+name] = data[name]
            value["notes"] = " ".join(row["notes"])
            review = row["schedule_review"]
            value["schedule_review_accepted"] = review["accepted"] if review else None
            value["schedule_review_old_gap_percent"] = review["old"]["mean_gap_percent"] if review else None
            value["schedule_review_new_gap_percent"] = review["new"]["mean_gap_percent"] if review else None
            table.append(value)
    with (output / "tuning_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]) if table else ["density", "solver"])
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(dict(total_trials=summary["total_trials"], total_errors=summary["total_errors"],
         densities=[{key: value[key] for key in ("density", "complete", "total_trials", "total_errors",
                   "pilot_fallback_count", "one_step_count", "selected_baseline_count")} for value in densities],
         output=str(output.resolve())), indent=2))


if __name__ == "__main__":
    main()
