#!/usr/bin/env python3
"""Generate stratified BiqMac data for a Torch SVL quality model."""

from __future__ import annotations

import argparse, csv, json, math, os, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from margin_calculator.optimization.optimization_solver.bqm_solver import TorchSVLBQMSolver
from run_biqmac_benchmark import _cut, _graph, _qubo

PARAMETERS = ("runs","steps","dt","mass","damping","temperature","integrator",
              "transverse_field_initial","transverse_field_final",
              "problem_scale_initial","problem_scale_final")
FIELDS = ("timestamp_utc","instance","split","stratum","vertices","edges","density",
          "weight_mean","weight_std","negative_weight_fraction","configuration_id",*PARAMETERS,
          "solver_seed","solve_seconds","cut","reference_cut","quality")


def _settings(count: int, seed: int) -> list[dict[str, object]]:
    sample = qmc.LatinHypercube(10, seed=seed).random(count)
    result = []
    for index, row in enumerate(sample):
        runs = [2, 4, 6, 8, 10, 12][min(int(row[0] * 6), 5)]
        steps = [1000, 2000, 3000, 4000, 5000, 6000][min(int(row[1] * 6), 5)]
        result.append({
            "runs": runs, "steps": steps, "dt": .005 + .025 * row[2],
            "mass": .4 + 1.2 * row[3], "damping": .05 + .30 * row[4],
            "temperature": .04 * row[5], "integrator": "weak_order_2" if index % 2 else "euler_maruyama",
            "transverse_field_initial": .3 + 1.2 * row[6], "transverse_field_final": .15 * row[7],
            "problem_scale_initial": .2 * row[8], "problem_scale_final": 1.0 + 2.0 * row[9],
        })
    return result


def run(config_path: Path, resume: bool) -> None:
    config = json.loads(config_path.read_text())
    data = (config_path.parent / config["data_dir"]).resolve()
    output = (config_path.parent / config["output_dir"]).resolve(); output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw_results.csv"
    completed = set()
    if raw.exists():
        if not resume: raise FileExistsError(f"{raw} exists; pass --resume")
        with raw.open(newline="") as source:
            completed = {(r["instance"], int(r["configuration_id"])) for r in csv.DictReader(source)}
    by_stratum: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in config["instances"]: by_stratum[str(item["stratum"])].append(item)
    split = {}
    for stratum, items in sorted(by_stratum.items()):
        names = sorted(Path(str(i["path"])).name for i in items)
        # Every stratum has nine members: six train, three evaluation.
        for index, name in enumerate(names): split[name] = "evaluation" if index % 3 == 2 else "train"
    settings = _settings(int(config["configuration_count"]), int(config["seed"]))
    manifest = dict(config, design="latin_hypercube_all_parameters", split=split,
                    expected_records=len(config["instances"])*len(settings), settings=settings,
                    started_utc=datetime.now(timezone.utc).isoformat())
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    solver = TorchSVLBQMSolver("cpu")
    first = _graph(data / config["instances"][0]["path"])
    solver.solve(_qubo(first), {**settings[0], "seed":int(config["seed"]), "dtype":"float32", "run_batch_size":2})
    for item in config["instances"]:
        graph = _graph(data / str(item["path"])); problem = _qubo(graph)
        weights = np.asarray(graph.weights, dtype=float); density = 2*len(graph.heads)/(graph.vertices*(graph.vertices-1))
        for cid, dynamics in enumerate(settings):
            if (graph.name, cid) in completed: continue
            params = {**dynamics, "seed":int(config["seed"])+cid*104729,
                      "dtype":"float32", "run_batch_size":int(dynamics["runs"])}
            started=time.perf_counter(); result=solver.solve(problem, params); elapsed=time.perf_counter()-started
            cut=_cut(graph,result.sample)
            if not math.isclose(float(result.energy),-cut,abs_tol=1e-4): raise RuntimeError("inconsistent energy")
            row={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"instance":graph.name,
                 "split":split[graph.name],"stratum":item["stratum"],"vertices":graph.vertices,
                 "edges":len(graph.heads),"density":density,"weight_mean":weights.mean(),
                 "weight_std":weights.std(),"negative_weight_fraction":np.mean(weights<0),
                 "configuration_id":cid,**dynamics,"solver_seed":params["seed"],
                 "solve_seconds":elapsed,"cut":cut,"reference_cut":item["reference"],
                 "quality":cut/float(item["reference"])}
            header=not raw.exists() or raw.stat().st_size==0
            with raw.open("a",newline="") as destination:
                writer=csv.DictWriter(destination,fieldnames=FIELDS)
                if header: writer.writeheader()
                writer.writerow(row); destination.flush(); os.fsync(destination.fileno())
            completed.add((graph.name,cid)); print(f"{graph.name} {cid+1}/{len(settings)} q={row['quality']:.4f} {elapsed:.3f}s",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(); run(args.config.resolve(),args.resume)
