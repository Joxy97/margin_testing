#!/usr/bin/env python3
"""Prepare native/exactly integer-encoded QUBOs and count memory-admissible cases."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
from fractions import Fraction
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import urllib.request

import numpy as np
from scipy import sparse


COMMIT = "2b400f43c197bb0eb9bc9802efa2b28b818ab63c"
RAW = f"https://raw.githubusercontent.com/ZIB-AOPT/QOBLIB/{COMMIT}/"
LIMIT = int(11 * 1024**3 * .8)


class Excluded(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def memory_estimates(n, edges):
    # Repository estimates, 32 total trajectories, batches of 8, float64 dynamics.
    # Sparse TRF only. No one-hot groups are synthesized for this general adapter.
    source = 8 * int(n) + 16 * int(edges)
    common = 8 * source + int(n) * 32 * 16 + min(int(edges), 8192) * 8 * 32
    return {"SBM": common + int(n) * 8 * 8 * 16,
            "SVL": common + int(n) * 8 * 8 * 32,
            "TRF": 12 * source + int(n) * 8 * 8 * 24 + int(n) * 128 * 32
                   + min(int(edges), 8192) * 128 * 32 + 10001 * 8}


def weights_for(width):
    width = int(width)
    if width < 0:
        raise Excluded("infeasible_bounds", "Empty integer domain")
    if width == 0:
        return []
    k = width.bit_length() - 1
    return [2**i for i in range(k)] + [width - (2**k - 1)]


def read_qs(path):
    offset = 0.
    with lzma.open(path, "rt") as stream:
        for line in stream:
            if line.startswith("# ObjectiveOffset"):
                offset = float(line.split()[-1])
            if line.strip() and not line.startswith("#"):
                n, expected = map(int, line.split())
                break
        else:
            raise ValueError("Missing QS dimensions")
        if max(memory_estimates(n, 0).values()) > LIMIT:
            raise Excluded("memory_lower_bound", f"{n} variables exceed the reference workspace budget")
        values = np.loadtxt(stream, dtype=np.float64, ndmin=2)
    if expected == 0:
        return np.zeros(n), sparse.csr_matrix((n, n)), offset, {"native_qs": True}
    if values.shape != (expected, 3) or not np.isfinite(values).all():
        raise ValueError("Invalid QS coordinate count or values")
    indices = values[:, :2]
    if not np.equal(indices, np.floor(indices)).all() or indices.min() < 1 or indices.max() > n:
        raise ValueError("Invalid QS variable indices")
    h, t = indices.astype(np.int64).T - 1
    if np.any(h > t):
        raise ValueError("Expected upper-triangular symmetric QS convention")
    diagonal = h == t
    linear = np.bincount(h[diagonal], weights=values[diagonal, 2], minlength=n)
    # QS stores symmetric-matrix entries: off-diagonals occur twice in x^T Q x.
    quadratic = sparse.coo_matrix((2 * values[~diagonal, 2], (h[~diagonal], t[~diagonal])), shape=(n, n)).tocsr()
    quadratic.sum_duplicates()
    quadratic.eliminate_zeros()
    return linear, quadratic, offset, {"native_qs": True, "qs_offdiagonal_multiplier": 2}


def integral_rows(matrix, rhs):
    """Scale rational linear rows to primitive integer rows, without rounding."""
    matrix = matrix.copy().tocsr()
    right = np.empty(len(rhs), dtype=np.float64)
    for row in range(matrix.shape[0]):
        lo, hi = matrix.indptr[row:row + 2]
        fractions = [Fraction(str(float(v))) for v in matrix.data[lo:hi]] + [Fraction(str(float(rhs[row])))]
        denominator = math.lcm(*(v.denominator for v in fractions))
        integers = [v.numerator * (denominator // v.denominator) for v in fractions]
        divisor = math.gcd(*integers) or 1
        integers = [v // divisor for v in integers]
        if max(map(abs, integers), default=0) > 2**40:
            raise Excluded("numerical_guard", "Exact rational row scaling exceeds the float64 integer safety limit")
        matrix.data[lo:hi] = integers[:-1]
        right[row] = integers[-1]
    matrix.eliminate_zeros()
    return matrix, right


def convert_lp(path, destination):
    import gurobipy as gp
    unpacked = destination / "source.lp"
    with lzma.open(path, "rb") as source, unpacked.open("wb") as target:
        total = 0
        while chunk := source.read(1024 * 1024):
            total += len(chunk)
            if total > 2 * 1024**3:
                raise Excluded("preparation_limit", "Decompressed LP exceeds 2 GiB preparation limit")
            target.write(chunk)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.read(str(unpacked), env=env)
        variables = model.getVars()
        counts = collections.Counter(v.VType for v in variables)
        if counts.get("C", 0):
            raise Excluded("continuous_variables", f"Contains {counts['C']} continuous variables; no discretization allowed")
        if any(v.VType not in ("B", "I") for v in variables):
            raise Excluded("unsupported_variable_type", str(dict(counts)))
        if model.NumQConstrs or model.NumGenConstrs or model.NumSOS:
            raise Excluded("unsupported_constraints", "Requires an additional exact quadratic/general/SOS constraint reformulation")
        lower, upper = [], []
        owner, bit_weights = [], []
        for i, var in enumerate(variables):
            if not math.isfinite(var.LB) or not math.isfinite(var.UB) or max(abs(var.LB), abs(var.UB)) >= gp.GRB.INFINITY:
                raise Excluded("unbounded_integer", f"No finite integer encoding for {var.VarName}")
            lb, ub = math.ceil(var.LB), math.floor(var.UB)
            if max(abs(lb), abs(ub)) > 2**40:
                raise Excluded("numerical_guard", f"Integer bounds too large for safe float64 encoding: {var.VarName}")
            lower.append(lb)
            upper.append(ub)
            weights = weights_for(ub - lb)
            owner.extend([i] * len(weights))
            bit_weights.extend(weights)
        nbits = len(owner)
        if max(memory_estimates(nbits, 0).values()) > LIMIT:
            raise Excluded("memory_lower_bound", f"{nbits} decision bits alone exceed workspace budget")
        base = np.array(lower, dtype=np.float64)
        B = sparse.coo_matrix((bit_weights, (owner, np.arange(nbits))), shape=(len(variables), nbits), dtype=np.float64).tocsr()
        objective = model.getObjective()
        if isinstance(objective, gp.QuadExpr):
            lin_expr = objective.getLinExpr()
            qi = [objective.getVar1(i).index for i in range(objective.size())]
            qj = [objective.getVar2(i).index for i in range(objective.size())]
            qv = [objective.getCoeff(i) for i in range(objective.size())]
            Q = sparse.coo_matrix((qv, (qi, qj)), shape=(len(variables), len(variables))).tocsr()
        else:
            lin_expr = objective
            Q = sparse.csr_matrix((len(variables), len(variables)))
        c = np.zeros(len(variables))
        for i in range(lin_expr.size()):
            c[lin_expr.getVar(i).index] += lin_expr.getCoeff(i)
        sign = float(model.ModelSense)
        constant = sign * (float(lin_expr.getConstant()) + c @ base + base @ (Q @ base))
        expanded_q = sign * (B.T @ Q @ B)
        linear = sign * np.asarray(B.T @ (c + (Q + Q.T) @ base)).ravel() + expanded_q.diagonal()
        objective_q = sparse.triu(expanded_q + expanded_q.T, k=1).tocsr()
        constraints = model.getConstrs()
        A, rhs = integral_rows(model.getA(), np.array([r.RHS for r in constraints]))
        original_binary = A @ B
        target = rhs - A @ base
        if len(target) and (np.max(np.abs(target)) > 2**50 or
                           np.max(np.asarray(abs(original_binary).sum(axis=1))) > 2**50):
            raise Excluded("numerical_guard", "Expanded integer residual exceeds exact float64 integer range")
        negative = original_binary.copy()
        negative.data = np.minimum(negative.data, 0)
        positive = original_binary.copy()
        positive.data = np.maximum(positive.data, 0)
        minima = np.asarray(negative.sum(axis=1)).ravel()
        maxima = np.asarray(positive.sum(axis=1)).ravel()
        slack_rows, slack_cols, slack_weights = [], [], []
        slack_count = 0
        for row, constraint in enumerate(constraints):
            sense = constraint.Sense
            if ((sense in ("<", "=") and target[row] < minima[row]) or
                (sense in (">", "=") and target[row] > maxima[row])):
                raise Excluded("infeasible_bounds", f"Constraint {constraint.ConstrName} cannot be satisfied within variable bounds")
            if sense == "=":
                continue
            width = int(target[row] - minima[row] if sense == "<" else maxima[row] - target[row])
            weights = weights_for(width)
            slack_rows.extend([row] * len(weights))
            slack_cols.extend(range(slack_count, slack_count + len(weights)))
            slack_weights.extend([v if sense == "<" else -v for v in weights])
            slack_count += len(weights)
        total_bits = nbits + slack_count
        if max(memory_estimates(total_bits, 0).values()) > LIMIT:
            raise Excluded("memory_lower_bound", f"{total_bits} decision/slack bits exceed workspace budget")
        slack = sparse.coo_matrix((slack_weights, (slack_rows, slack_cols)), shape=(len(constraints), slack_count)).tocsr()
        C = sparse.hstack((original_binary, slack), format="csr")
        C.eliminate_zeros()
        pair_upper = min(total_bits * (total_bits - 1) // 2,
                         objective_q.nnz + sum(int(k) * (int(k) - 1) // 2 for k in np.diff(C.indptr)))
        if max(memory_estimates(total_bits, pair_upper).values()) > LIMIT:
            raise Excluded("conservative_pair_limit", "Penalty clique upper-bound exceeds budget; actual overlap has not been materialized")
        objective_range = float(np.abs(linear).sum() + np.abs(objective_q.data).sum())
        penalty = float(np.nextafter(1.01 * objective_range + 1., np.inf))
        if not math.isfinite(penalty):
            raise Excluded("numerical_guard", "Nonfinite sufficient penalty")
        objective_q.resize((total_bits, total_bits))
        linear = np.pad(linear, (0, slack_count))
        gram = (C.T @ C).tocsr()
        linear += penalty * (gram.diagonal() - 2 * np.asarray(C.T @ target).ravel())
        quadratic = objective_q + 2 * penalty * sparse.triu(gram, k=1)
        constant += penalty * float(target @ target)
        np.savez_compressed(destination / "decode.npz", variable_names=np.array([v.VarName for v in variables]),
                            lower=base, upper=np.array(upper), bit_owner=np.array(owner, dtype=np.int64),
                            bit_weight=np.array(bit_weights, dtype=np.float64), decision_bits=nbits,
                            objective_sense=sign)
        sparse.save_npz(destination / "residual_matrix.npz", C)
        np.save(destination / "residual_target.npy", target)
        detail = dict(native_qs=False, original_variables=len(variables), original_constraints=len(constraints),
                      variable_types=dict(counts), decision_bits=nbits, slack_bits=slack_count,
                      penalty=penalty, objective_range_bound=objective_range, objective_sense=sign,
                      constraint_policy="primitive integer residuals; bounded binary slack; squared penalty",
                      arithmetic="Algebraic integer-domain encoding; coefficients stored in float64")
        model.dispose()
    unpacked.unlink()
    return linear, quadratic.tocsr(), constant, detail


def worker(job_path, output):
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
    job = json.loads(Path(job_path).read_text())
    destination = Path(output) / "cases" / job["id"]
    destination.mkdir(parents=True, exist_ok=True)
    row = dict(id=job["id"], family=job["family"], source=job["path"], status="error")
    start = time.monotonic()
    try:
        source = destination / ("source.qs.xz" if job["kind"] == "qs" else "source.lp.xz")
        request = urllib.request.Request(RAW + job["path"], headers={"User-Agent": "margin-testing-qoblib-preparation"})
        sha = hashlib.sha256()
        gitsha = hashlib.sha1(f"blob {job['size']}\0".encode())
        downloaded = 0
        with urllib.request.urlopen(request, timeout=40) as remote, source.open("wb") as local:
            while block := remote.read(1024 * 1024):
                downloaded += len(block)
                if downloaded > max(job["size"] + 1, 512 * 1024**2):
                    raise ValueError("Download exceeds manifest bounds")
                sha.update(block)
                gitsha.update(block)
                local.write(block)
        if downloaded != job["size"] or gitsha.hexdigest() != job["sha"]:
            raise ValueError("Pinned source blob hash mismatch")
        row["sha256"] = sha.hexdigest()
        linear, q, constant, detail = read_qs(source) if job["kind"] == "qs" else convert_lp(source, destination)
        q.sum_duplicates()
        q.eliminate_zeros()
        if not np.isfinite(linear).all() or not np.isfinite(q.data).all() or not math.isfinite(constant):
            raise Excluded("numerical_guard", "Nonfinite final QUBO")
        n = len(linear)
        row.update(variables=n, interactions=q.nnz, memory_estimates=memory_estimates(n, q.nnz), **detail)
        if max(row["memory_estimates"].values()) > LIMIT:
            raise Excluded("memory_estimate", "Final QUBO exceeds reference solver workspace budget")
        coo = q.tocoo()
        np.savez_compressed(destination / "qubo.npz", linear=linear, heads=coo.row.astype(np.uint32),
                            tails=coo.col.astype(np.uint32), biases=coo.data, offset=constant)
        row.update(status="eligible_estimated", qubo=str((destination / "qubo.npz").relative_to(Path(output))))
    except Excluded as error:
        row.update(status=error.status, detail=str(error))
    except Exception as error:
        row.update(status="error", detail=f"{type(error).__name__}: {error}")
    finally:
        temporary_lp = destination / "source.lp"
        if temporary_lp.exists():
            temporary_lp.unlink()
        row["preparation_seconds"] = time.monotonic() - start
        atomic_json(destination / "result.json", row)


def inventory(tree):
    blobs = [x for x in tree["tree"] if x["type"] == "blob"]
    native_families = {x["path"].split("/")[0] for x in blobs if "/models/" in x["path"] and x["path"].endswith(".qs.xz")}
    jobs = []
    for item in blobs:
        path = item["path"]
        family = path.split("/")[0]
        if not family[:2].isdigit() or "/models/" not in path:
            continue
        if family in native_families:
            if not path.endswith(".qs.xz"):
                continue
            kind = "qs"
        else:
            if not path.endswith(".lp.xz"):
                continue
            kind = "lp"
        identifier = family + "/" + hashlib.sha256(path.encode()).hexdigest()[:16]
        jobs.append(dict(id=identifier, family=family, path=path, kind=kind, size=item["size"], sha=item["sha"],
                         case_key=family + "/" + Path(path).name[:-6]))
    return sorted(jobs, key=lambda j: (j["kind"] != "qs", j["size"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
        return
    if args.workers < 1 or args.workers > 8:
        parser.error("workers must be between 1 and 8")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = inventory(json.loads(args.tree.read_text()))
    if (args.output / "manifest.json").exists():
        raise RuntimeError("Existing inventory; use a new output directory")
    atomic_json(args.output / "manifest.json", dict(commit=COMMIT, candidates=jobs, memory_limit=LIMIT,
        memory_policy="Repository estimates, float64, runs32, batch8, steps10000; SVL noise16; sparse TRF candidate batch128",
        data_license="QOBLIB CC-BY-4.0; retain original source attribution", no_continuous_discretization=True))
    results = []
    started = time.time()

    def prepare(job):
        job_path = args.output / "jobs" / (job["id"].replace("/", "_") + ".json")
        atomic_json(job_path, job)
        destination = args.output / "cases" / job["id"]
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "preparation.log").open("w") as log:
            try:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
                                         "--worker", str(job_path)], stdout=log, stderr=subprocess.STDOUT, timeout=180,
                                        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
                path = destination / "result.json"
                row = json.loads(path.read_text()) if path.exists() else dict(id=job["id"], family=job["family"],
                    source=job["path"], status="preparation_limit", detail=f"Worker exit {result.returncode}")
            except subprocess.TimeoutExpired:
                row = dict(id=job["id"], family=job["family"], source=job["path"],
                           status="preparation_timeout", detail="180-second per-formulation preparation cap")
        row["case_key"] = job["case_key"]
        return row

    def publish(complete=False):
        statuses = collections.Counter(r["status"] for r in results)
        by_case = collections.defaultdict(list)
        for row in results:
            by_case[row["case_key"]].append(row)
        families = sorted({j["family"] for j in jobs})
        per_family = []
        for family in families:
            candidates = {j["case_key"] for j in jobs if j["family"] == family}
            eligible = {key for key, rows in by_case.items() if key.startswith(family + "/") and
                        any(r["status"] == "eligible_estimated" for r in rows)}
            per_family.append(dict(family=family, available_cases=len(candidates), eligible_cases=len(eligible)))
        payload = dict(phase="complete" if complete else "preparing", processed=len(results), total_formulations=len(jobs),
                       total_available_cases=len({j["case_key"] for j in jobs}),
                       eligible_cases=sum(r["eligible_cases"] for r in per_family), statuses=dict(statuses),
                       by_family=per_family, elapsed_seconds=time.time() - started,
                       note="Eligibility is conversion + conservative memory admission, not a GPU allocation test; preparation/numerical/upper-bound exclusions may be unresolved.")
        atomic_json(args.output / "status.json", payload)
        if results:
            columns = sorted({k for r in results for k in r})
            temporary = args.output / "inventory.csv.tmp"
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in results)
            temporary.replace(args.output / "inventory.csv")
        if complete:
            atomic_json(args.output / "summary.json", payload)
        print(json.dumps(payload), flush=True)

    publish()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(prepare, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            results.append(row)
            print(f"{len(results)}/{len(jobs)} {row['status']} {row['source']} {row.get('detail', '')}", flush=True)
            if len(results) % 8 == 0:
                publish()
    publish(True)


if __name__ == "__main__":
    main()
