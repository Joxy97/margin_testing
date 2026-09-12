"""Boundary-aware candidate search on one fixed mixed-option stress lattice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from option_pricing import NonsmoothOptionError
from risk_state_generator.option_factor_stress_model import OptionFactorStressModel
from .factor_stress import FactorStressQUBO, FactorStressQUBOConfig, QuadraticStressObjective, _owned
from .factor_stress_repair import (
    FactorStressRepair, FactorStressRepairConfig, FactorStressRepairResult, projectIntegerBall,
)


@dataclass(frozen=True)
class OptionStressSearchResult:
    """Greatest full-model loss found; no global nonlinear or pricing certificate."""

    margin: float
    pnl: float
    coordinates: np.ndarray
    stressedSpots: np.ndarray
    stockPnL: float
    europeanPnL: float
    americanPnL: float
    candidates: tuple[FactorStressRepairResult, ...]
    solverFeasibility: tuple[bool, ...]
    smoothAnchors: int
    nonsmoothAnchors: int
    boundaryCoverage: tuple[tuple[int, bool, bool], ...]

    def __post_init__(self):
        for name in ("coordinates", "stressedSpots"):
            object.__setattr__(self, name, _owned(getattr(self, name)))


def solveOptionFactorStress(model: OptionFactorStressModel, solver, *,
                           config: FactorStressQUBOConfig = FactorStressQUBOConfig(),
                           solverParameters: dict | None = None,
                           repairConfig: FactorStressRepairConfig = FactorStressRepairConfig(),
                           ) -> OptionStressSearchResult:
    """Solve smooth-anchor QUBOs and fully reprice/repair deterministic seeds.

    Center, axes, nearest boundary points, both nearby sides and marginal
    extremes provide nonlocal starts. Side coverage is rechecked on the lattice.
    Multi-anchor Taylor minimization is heuristic; no local trust ball is added.
    """
    if not isinstance(model, OptionFactorStressModel):
        raise TypeError("model must be an OptionFactorStressModel")
    domain = model.domainDiagnostics(config.radius)
    dimension, radius = model.dimension, config.radius
    m = 2**(config.bitsPerCoordinate-1)-1
    scale = radius/m
    seeds: dict[tuple[int, ...], np.ndarray] = {}

    def add(point):
        integers = projectIntegerBall(np.rint(np.asarray(point)/scale).astype(np.int64), m)
        seeds.setdefault(tuple(int(v) for v in integers), integers)
        return integers

    add(np.zeros(dimension))
    for axis in np.eye(dimension):
        for sign in (-1, 1):
            add(sign*radius*axis)
            add(sign*.5*radius*axis)
    for item in domain:
        if not item["boundary_reachable"] or item["loading_norm"] == 0:
            continue
        underlying = model.options._underlyingIndices[item["position"]]
        row = model.equityModel.directions[underlying]
        norm, distance = item["loading_norm"], item["boundary_log_distance"]
        closest = add(distance*row/(norm*norm))
        for sign in (-1, 1):
            add(sign*radius*row/norm)
            add(distance*row/(norm*norm)+sign*scale*row/norm)
            for axis in np.eye(dimension):
                add((closest+sign*axis)*scale)

    # Always have an encoding for seed repair, even if every derivative anchor
    # is nonsmooth. A zero objective creates only the common integer geometry.
    base = FactorStressQUBO.build(QuadraticStressObjective(0., np.zeros(dimension),
        np.zeros((dimension, dimension))), config)
    repair = FactorStressRepair(model, base, repairConfig)
    candidates, raw_feasibility = [], []
    smooth, nonsmooth = 0, 0
    for integers in seeds.values():
        candidates.append(repair.repair(base.encodeIntegers(integers)))
        anchor = integers*scale
        try:
            local = model.localQuadratic(anchor)
        except NonsmoothOptionError:
            nonsmooth += 1
            continue
        hessian, gradient, constant = local.hessian, local.gradient, local.value
        # Express the anchor polynomial in the original z coordinates. A, c
        # and the radius stay fixed; each polynomial gets its own penalty bound.
        objective = QuadraticStressObjective(constant-gradient@anchor+.5*anchor@hessian@anchor,
            gradient-hessian@anchor, hessian)
        encoding = FactorStressQUBO.build(objective, config)
        solved = solver.solve(encoding.problem, dict(solverParameters or {}))
        raw_feasibility.append(bool(encoding.diagnostics(solved.sample)["encoding_feasible"]))
        # Project before any price evaluation: a raw binary sample may be far
        # outside the screened pricing domain. Rebuild all product/slack bits.
        projected = projectIntegerBall(encoding.integerCoordinates(solved.sample), m)
        candidates.append(repair.repair(base.encodeIntegers(projected)))
        smooth += 1
    best = min(candidates, key=lambda result: result.pnl)
    point = best.integers*scale
    logs = model._logReturns(point)
    styles = model.options.pnlByStyle(logs)
    coverage = []
    points = np.array(list(seeds.values()))*scale
    for item in domain:
        if item["boundary_reachable"]:
            underlying = model.options._underlyingIndices[item["position"]]
            sides = points@model.equityModel.directions[underlying]-item["boundary_log_distance"]
            coverage.append((item["position"], bool(np.any(sides < -1e-10)), bool(np.any(sides > 1e-10))))
    return OptionStressSearchResult(max(0., -best.pnl), best.pnl, point,
        model.options.spotPrices*np.exp(logs), float(model.equityModel.pnl(point)),
        float(styles["E"]), float(styles["A"]), tuple(candidates), tuple(raw_feasibility),
        smooth, nonsmooth, tuple(coverage))
