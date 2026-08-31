#!/usr/bin/env python3
"""Rolling, leakage-safe QUBO and greedy-baseline margin backtester.

The orchestrator can evaluate either method or both from the same rolling PCA
and scenarios.  The QUBO path keeps BQMs in memory and batches independent
scenario graphs for simulated bifurcation.  The baseline path independently
selects the minimum weighted-PnL state for every asset.  CUDA is selected
automatically when a CUDA-enabled PyTorch installation and NVIDIA device are
available; otherwise the identical sparse dynamics run on CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from market_to_qubo import (
    DEFAULT_GRID_POINTS,
    PCAResult,
    PlausibilityModel,
    PreparedData,
    build_qubos,
    build_scenario_grid,
    conditional_neighbors_device,
    convert_z_to_returns,
    create_z_shock_grids,
    fit_ew_pca,
    fit_plausibility_model,
    plausibility_score,
    plausibility_state_values,
    prepare_shock_grid_context,
    read_close_prices,
    read_portfolio,
)
from qubo_model import CompactQubo, repair_one_hot
from sbm_torch import SBMConfig, SBMSolveResult, TorchBatchSBMSolver, resolve_torch_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_MARKET_ROOT = PROJECT_ROOT / "synthetic_market"
RESULT_COLUMNS = (
    "date",
    "method",
    "calibration_start",
    "calibration_end",
    "calibration_returns",
    "assets",
    "scenarios",
    "selected_scenario",
    "selected_energy",
    "raw_energy",
    "repaired_energy",
    "raw_one_hot_violations",
    "raw_feasible_candidates",
    "solver_candidates",
    "plausibility_score",
    "plausibility_threshold",
    "hard_plausibility_feasible",
    "margin",
    "worst_scenario_pnl",
    "realized_pnl",
    "realized_loss",
    "gross_exposure",
    "signed_margin_error",
    "pca_seconds",
    "scenario_build_seconds",
    "qubo_build_seconds",
    "solve_seconds",
    "baseline_seconds",
    "day_seconds",
    "device",
    "steps",
    "runs",
    "seed",
)
RESOURCE_COLUMNS = (
    "date",
    "method",
    "device",
    "correlation_device",
    "pca_dtype",
    "sbm_dtype",
    "assets",
    "scenarios",
    "scenario_batch_size",
    "scenario_batches",
    "build_workers",
    "steps",
    "runs",
    "run_batch_size",
    "solver_run_batches_per_scenario_batch",
    "pca_seconds",
    "scenario_build_seconds",
    "qubo_build_seconds",
    "solve_seconds",
    "baseline_seconds",
    "day_seconds",
    "resource_samples",
    "mean_cpu_percent",
    "peak_cpu_percent",
    "mean_ram_mib",
    "peak_ram_mib",
    "mean_gpu_percent",
    "peak_gpu_percent",
    "mean_vram_mib",
    "peak_vram_mib",
    "gpu_vram_total_mib",
)


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame
    simple_returns: pd.DataFrame
    log_returns: pd.DataFrame
    evaluation_dates: pd.DatetimeIndex
    tickers: list[str]
    weights: np.ndarray
    gross_exposure: float


@dataclass
class ScenarioProblem:
    scenario_index: int
    bqm: CompactQubo
    group_offsets: np.ndarray
    portfolio_linear: np.ndarray
    gross_exposure: float
    plausibility: PlausibilityModel
    plausibility_states: tuple[np.ndarray, ...]


@dataclass
class ScenarioData:
    scenario_index: int
    shocks: Any
    asset_grids: Sequence[dict[str, np.ndarray]]
    group_offsets: np.ndarray
    portfolio_linear: np.ndarray


@dataclass(frozen=True)
class BacktestConfig:
    subfolder: Path
    output: Path
    backtest_start: pd.Timestamp | None = None
    backtest_end: pd.Timestamp | None = None
    window: int = 125
    ew_lambda: float = 0.93
    grid_points: tuple[int, int] = (21, 5)
    scenario_indices: tuple[int, ...] | None = None
    method: str = "qubo"
    z_bins: int = 21
    nearest: int = 100
    residual_sigma_range: float = 5.0
    distance_inflation_alpha: float = 0.50
    distance_inflation_power: float = 2.0
    max_inflation_factor: float = 5.0
    lambda_one_hot: float = 1.0
    lambda_compat: float = 0.1
    top_k_neighbors: int = 5
    plausibility_confidence: float = 0.998
    device: str = "auto"
    pca_dtype: str = "float32"
    correlation_device: str = "auto"
    correlation_block_mib: int = 256
    scenario_batch_size: int = 1
    build_workers: int = 1
    decode_sweeps: int = 100
    seed: int = 1
    day_limit: int | None = None
    resume: bool = False
    progress_interval: float = 10.0
    evaluation_dates: tuple[str, ...] | None = None
    sbm: SBMConfig = SBMConfig()

    def validate(self) -> None:
        if self.window < 2:
            raise ValueError("window must contain at least two returns")
        if not 0.0 < self.ew_lambda <= 1.0:
            raise ValueError("ew_lambda must be in (0, 1]")
        if len(self.grid_points) != 2 or any(
            value < 1 or value % 2 == 0 for value in self.grid_points
        ):
            raise ValueError("grid_points must contain two positive odd values")
        if self.z_bins < 1 or self.nearest < 2:
            raise ValueError("z_bins must be positive and nearest must be at least 2")
        if not math.isfinite(self.residual_sigma_range) or self.residual_sigma_range <= 0:
            raise ValueError("residual_sigma_range must be finite and positive")
        if not math.isfinite(self.distance_inflation_alpha) or self.distance_inflation_alpha < 0:
            raise ValueError("distance_inflation_alpha must be finite and non-negative")
        if not math.isfinite(self.distance_inflation_power) or self.distance_inflation_power <= 0:
            raise ValueError("distance_inflation_power must be finite and positive")
        if not math.isfinite(self.max_inflation_factor) or self.max_inflation_factor < 1:
            raise ValueError("max_inflation_factor must be finite and at least one")
        if not math.isfinite(self.lambda_one_hot) or self.lambda_one_hot < 0:
            raise ValueError("lambda_one_hot must be finite and non-negative")
        if not math.isfinite(self.lambda_compat) or self.lambda_compat < 0:
            raise ValueError("lambda_compat must be finite and non-negative")
        if self.top_k_neighbors < 0:
            raise ValueError("top_k_neighbors cannot be negative")
        if not 0.0 < self.plausibility_confidence <= 1.0:
            raise ValueError("plausibility_confidence must be in (0, 1]")
        if self.scenario_batch_size < 1 or self.build_workers < 1:
            raise ValueError("batch size and build workers must be positive")
        if self.decode_sweeps < 0:
            raise ValueError("decode_sweeps cannot be negative")
        if self.correlation_block_mib < 1:
            raise ValueError("correlation_block_mib must be positive")
        if self.pca_dtype not in {"float32", "float64"}:
            raise ValueError("pca_dtype must be float32 or float64")
        if self.day_limit is not None and self.day_limit < 1:
            raise ValueError("day_limit must be positive")
        if not math.isfinite(self.progress_interval) or self.progress_interval < 0:
            raise ValueError("progress_interval must be finite and non-negative")
        if self.scenario_indices is not None and not self.scenario_indices:
            raise ValueError("scenario_indices cannot be empty")
        if self.method not in {"qubo", "baseline", "both"}:
            raise ValueError("method must be qubo, baseline, or both")
        self.sbm.validate()


class ResourceMonitor:
    """Sample process resources and the selected NVIDIA device during one day."""

    def __init__(self, device_name: str, interval_seconds: float = 1.0) -> None:
        import psutil

        self.process = psutil.Process()
        self.device_name = device_name
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._cpu_started = 0.0
        self.ram_mib: list[float] = []
        self.cpu_percent: list[float] = []
        self.gpu_percent: list[float] = []
        self.vram_mib: list[float] = []
        self.gpu_vram_total_mib: float | None = None

    def _gpu_index(self) -> str | None:
        if not self.device_name.startswith("cuda"):
            return None
        return self.device_name.partition(":")[2] or "0"

    def _sample(self) -> None:
        try:
            self.ram_mib.append(self.process.memory_info().rss / (1024 * 1024))
            self.cpu_percent.append(float(self.process.cpu_percent(interval=None)))
        except Exception:
            pass
        gpu_index = self._gpu_index()
        if gpu_index is None:
            return
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_index}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            values = [float(item.strip()) for item in completed.stdout.splitlines()[0].split(",")]
            self.gpu_percent.append(values[0])
            self.vram_mib.append(values[1])
            self.gpu_vram_total_mib = values[2]
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        cpu_times = self.process.cpu_times()
        self._cpu_started = float(cpu_times.user + cpu_times.system)
        self._started = time.perf_counter()
        self.process.cpu_percent(interval=None)
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | int | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1.0)
        self._sample()
        elapsed = max(time.perf_counter() - self._started, 1e-12)
        cpu_times = self.process.cpu_times()
        cpu_seconds = float(cpu_times.user + cpu_times.system) - self._cpu_started
        # psutil's process convention is 100% for one fully occupied CPU core.
        mean_cpu = 100.0 * cpu_seconds / elapsed

        def mean(values: list[float]) -> float | None:
            return float(np.mean(values)) if values else None

        def peak(values: list[float]) -> float | None:
            return float(np.max(values)) if values else None

        return {
            "resource_samples": max(len(self.ram_mib), len(self.gpu_percent)),
            "mean_cpu_percent": mean_cpu,
            "peak_cpu_percent": peak(self.cpu_percent),
            "mean_ram_mib": mean(self.ram_mib),
            "peak_ram_mib": peak(self.ram_mib),
            "mean_gpu_percent": mean(self.gpu_percent),
            "peak_gpu_percent": peak(self.gpu_percent),
            "mean_vram_mib": mean(self.vram_mib),
            "peak_vram_mib": peak(self.vram_mib),
            "gpu_vram_total_mib": self.gpu_vram_total_mib,
        }


def resolve_market_folder(value: str | Path) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_absolute():
        folder = supplied.resolve()
    elif supplied.parts and supplied.parts[0].lower() == "synthetic_market":
        folder = (PROJECT_ROOT / supplied).resolve()
    else:
        folder = (SYNTHETIC_MARKET_ROOT / supplied).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"market universe folder does not exist: {folder}")
    return folder


def _merge_price_panels(history: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    if list(history.columns) != list(backtest.columns):
        raise ValueError("historical and backtesting asset columns must match")
    combined = pd.concat([history, backtest]).sort_index(kind="stable")
    for date in combined.index[combined.index.duplicated()].unique():
        duplicate = combined.loc[[date]].to_numpy(dtype=float)
        if not np.allclose(duplicate, duplicate[0], rtol=1e-12, atol=1e-12):
            raise ValueError(f"conflicting close prices for duplicate date {date.date()}")
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    if not combined.index.is_monotonic_increasing:
        raise ValueError("combined market dates are not increasing")
    return combined


def load_market_data(config: BacktestConfig) -> MarketData:
    folder = config.subfolder
    history = read_close_prices(folder / "historical_close.csv")
    backtest = read_close_prices(folder / "backtest_close.csv")
    prices = _merge_price_panels(history, backtest)
    simple_returns = prices.pct_change(fill_method=None).iloc[1:]
    log_returns = np.log(prices).diff().iloc[1:]
    if not np.isfinite(simple_returns.to_numpy()).all() or not np.isfinite(
        log_returns.to_numpy()
    ).all():
        raise ValueError("combined market returns contain non-finite values")

    if config.backtest_start is None:
        if len(backtest.index) < 2:
            raise ValueError(
                "backtest_close.csv needs an anchor close plus one evaluation date, "
                "or --backtest-start must be supplied"
            )
        start = backtest.index[1]
    else:
        start = config.backtest_start
    end = config.backtest_end or backtest.index[-1]
    evaluation_dates = backtest.index[(backtest.index >= start) & (backtest.index <= end)]
    evaluation_dates = evaluation_dates.intersection(simple_returns.index)
    if config.day_limit is not None:
        evaluation_dates = evaluation_dates[: config.day_limit]
    if config.evaluation_dates is not None:
        selected = set(config.evaluation_dates)
        evaluation_dates = pd.DatetimeIndex(
            [
                date
                for date in evaluation_dates
                if pd.Timestamp(date).date().isoformat() in selected
            ]
        )
    if evaluation_dates.empty:
        raise ValueError("no evaluation dates remain after applying the backtest range")

    _, weights = read_portfolio(folder / "portfolio.csv", list(prices.columns))
    gross = float(np.abs(weights).sum())
    if not math.isfinite(gross) or gross <= 0.0:
        raise ValueError("portfolio gross exposure must be positive")
    return MarketData(
        prices=prices,
        simple_returns=simple_returns,
        log_returns=log_returns,
        evaluation_dates=evaluation_dates,
        tickers=list(prices.columns),
        weights=weights,
        gross_exposure=gross,
    )


def rolling_ew_pca(
    log_returns: pd.DataFrame,
    decay: float,
    device_name: str,
    dtype_name: str,
) -> tuple[PreparedData, PCAResult]:
    """Fit a two-component EW PCA using the shared exporter/backtester path."""

    values = log_returns.to_numpy(dtype=np.float64, copy=False)
    observations, assets = values.shape
    if observations < 2 or assets < 2:
        raise ValueError("rolling PCA requires at least two returns and two assets")
    empty = np.empty((0, assets), dtype=np.float64)
    scaler, z_values, _, pca = fit_ew_pca(
        values,
        empty,
        2,
        decay,
        device_name=device_name,
        dtype_name=dtype_name,
    )
    empty = np.empty((0, assets), dtype=np.float64)
    prepared = PreparedData(
        tickers=list(log_returns.columns),
        historical_close=pd.DataFrame(),
        backtest_close=pd.DataFrame(),
        historical_returns=np.expm1(log_returns),
        historical_log_returns=log_returns,
        backtest_returns=pd.DataFrame(columns=log_returns.columns),
        backtest_log_returns=pd.DataFrame(columns=log_returns.columns),
        historical_z=z_values,
        backtest_z=empty,
        standardizer=scaler,
    )
    return prepared, pca


def _scenario_data(
    scenario_index: int,
    prepared: PreparedData,
    pca: PCAResult,
    scenario_grid: np.ndarray,
    shock_context: Any,
    market: MarketData,
    config: BacktestConfig,
) -> ScenarioData:
    shocks = create_z_shock_grids(
        prepared,
        pca,
        scenario_grid,
        scenario_index,
        config.window,
        config.nearest,
        config.z_bins,
        config.residual_sigma_range,
        config.distance_inflation_alpha,
        config.distance_inflation_power,
        config.max_inflation_factor,
        context=shock_context,
    )
    asset_grids = convert_z_to_returns(prepared.standardizer, shocks.asset_z_grids)
    state_counts = np.fromiter(
        (len(grid["z"]) for grid in asset_grids),
        dtype=np.int64,
        count=len(asset_grids),
    )
    group_offsets = np.empty(len(state_counts) + 1, dtype=np.int64)
    group_offsets[0] = 0
    np.cumsum(state_counts, out=group_offsets[1:])
    portfolio_linear = np.concatenate(
        [
            market.weights[asset] * grid["simple_return"]
            for asset, grid in enumerate(asset_grids)
        ]
    ).astype(np.float64, copy=False)
    return ScenarioData(
        scenario_index=scenario_index,
        shocks=shocks,
        asset_grids=asset_grids,
        group_offsets=group_offsets,
        portfolio_linear=portfolio_linear,
    )


def _scenario_problem(
    data: ScenarioData,
    market: MarketData,
    config: BacktestConfig,
    correlation_device: str,
) -> ScenarioProblem:
    conditional_std, neighbor_indices, neighbor_correlations = (
        conditional_neighbors_device(
            data.shocks.inflated_residuals,
            config.top_k_neighbors,
            correlation_device,
            config.correlation_block_mib,
        )
    )
    plausibility = fit_plausibility_model(
        data.shocks.inflated_residuals,
        config.top_k_neighbors,
        config.plausibility_confidence,
        conditional_standard_deviation=conditional_std,
        neighbor_indices=neighbor_indices,
        neighbor_correlations=neighbor_correlations,
    )
    state_values = plausibility_state_values(
        data.asset_grids, data.shocks.z_hat, plausibility
    )
    center = np.fromiter(
        (int(np.argmin(np.abs(values))) for values in state_values),
        dtype=np.int64,
        count=len(state_values),
    )
    center_score = plausibility_score(state_values, center, plausibility)
    if center_score > plausibility.threshold:
        # A finite grid can miss the continuous residual mean.  Widen only by
        # the minimum center discretization error so the hard set is non-empty.
        plausibility = replace(
            plausibility,
            threshold=center_score
            + 64.0 * np.finfo(float).eps * max(1.0, center_score),
        )
    bqm, _, _ = build_qubos(
        data.asset_grids,
        market.weights,
        data.shocks.z_hat,
        conditional_std,
        neighbor_indices,
        neighbor_correlations,
        config.lambda_one_hot,
        config.lambda_compat,
        plausibility_model=plausibility,
        gross_exposure=market.gross_exposure,
        numeric_labels=True,
        compact=True,
    )
    if not isinstance(bqm, CompactQubo):  # pragma: no cover - fixed call contract
        raise TypeError("rolling backtests require a compact QUBO")
    return ScenarioProblem(
        scenario_index=data.scenario_index,
        bqm=bqm,
        group_offsets=data.group_offsets,
        portfolio_linear=data.portfolio_linear,
        gross_exposure=market.gross_exposure,
        plausibility=plausibility,
        plausibility_states=state_values,
    )


def greedy_baseline_pnl(data: ScenarioData) -> float:
    """Select each asset's minimum weighted PnL state independently."""

    minima = np.minimum.reduceat(data.portfolio_linear, data.group_offsets[:-1])
    return float(minima.sum())


def decode_hard_plausible(
    problem: ScenarioProblem,
    one_hot_sample: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Minimize portfolio P&L by categorical descent inside the hard cutoff."""

    offsets = problem.group_offsets
    assets = len(offsets) - 1
    tolerance = 128.0 * np.finfo(float).eps * max(
        1.0, problem.plausibility.threshold
    )
    normalized_portfolio = problem.portfolio_linear / problem.gross_exposure
    pnl_states = tuple(
        normalized_portfolio[int(offsets[a]) : int(offsets[a + 1])]
        for a in range(assets)
    )

    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(assets)]
    for head, tail, weight in zip(
        problem.plausibility.edge_heads,
        problem.plausibility.edge_tails,
        problem.plausibility.edge_weights,
    ):
        lower, upper, edge_weight = int(head), int(tail), float(weight)
        neighbors[lower].append((upper, edge_weight))
        neighbors[upper].append((lower, edge_weight))

    def sample_to_selection(sample: np.ndarray) -> np.ndarray:
        return np.fromiter(
            (
                int(np.flatnonzero(sample[int(offsets[a]) : int(offsets[a + 1])])[0])
                for a in range(assets)
            ),
            dtype=np.int64,
            count=assets,
        )

    def selection_to_sample(selection: np.ndarray) -> np.ndarray:
        sample = np.zeros(int(offsets[-1]), dtype=np.uint8)
        for asset, state in enumerate(selection):
            sample[int(offsets[asset]) + int(state)] = 1
        return sample

    def descend(seed: np.ndarray) -> tuple[np.ndarray, float, float]:
        selection = np.asarray(seed, dtype=np.int64).copy()
        values = np.fromiter(
            (
                problem.plausibility_states[asset][int(state)]
                for asset, state in enumerate(selection)
            ),
            dtype=float,
            count=assets,
        )
        contexts = np.zeros(assets, dtype=float)
        for asset, adjacent in enumerate(neighbors):
            contexts[asset] = sum(weight * values[other] for other, weight in adjacent)
        score = plausibility_score(
            problem.plausibility_states, selection, problem.plausibility
        )
        pnl = float(sum(pnl_states[a][selection[a]] for a in range(assets)))

        while True:
            changed = False
            for asset in range(assets):
                old_state = int(selection[asset])
                old_value = float(values[asset])
                candidates = problem.plausibility_states[asset]
                score_changes = (
                    candidates * candidates
                    - old_value * old_value
                    - 2.0 * (candidates - old_value) * contexts[asset]
                )
                feasible = score + score_changes <= problem.plausibility.threshold + tolerance
                if not np.any(feasible):
                    continue
                candidate_pnl = pnl_states[asset]
                feasible_indices = np.flatnonzero(feasible)
                best_local = int(
                    feasible_indices[
                        np.lexsort(
                            (
                                score_changes[feasible_indices],
                                candidate_pnl[feasible_indices],
                            )
                        )[0]
                    ]
                )
                pnl_change = float(candidate_pnl[best_local] - candidate_pnl[old_state])
                if pnl_change >= -tolerance:
                    continue
                new_value = float(candidates[best_local])
                delta_value = new_value - old_value
                score = max(0.0, score + float(score_changes[best_local]))
                pnl += pnl_change
                selection[asset] = best_local
                values[asset] = new_value
                for other, edge_weight in neighbors[asset]:
                    contexts[other] += edge_weight * delta_value
                changed = True
            if not changed:
                break
        return selection, pnl, score

    repaired_selection = sample_to_selection(one_hot_sample)
    center_selection = np.fromiter(
        (int(np.argmin(np.abs(values))) for values in problem.plausibility_states),
        dtype=np.int64,
        count=assets,
    )
    greedy_selection = np.fromiter(
        (int(np.argmin(values)) for values in pnl_states),
        dtype=np.int64,
        count=assets,
    )
    seeds = (repaired_selection, center_selection, greedy_selection)
    candidates: list[tuple[float, float, np.ndarray]] = []
    for seed in seeds:
        seed_score = plausibility_score(
            problem.plausibility_states, seed, problem.plausibility
        )
        if seed_score <= problem.plausibility.threshold + tolerance:
            selection, pnl, score = descend(seed)
            candidates.append((pnl, score, selection))
    if not candidates:
        raise RuntimeError("discretized state space contains no plausible portfolio")
    _, score, selection = min(candidates, key=lambda item: (item[0], item[1]))
    sample = selection_to_sample(selection)
    pnl = float(np.dot(problem.portfolio_linear, sample))
    final_score = plausibility_score(
        problem.plausibility_states, selection, problem.plausibility
    )
    if final_score > problem.plausibility.threshold + tolerance:  # pragma: no cover
        raise RuntimeError("hard-plausibility decoder returned an infeasible sample")
    return sample, float(pnl), float(final_score)


def _scenario_batches(indices: Sequence[int], size: int) -> list[list[int]]:
    return [list(indices[start : start + size]) for start in range(0, len(indices), size)]


def _requested_methods(method: str) -> tuple[str, ...]:
    return ("qubo", "baseline") if method == "both" else (method,)


def _existing_method_dates(output: Path) -> set[tuple[str, str]]:
    if not output.exists():
        return set()
    frame = pd.read_csv(output, dtype=str)
    if "method" not in frame:
        return {(date, "qubo") for date in frame["date"]}
    return set(zip(frame["date"], frame["method"]))


def _upgrade_csv_schema(
    output: Path,
    columns: Sequence[str],
    defaults: dict[str, Any],
) -> None:
    """Upgrade older resumable CSVs before appending newly added columns."""

    if not output.exists() or output.stat().st_size == 0:
        return
    frame = pd.read_csv(output)
    if tuple(frame.columns) == tuple(columns):
        return
    unknown = [column for column in frame.columns if column not in columns]
    if unknown:
        raise ValueError(
            f"cannot resume {output}: unrecognized columns {unknown}"
        )
    for column in columns:
        if column not in frame:
            frame[column] = defaults.get(column, np.nan)
    temporary = output.with_suffix(output.suffix + ".schema.tmp")
    frame.to_csv(temporary, index=False, columns=columns)
    temporary.replace(output)


def _diagnostics_path(output: Path) -> Path:
    return output.with_suffix(".diagnostics.csv")


def _append_csv(output: Path, row: dict[str, Any], columns: Sequence[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists() and output.stat().st_size > 0
    with output.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        destination.flush()


def _append_result(output: Path, row: dict[str, Any]) -> None:
    _append_csv(output, row, RESULT_COLUMNS)


def write_error_summary(output: Path) -> Path:
    """Write signed-error statistics separately for every margin method."""

    frame = pd.read_csv(output)
    if frame.empty:
        raise ValueError("cannot summarize an empty backtest result")
    if "method" not in frame:
        frame.insert(1, "method", "qubo")
    methods: dict[str, dict[str, Any]] = {}
    for method, method_frame in frame.groupby("method", sort=False):
        errors = method_frame["signed_margin_error"].to_numpy(dtype=float)
        shortfalls = errors[errors < 0.0]
        quantiles = np.quantile(errors, [0.01, 0.05, 0.50, 0.95, 0.99])
        methods[str(method)] = {
            "days": int(len(errors)),
            "mean_signed_margin_error": float(errors.mean()),
            "std_signed_margin_error": (
                float(errors.std(ddof=1)) if len(errors) > 1 else 0.0
            ),
            "minimum": float(errors.min()),
            "q01": float(quantiles[0]),
            "q05": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q95": float(quantiles[3]),
            "q99": float(quantiles[4]),
            "maximum": float(errors.max()),
            "under_margin_days": int(len(shortfalls)),
            "under_margin_rate": float(len(shortfalls) / len(errors)),
            "mean_shortfall": float(shortfalls.mean()) if len(shortfalls) else 0.0,
            "worst_shortfall": float(shortfalls.min()) if len(shortfalls) else 0.0,
            "total_runtime_seconds": float(method_frame["day_seconds"].sum()),
        }
    summary: dict[str, Any] = {"methods": methods}
    # Preserve the original top-level fields for single-method consumers while
    # making the per-method grouping authoritative for comparisons.
    if len(methods) == 1:
        summary.update(next(iter(methods.values())))
    summary_path = output.with_suffix(".summary.json")
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return summary_path


def run_backtest(config: BacktestConfig) -> pd.DataFrame:
    config.validate()
    requested_methods = _requested_methods(config.method)
    run_qubo = "qubo" in requested_methods
    run_baseline = "baseline" in requested_methods
    if not config.resume:
        for generated in (
            config.output,
            config.output.with_suffix(".summary.json"),
            _diagnostics_path(config.output),
        ):
            if generated.exists():
                generated.unlink()
    else:
        _upgrade_csv_schema(
            config.output,
            RESULT_COLUMNS,
            {"method": "qubo", "scenario_build_seconds": 0.0, "baseline_seconds": 0.0},
        )
        _upgrade_csv_schema(
            _diagnostics_path(config.output),
            RESOURCE_COLUMNS,
            {"method": "qubo", "scenario_build_seconds": 0.0, "baseline_seconds": 0.0},
        )
    resolved_device = resolve_torch_device(config.device)
    solver = TorchBatchSBMSolver(resolved_device) if run_qubo else None
    pca_device = resolved_device
    correlation_requested = (
        resolved_device
        if config.correlation_device == "auto"
        else config.correlation_device
    )
    correlation_device = (
        resolve_torch_device(correlation_requested) if run_qubo else "not-used"
    )
    if run_qubo and correlation_device != "cpu" and config.build_workers != 1:
        raise ValueError("GPU correlation construction requires build_workers=1")
    market = load_market_data(config)
    completed = _existing_method_dates(config.output) if config.resume else set()
    output_rows: list[dict[str, Any]] = []
    all_scenario_count = int(np.prod(config.grid_points, dtype=np.int64))
    scenario_indices = (
        tuple(range(all_scenario_count))
        if config.scenario_indices is None
        else tuple(dict.fromkeys(config.scenario_indices))
    )
    invalid = [index for index in scenario_indices if not 0 <= index < all_scenario_count]
    if invalid:
        raise ValueError(
            f"scenario indices must be in [0, {all_scenario_count - 1}]: {invalid}"
        )
    scenario_batch_count = math.ceil(
        len(scenario_indices) / config.scenario_batch_size
    )

    for day_position, date in enumerate(market.evaluation_dates):
        date_text = pd.Timestamp(date).date().isoformat()
        if all((date_text, method) in completed for method in requested_methods):
            print(f"[{day_position + 1}/{len(market.evaluation_dates)}] {date_text}: resumed")
            continue
        day_started = time.perf_counter()
        progress_context = f"[{resolved_device} {date_text}]"
        if config.progress_interval > 0:
            print(
                "\n".join(
                    (
                        "=" * 88,
                        f"DAY {day_position + 1}/{len(market.evaluation_dates)} | "
                        f"{date_text} | {resolved_device}",
                        "-" * 88,
                        f"Assets: {len(market.tickers):,} | Portfolios: 1 | "
                        f"Scenarios: {len(scenario_indices)} | "
                        f"Batches: {scenario_batch_count} | "
                        f"Steps: {config.sbm.steps if run_qubo else 0} | "
                        f"Runs: {config.sbm.runs if run_qubo else 0}",
                        f"Method: {config.method} | Solver: {resolved_device} | "
                        f"PCA: {pca_device} | Correlation: {correlation_device}",
                        "-" * 88,
                    )
                ),
                flush=True,
            )
        resources = ResourceMonitor(resolved_device)
        resources.start()
        prior = market.log_returns.loc[market.log_returns.index < date].tail(config.window)
        if len(prior) != config.window:
            raise ValueError(
                f"{date_text} has {len(prior)} prior returns; {config.window} required"
            )
        pca_started = time.perf_counter()
        prepared, pca = rolling_ew_pca(
            prior, config.ew_lambda, pca_device, config.pca_dtype
        )
        _, scenario_grid, _ = build_scenario_grid(pca, config.grid_points, 1.0)
        shock_context = prepare_shock_grid_context(prepared, pca, config.window)
        pca_seconds = time.perf_counter() - pca_started
        if config.progress_interval > 0:
            print(
                f"{progress_context} PCA completed in {pca_seconds:.2f}s | "
                f"Factor scenarios enumerated: {len(scenario_indices)}/{len(scenario_indices)}",
                flush=True,
            )
            print("-" * 88, flush=True)

        best_qubo: tuple[
            float, int, float, float, float, int, int, int, float, float
        ] | None = None
        best_baseline: tuple[float, int] | None = None
        scenario_build_seconds = 0.0
        build_seconds = 0.0
        solve_seconds = 0.0
        decode_seconds = 0.0
        baseline_seconds = 0.0
        scenario_progress_started = time.perf_counter()
        last_progress_report = scenario_progress_started
        completed_scenarios = 0
        for batch_position, scenario_batch in enumerate(
            _scenario_batches(scenario_indices, config.scenario_batch_size), start=1
        ):
            scenario_started = time.perf_counter()
            if config.build_workers == 1:
                scenario_data = [
                    _scenario_data(
                        scenario_index,
                        prepared,
                        pca,
                        scenario_grid,
                        shock_context,
                        market,
                        config,
                    )
                    for scenario_index in scenario_batch
                ]
            else:
                with ThreadPoolExecutor(max_workers=config.build_workers) as executor:
                    scenario_data = list(
                        executor.map(
                            lambda scenario_index: _scenario_data(
                                scenario_index,
                                prepared,
                                pca,
                                scenario_grid,
                                shock_context,
                                market,
                                config,
                            ),
                            scenario_batch,
                        )
                    )
            batch_scenario_seconds = time.perf_counter() - scenario_started
            scenario_build_seconds += batch_scenario_seconds

            batch_baseline_seconds = 0.0
            if run_baseline:
                baseline_started = time.perf_counter()
                for data in scenario_data:
                    candidate = (greedy_baseline_pnl(data), data.scenario_index)
                    if best_baseline is None or candidate[0] < best_baseline[0]:
                        best_baseline = candidate
                batch_baseline_seconds = time.perf_counter() - baseline_started
                baseline_seconds += batch_baseline_seconds

            if not run_qubo:
                completed_scenarios += len(scenario_batch)
                progress_now = time.perf_counter()
                if config.progress_interval > 0 and (
                    completed_scenarios == len(scenario_indices)
                    or progress_now - last_progress_report >= config.progress_interval
                ):
                    progress_elapsed = progress_now - scenario_progress_started
                    progress_rate = completed_scenarios / max(progress_elapsed, 1e-12)
                    progress_eta = (
                        len(scenario_indices) - completed_scenarios
                    ) / progress_rate
                    scenario_label = (
                        f"Scenario {scenario_batch[0] + 1}/{all_scenario_count}"
                        if len(scenario_batch) == 1
                        else f"Batch {batch_position}/{scenario_batch_count} "
                        f"({len(scenario_batch)} scenarios)"
                    )
                    print(
                        f"{progress_context} {scenario_label} | "
                        f"data={batch_scenario_seconds:.2f}s | "
                        f"baseline={batch_baseline_seconds:.4f}s | "
                        f"progress={completed_scenarios}/{len(scenario_indices)} "
                        f"({100.0 * completed_scenarios / len(scenario_indices):.1f}%) | "
                        f"elapsed={progress_elapsed:.1f}s | ETA={progress_eta:.1f}s",
                        flush=True,
                    )
                    last_progress_report = progress_now
                del scenario_data
                continue

            batch_build_started = time.perf_counter()
            if config.build_workers == 1:
                problems = [
                    _scenario_problem(
                        data,
                        market,
                        config,
                        correlation_device,
                    )
                    for data in scenario_data
                ]
            else:
                with ThreadPoolExecutor(max_workers=config.build_workers) as executor:
                    problems = list(
                        executor.map(
                            lambda data: _scenario_problem(
                                data,
                                market,
                                config,
                                correlation_device,
                            ),
                            scenario_data,
                        )
                    )
            batch_build_seconds = time.perf_counter() - batch_build_started
            build_seconds += batch_build_seconds
            seeds = [
                (
                    config.seed
                    + int(pd.Timestamp(date).strftime("%Y%m%d")) * 1_000_003
                    + problem.scenario_index * 97_409
                )
                % (2**63)
                for problem in problems
            ]
            if solver is None:  # pragma: no cover - protected by run_qubo
                raise RuntimeError("QUBO method requires a solver")
            solved: list[SBMSolveResult] = solver.solve_batch(
                [problem.bqm for problem in problems],
                config.sbm,
                seeds,
                [problem.group_offsets for problem in problems],
                config.decode_sweeps,
            )
            batch_solve_seconds = sum(result.solve_seconds for result in solved)
            solve_seconds += batch_solve_seconds
            decode_started = time.perf_counter()
            for problem, result in zip(problems, solved):
                decoded, scenario_pnl, plausible_score = decode_hard_plausible(
                    problem, result.sample
                )
                decoded_energy = problem.bqm.energy(decoded)
                candidate = (
                    scenario_pnl,
                    problem.scenario_index,
                    decoded_energy,
                    result.raw_energy,
                    result.energy,
                    result.raw_one_hot_violations,
                    result.raw_feasible_candidates,
                    result.candidate_count,
                    plausible_score,
                    problem.plausibility.threshold,
                )
                if best_qubo is None or candidate[0] < best_qubo[0]:
                    best_qubo = candidate
            batch_decode_seconds = time.perf_counter() - decode_started
            decode_seconds += batch_decode_seconds
            del scenario_data, problems, solved
            completed_scenarios += len(scenario_batch)
            progress_now = time.perf_counter()
            if config.progress_interval > 0 and (
                completed_scenarios == len(scenario_indices)
                or progress_now - last_progress_report >= config.progress_interval
            ):
                progress_elapsed = progress_now - scenario_progress_started
                progress_rate = completed_scenarios / max(progress_elapsed, 1e-12)
                progress_eta = (
                    len(scenario_indices) - completed_scenarios
                ) / progress_rate
                scenario_label = (
                    f"Scenario {scenario_batch[0] + 1}/{all_scenario_count}"
                    if len(scenario_batch) == 1
                    else f"Batch {batch_position}/{scenario_batch_count} "
                    f"({len(scenario_batch)} scenarios)"
                )
                print(
                    f"{progress_context} {scenario_label} | "
                    f"data={batch_scenario_seconds:.2f}s | "
                    f"QUBO={batch_build_seconds:.2f}s | solve={batch_solve_seconds:.2f}s | "
                    f"decode={batch_decode_seconds:.2f}s | "
                    f"progress={completed_scenarios}/{len(scenario_indices)} "
                    f"({100.0 * completed_scenarios / len(scenario_indices):.1f}%) | "
                    f"ETA={progress_eta:.1f}s",
                    flush=True,
                )
                last_progress_report = progress_now

        if run_qubo and best_qubo is None:  # pragma: no cover
            raise RuntimeError("no QUBO scenario was solved")
        if run_baseline and best_baseline is None:  # pragma: no cover
            raise RuntimeError("no baseline scenario was evaluated")
        realized_vector = market.simple_returns.loc[date].to_numpy(dtype=float)
        realized_pnl = float(np.dot(market.weights, realized_vector))
        resource_values = resources.stop()
        actual_day_seconds = time.perf_counter() - day_started

        method_results: list[tuple[
            str,
            float,
            int,
            float | None,
            float | None,
            float | None,
            int | None,
            int | None,
            int | None,
            float | None,
            float | None,
        ]] = []
        if best_qubo is not None:
            method_results.append(
                ("qubo", *best_qubo)
            )
        if best_baseline is not None:
            method_results.append(
                (
                    "baseline",
                    best_baseline[0],
                    best_baseline[1],
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )

        for (
            method,
            worst_pnl,
            selected_scenario,
            selected_energy,
            raw_energy,
            repaired_energy,
            raw_violations,
            raw_feasible,
            solver_candidates,
            plausible_score,
            plausible_threshold,
        ) in method_results:
            if (date_text, method) in completed:
                continue
            margin = max(0.0, -worst_pnl)
            signed_error = (margin + realized_pnl) / market.gross_exposure
            method_day_seconds = (
                pca_seconds
                + scenario_build_seconds
                + (
                    build_seconds + solve_seconds + decode_seconds
                    if method == "qubo"
                    else baseline_seconds
                )
            )
            row: dict[str, Any] = {
                "date": date_text,
                "method": method,
                "calibration_start": prior.index[0].date().isoformat(),
                "calibration_end": prior.index[-1].date().isoformat(),
                "calibration_returns": len(prior),
                "assets": len(market.tickers),
                "scenarios": len(scenario_indices),
                "selected_scenario": selected_scenario,
                "selected_energy": selected_energy,
                "raw_energy": raw_energy,
                "repaired_energy": repaired_energy,
                "raw_one_hot_violations": raw_violations,
                "raw_feasible_candidates": raw_feasible,
                "solver_candidates": solver_candidates,
                "plausibility_score": plausible_score,
                "plausibility_threshold": plausible_threshold,
                "hard_plausibility_feasible": (
                    plausible_score <= plausible_threshold
                    if plausible_score is not None and plausible_threshold is not None
                    else None
                ),
                "margin": margin,
                "worst_scenario_pnl": worst_pnl,
                "realized_pnl": realized_pnl,
                "realized_loss": -realized_pnl,
                "gross_exposure": market.gross_exposure,
                "signed_margin_error": signed_error,
                "pca_seconds": pca_seconds,
                "scenario_build_seconds": scenario_build_seconds,
                "qubo_build_seconds": build_seconds if method == "qubo" else 0.0,
                "solve_seconds": solve_seconds if method == "qubo" else 0.0,
                "baseline_seconds": baseline_seconds if method == "baseline" else 0.0,
                "day_seconds": method_day_seconds,
                "device": resolved_device,
                "steps": config.sbm.steps if method == "qubo" else 0,
                "runs": config.sbm.runs if method == "qubo" else 0,
                "seed": config.seed,
            }
            _append_result(config.output, row)
            output_rows.append(row)
            print(
                f"{progress_context} RESULT {method.upper()} | margin={margin:.8g} | "
                f"realized P&L={realized_pnl:.8g} | "
                f"realized loss={-realized_pnl:.8g} | SME={signed_error:.8g} | "
                f"selected scenario={selected_scenario} | time={method_day_seconds:.2f}s",
                flush=True,
            )

        if config.progress_interval > 0:
            print("-" * 88, flush=True)
            print(
                f"{progress_context} DAY COMPLETE | PCA={pca_seconds:.2f}s | "
                f"scenario data={scenario_build_seconds:.2f}s | "
                f"QUBO build={build_seconds:.2f}s | solve={solve_seconds:.2f}s | "
                f"baseline={baseline_seconds:.4f}s | total={actual_day_seconds:.2f}s",
                flush=True,
            )
            print("=" * 88, flush=True)

        effective_run_batch_size = (
            min(config.sbm.run_batch_size or config.sbm.runs, config.sbm.runs)
            if run_qubo
            else 0
        )
        resource_row: dict[str, Any] = {
            "date": date_text,
            "method": config.method,
            "device": resolved_device,
            "correlation_device": correlation_device,
            "pca_dtype": config.pca_dtype,
            "sbm_dtype": config.sbm.dtype,
            "assets": len(market.tickers),
            "scenarios": len(scenario_indices),
            "scenario_batch_size": config.scenario_batch_size,
            "scenario_batches": scenario_batch_count,
            "build_workers": config.build_workers,
            "steps": config.sbm.steps if run_qubo else 0,
            "runs": config.sbm.runs if run_qubo else 0,
            "run_batch_size": effective_run_batch_size,
            "solver_run_batches_per_scenario_batch": (
                math.ceil(config.sbm.runs / effective_run_batch_size)
                if run_qubo
                else 0
            ),
            "pca_seconds": pca_seconds,
            "scenario_build_seconds": scenario_build_seconds,
            "qubo_build_seconds": build_seconds,
            "solve_seconds": solve_seconds,
            "baseline_seconds": baseline_seconds,
            "day_seconds": actual_day_seconds,
            **resource_values,
        }
        _append_csv(_diagnostics_path(config.output), resource_row, RESOURCE_COLUMNS)
    if config.output.exists() and config.output.stat().st_size:
        summary_path = write_error_summary(config.output)
        print(f"Summary: {summary_path}", flush=True)
    return pd.DataFrame(output_rows, columns=RESULT_COLUMNS)


def _device_worker(config: BacktestConfig) -> str:
    run_backtest(config)
    return str(config.output)


def run_backtest_multi_device(
    config: BacktestConfig, devices: Sequence[str]
) -> pd.DataFrame:
    """Shard independent dates across GPU processes and merge checkpoints."""

    normalized = tuple(resolve_torch_device(device.strip()) for device in devices)
    if len(normalized) < 2:
        return run_backtest(replace(config, device=normalized[0]))
    if len(set(normalized)) != len(normalized):
        raise ValueError("multi-device execution requires distinct device names")

    market = load_market_data(config)
    requested_methods = _requested_methods(config.method)
    completed = _existing_method_dates(config.output) if config.resume else set()
    pending = [
        pd.Timestamp(date).date().isoformat()
        for date in market.evaluation_dates
        if not all(
            (pd.Timestamp(date).date().isoformat(), method) in completed
            for method in requested_methods
        )
    ]
    if not pending:
        if config.output.exists():
            write_error_summary(config.output)
        return pd.DataFrame(columns=RESULT_COLUMNS)

    shards = [pending[index:: len(normalized)] for index in range(len(normalized))]
    worker_configs: list[BacktestConfig] = []
    for index, (device, dates) in enumerate(zip(normalized, shards)):
        if not dates:
            continue
        part = config.output.with_suffix(f".device{index}.part.csv")
        for generated in (
            part,
            part.with_suffix(".summary.json"),
            _diagnostics_path(part),
        ):
            if generated.exists():
                generated.unlink()
        worker_configs.append(
            replace(
                config,
                output=part,
                device=device,
                correlation_device=(
                    device
                    if config.correlation_device in {"auto", "cuda", "gpu"}
                    else config.correlation_device
                ),
                resume=False,
                day_limit=None,
                evaluation_dates=tuple(dates),
            )
        )

    # CUDA runtimes cannot safely inherit a process created with POSIX fork.
    # Spawn also gives every GPU worker an isolated Torch/CUDA context.
    with ProcessPoolExecutor(
        max_workers=len(worker_configs), mp_context=mp.get_context("spawn")
    ) as executor:
        part_paths = list(executor.map(_device_worker, worker_configs))

    frames: list[pd.DataFrame] = []
    if config.resume and config.output.exists():
        existing = pd.read_csv(config.output)
        if "method" not in existing:
            existing.insert(1, "method", "qubo")
        frames.append(existing)
    frames.extend(pd.read_csv(path) for path in part_paths)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "method"], keep="last")
    combined = combined.sort_values(["date", "method"], kind="stable")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output.with_suffix(config.output.suffix + ".tmp")
    combined.to_csv(temporary, index=False, columns=RESULT_COLUMNS)
    temporary.replace(config.output)
    write_error_summary(config.output)
    diagnostic_frames: list[pd.DataFrame] = []
    final_diagnostics = _diagnostics_path(config.output)
    if config.resume and final_diagnostics.exists():
        diagnostic_frames.append(pd.read_csv(final_diagnostics))
    diagnostic_frames.extend(
        pd.read_csv(_diagnostics_path(Path(path)))
        for path in part_paths
        if _diagnostics_path(Path(path)).exists()
    )
    if diagnostic_frames:
        combined_diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
        combined_diagnostics = combined_diagnostics.drop_duplicates(
            subset=["date"], keep="last"
        ).sort_values("date", kind="stable")
        diagnostic_temporary = final_diagnostics.with_suffix(
            final_diagnostics.suffix + ".tmp"
        )
        combined_diagnostics.to_csv(
            diagnostic_temporary, index=False, columns=RESOURCE_COLUMNS
        )
        diagnostic_temporary.replace(final_diagnostics)
    for part_path in map(Path, part_paths):
        part_path.unlink(missing_ok=True)
        part_path.with_suffix(".summary.json").unlink(missing_ok=True)
        _diagnostics_path(part_path).unlink(missing_ok=True)
    return combined.loc[combined["date"].isin(pending), list(RESULT_COLUMNS)]


def _grid_points(value: str) -> tuple[int, int]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid points must be comma-separated integers") from exc
    if len(parsed) != 2:
        raise argparse.ArgumentTypeError("exactly two PCA grid sizes are required")
    return parsed  # type: ignore[return-value]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rolling QUBO and/or greedy-baseline margin backtesting."
    )
    parser.add_argument("subfolder", help="Market universe, e.g. assets_00100")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backtest-start", type=pd.Timestamp)
    parser.add_argument("--backtest-end", type=pd.Timestamp)
    parser.add_argument("--window", type=int, default=125)
    parser.add_argument("--ew-lambda", type=float, default=0.93)
    parser.add_argument("--grid-points", type=_grid_points, default=DEFAULT_GRID_POINTS[:2])
    parser.add_argument("--scenario-index", type=int, action="append")
    parser.add_argument(
        "--method",
        choices=("qubo", "baseline", "both"),
        default="qubo",
        help="Margin method to evaluate (default: qubo).",
    )
    parser.add_argument("--z-bins", type=int, default=21)
    parser.add_argument("--nearest", type=int, default=100)
    parser.add_argument("--residual-sigma-range", type=float, default=5.0)
    parser.add_argument("--distance-inflation-alpha", type=float, default=0.50)
    parser.add_argument("--distance-inflation-power", type=float, default=2.0)
    parser.add_argument("--max-inflation-factor", type=float, default=5.0)
    parser.add_argument("--lambda-one-hot", type=float, default=1.0)
    parser.add_argument("--lambda-compat", type=float, default=0.1)
    parser.add_argument("--top-k-neighbors", type=int, default=5)
    parser.add_argument("--plausibility-confidence", type=float, default=0.998)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--devices",
        help="Comma-separated distinct devices for date sharding, e.g. cuda:0,cuda:1",
    )
    parser.add_argument("--pca-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--correlation-device", default="auto")
    parser.add_argument("--correlation-block-mib", type=int, default=256)
    parser.add_argument("--scenario-batch-size", type=int, default=1)
    parser.add_argument("--build-workers", type=int, default=1)
    parser.add_argument("--decode-sweeps", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--a0", type=float, default=1.0)
    parser.add_argument("--c0", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--initial-scale", type=float, default=0.05)
    parser.add_argument("--sbm-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--run-batch-size", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--day-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Minimum seconds between scenario progress lines; 0 disables them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    folder = resolve_market_folder(args.subfolder)
    output = args.output or (
        PROJECT_ROOT / "backtest-results" / f"{folder.name}_margin_backtest.csv"
    )
    config = BacktestConfig(
        subfolder=folder,
        output=output.resolve(),
        backtest_start=args.backtest_start,
        backtest_end=args.backtest_end,
        window=args.window,
        ew_lambda=args.ew_lambda,
        grid_points=args.grid_points,
        scenario_indices=(
            tuple(args.scenario_index) if args.scenario_index is not None else None
        ),
        method=args.method,
        z_bins=args.z_bins,
        nearest=args.nearest,
        residual_sigma_range=args.residual_sigma_range,
        distance_inflation_alpha=args.distance_inflation_alpha,
        distance_inflation_power=args.distance_inflation_power,
        max_inflation_factor=args.max_inflation_factor,
        lambda_one_hot=args.lambda_one_hot,
        lambda_compat=args.lambda_compat,
        top_k_neighbors=args.top_k_neighbors,
        plausibility_confidence=args.plausibility_confidence,
        device=args.device,
        pca_dtype=args.pca_dtype,
        correlation_device=args.correlation_device,
        correlation_block_mib=args.correlation_block_mib,
        scenario_batch_size=args.scenario_batch_size,
        build_workers=args.build_workers,
        decode_sweeps=args.decode_sweeps,
        seed=args.seed,
        day_limit=args.day_limit,
        resume=args.resume,
        progress_interval=args.progress_interval,
        sbm=SBMConfig(
            steps=args.steps,
            runs=args.runs,
            dt=args.dt,
            a0=args.a0,
            c0=args.c0,
            gamma=args.gamma,
            initial_scale=args.initial_scale,
            dtype=args.sbm_dtype,
            run_batch_size=args.run_batch_size,
        ),
    )
    devices = (
        tuple(item.strip() for item in args.devices.split(",") if item.strip())
        if args.devices
        else (config.device,)
    )
    resolved_devices = tuple(resolve_torch_device(device) for device in devices)
    print(
        f"Backtesting {folder.name} on {', '.join(resolved_devices)}; "
        f"method={config.method}; output={config.output}",
        flush=True,
    )
    if len(resolved_devices) == 1:
        run_backtest(replace(config, device=resolved_devices[0]))
    else:
        run_backtest_multi_device(config, resolved_devices)


if __name__ == "__main__":
    main()
