#!/usr/bin/env python3
"""Rolling, leakage-safe scenario-QUBO margin backtester.

The orchestrator keeps BQMs in memory, batches independent scenario graphs for
simulated bifurcation, and writes one compact record per backtesting day.  CUDA
is selected automatically when a CUDA-enabled PyTorch installation and NVIDIA
device are available; otherwise the identical sparse dynamics run on CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import dimod
import numpy as np
import pandas as pd
from scipy import sparse

from market_to_qubo import (
    DEFAULT_GRID_POINTS,
    PCAResult,
    PreparedData,
    build_qubos,
    build_scenario_grid,
    conditional_neighbors,
    convert_z_to_returns,
    create_z_shock_grids,
    prepare_shock_grid_context,
    read_close_prices,
    read_portfolio,
)
from qubo_model import CompactQubo
from sbm_torch import SBMConfig, SBMSolveResult, TorchBatchSBMSolver, resolve_torch_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_MARKET_ROOT = PROJECT_ROOT / "synthetic_market"
RESULT_COLUMNS = (
    "date",
    "calibration_start",
    "calibration_end",
    "calibration_returns",
    "assets",
    "scenarios",
    "selected_scenario",
    "selected_energy",
    "raw_one_hot_violations",
    "margin",
    "worst_scenario_pnl",
    "realized_pnl",
    "realized_loss",
    "gross_exposure",
    "signed_margin_error",
    "pca_seconds",
    "qubo_build_seconds",
    "solve_seconds",
    "day_seconds",
    "device",
    "steps",
    "runs",
    "seed",
)


@dataclass(frozen=True)
class RollingScaler:
    mean_: np.ndarray
    scale_: np.ndarray


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
    z_bins: int = 21
    nearest: int = 100
    residual_sigma_range: float = 5.0
    distance_inflation_alpha: float = 0.50
    distance_inflation_power: float = 2.0
    max_inflation_factor: float = 5.0
    lambda_one_hot: float = 1.0
    lambda_compat: float = 0.1
    top_k_neighbors: int = 5
    device: str = "auto"
    pca_dtype: str = "float64"
    correlation_device: str = "auto"
    correlation_block_mib: int = 256
    scenario_batch_size: int = 1
    build_workers: int = 1
    decode_sweeps: int = 1
    seed: int = 1
    day_limit: int | None = None
    resume: bool = False
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
        if self.scenario_indices is not None and not self.scenario_indices:
            raise ValueError("scenario_indices cannot be empty")
        self.sbm.validate()


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
    """Fit a two-component EW PCA using only the supplied rolling window."""

    import torch

    values = log_returns.to_numpy(dtype=np.float64, copy=False)
    observations, assets = values.shape
    if observations < 2 or assets < 2:
        raise ValueError("rolling PCA requires at least two returns and two assets")
    device = torch.device(device_name)
    dtype = torch.float32 if dtype_name == "float32" else torch.float64
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    weights = decay ** torch.arange(
        observations - 1, -1, -1, dtype=dtype, device=device
    )
    weights /= weights.sum()
    mean = torch.sum(weights[:, None] * tensor, dim=0)
    centered_returns = tensor - mean
    variance = torch.sum(weights[:, None] * centered_returns.square(), dim=0)
    scale = torch.sqrt(torch.clamp(variance, min=torch.finfo(dtype).eps))
    z = centered_returns / scale
    z_mean = torch.sum(weights[:, None] * z, dim=0)
    centered_z = z - z_mean
    weighted_z = torch.sqrt(weights[:, None]) * centered_z

    if assets <= observations:
        covariance = weighted_z.T @ weighted_z
        all_eigenvalues, all_vectors = torch.linalg.eigh(covariance)
        selected = torch.arange(
            len(all_eigenvalues) - 1,
            len(all_eigenvalues) - 3,
            -1,
            device=device,
        )
        eigenvalues = torch.clamp(all_eigenvalues[selected], min=0.0)
        loadings = all_vectors[:, selected].T
        solver_name = f"{device.type} asset-space symmetric eigendecomposition"
    else:
        gram = weighted_z @ weighted_z.T
        all_eigenvalues, all_vectors = torch.linalg.eigh(gram)
        selected = torch.arange(
            len(all_eigenvalues) - 1,
            len(all_eigenvalues) - 3,
            -1,
            device=device,
        )
        eigenvalues = torch.clamp(all_eigenvalues[selected], min=0.0)
        if bool(torch.any(eigenvalues <= torch.finfo(dtype).eps)):
            raise ValueError("rolling PCA contains a zero-variance selected component")
        left = all_vectors[:, selected]
        loadings = (weighted_z.T @ left / torch.sqrt(eigenvalues)[None, :]).T
        solver_name = f"{device.type} observation-space dual eigendecomposition"

    # Fix the arbitrary eigenvector sign for reproducible scenario numbering.
    for component in range(loadings.shape[0]):
        pivot = int(torch.argmax(torch.abs(loadings[component])).item())
        if float(loadings[component, pivot]) < 0.0:
            loadings[component].neg_()
    factors = centered_z @ loadings.T
    total_variance = torch.clamp(all_eigenvalues, min=0.0).sum()
    if float(total_variance) <= 0.0:
        raise ValueError("rolling PCA has no positive variance")

    mean_np = mean.cpu().numpy().astype(np.float64, copy=False)
    scale_np = scale.cpu().numpy().astype(np.float64, copy=False)
    z_np = z.cpu().numpy().astype(np.float64, copy=False)
    factors_np = factors.cpu().numpy().astype(np.float64, copy=False)
    loadings_np = loadings.cpu().numpy().astype(np.float64, copy=False)
    eigenvalues_np = eigenvalues.cpu().numpy().astype(np.float64, copy=False)
    weights_np = weights.cpu().numpy().astype(np.float64, copy=False)
    scaler = RollingScaler(mean_np, scale_np)
    empty = np.empty((0, assets), dtype=np.float64)
    prepared = PreparedData(
        tickers=list(log_returns.columns),
        historical_close=pd.DataFrame(),
        backtest_close=pd.DataFrame(),
        historical_returns=np.expm1(log_returns),
        historical_log_returns=log_returns,
        backtest_returns=pd.DataFrame(columns=log_returns.columns),
        backtest_log_returns=pd.DataFrame(columns=log_returns.columns),
        historical_z=z_np,
        backtest_z=empty,
        standardizer=scaler,  # type: ignore[arg-type]
    )
    pca = PCAResult(
        factors=factors_np,
        backtest_factors=np.empty((0, 2), dtype=np.float64),
        loadings=loadings_np,
        eigenvalues=eigenvalues_np,
        explained=eigenvalues_np / float(total_variance.cpu()),
        weighted_mean=z_mean.cpu().numpy().astype(np.float64, copy=False),
        weights=weights_np,
        solver=solver_name,
    )
    return prepared, pca


def conditional_neighbors_torch(
    residual_samples: np.ndarray,
    top_k: int,
    device_name: str,
    block_mib: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find exact top correlations in bounded GPU/CPU row blocks."""

    if device_name == "cpu":
        return conditional_neighbors(residual_samples, top_k)
    import torch

    observations, assets = residual_samples.shape
    if observations < 2:
        raise ValueError("at least two residual samples are required")
    neighbor_count = min(top_k, max(assets - 1, 0))
    if neighbor_count == 0:
        return (
            np.std(residual_samples, axis=0, ddof=1),
            np.empty((assets, 0), dtype=np.int64),
            np.empty((assets, 0), dtype=float),
        )
    device = torch.device(device_name)
    tensor = torch.as_tensor(residual_samples, dtype=torch.float32, device=device)
    centered = tensor - tensor.mean(dim=0)
    variance = centered.square().sum(dim=0) / (observations - 1)
    standard_deviation = torch.sqrt(torch.clamp(variance, min=1e-12))
    normalized = centered / standard_deviation
    target_bytes = block_mib * 1024 * 1024
    rows_per_block = max(1, min(assets, target_bytes // max(4 * assets, 1)))
    indices = np.empty((assets, neighbor_count), dtype=np.int64)
    correlations_out = np.empty((assets, neighbor_count), dtype=np.float64)
    for start in range(0, assets, rows_per_block):
        stop = min(start + rows_per_block, assets)
        correlations = normalized[:, start:stop].T @ normalized
        correlations /= observations - 1
        correlations.nan_to_num_(0.0, posinf=0.0, neginf=0.0)
        correlations.clamp_(-0.999, 0.999)
        absolute = correlations.abs()
        local = torch.arange(stop - start, device=device)
        absolute[local, torch.arange(start, stop, device=device)] = -torch.inf
        _, chosen = torch.topk(
            absolute, k=neighbor_count, dim=1, largest=True, sorted=True
        )
        chosen_correlations = torch.gather(correlations, 1, chosen)
        indices[start:stop] = chosen.cpu().numpy()
        correlations_out[start:stop] = chosen_correlations.cpu().numpy()
    return standard_deviation.cpu().numpy(), indices, correlations_out


def _scenario_problem(
    scenario_index: int,
    prepared: PreparedData,
    pca: PCAResult,
    scenario_grid: np.ndarray,
    shock_context: Any,
    market: MarketData,
    config: BacktestConfig,
    correlation_device: str,
) -> ScenarioProblem:
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
    conditional_std, neighbor_indices, neighbor_correlations = (
        conditional_neighbors_torch(
            shocks.inflated_residuals,
            config.top_k_neighbors,
            correlation_device,
            config.correlation_block_mib,
        )
    )
    bqm, _, _ = build_qubos(
        asset_grids,
        market.weights,
        shocks.z_hat,
        conditional_std,
        neighbor_indices,
        neighbor_correlations,
        config.lambda_one_hot,
        config.lambda_compat,
        numeric_labels=True,
        compact=True,
    )
    if not isinstance(bqm, CompactQubo):  # pragma: no cover - fixed call contract
        raise TypeError("rolling backtests require a compact QUBO")
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
    return ScenarioProblem(
        scenario_index=scenario_index,
        bqm=bqm,
        group_offsets=group_offsets,
        portfolio_linear=portfolio_linear,
    )


def repair_one_hot(
    bqm: dimod.BinaryQuadraticModel | CompactQubo,
    raw_sample: np.ndarray,
    group_offsets: np.ndarray,
    sweeps: int,
) -> tuple[np.ndarray, int, float]:
    """Project to exactly one state per asset and run categorical descent."""

    if isinstance(bqm, CompactQubo):
        labels = None
        linear = bqm.linear
        row = bqm.heads
        col = bqm.tails
        bias = bqm.quadratic
    else:
        labels = tuple(bqm.variables)
        vectors = bqm.to_numpy_vectors(
            variable_order=labels,
            sort_indices=True,
            sort_labels=False,
        )
        linear = np.asarray(vectors.linear_biases, dtype=np.float64)
        row = np.asarray(vectors.quadratic.row_indices, dtype=np.int64)
        col = np.asarray(vectors.quadratic.col_indices, dtype=np.int64)
        bias = np.asarray(vectors.quadratic.biases, dtype=np.float64)
    n = len(linear)
    adjacency = sparse.csr_matrix(
        (
            np.concatenate((bias, bias)),
            (np.concatenate((row, col)), np.concatenate((col, row))),
        ),
        shape=(n, n),
    )
    sample = np.asarray(raw_sample, dtype=np.uint8).copy()
    counts = np.add.reduceat(sample, group_offsets[:-1])
    violations = int(np.count_nonzero(counts != 1))
    local_fields = linear + adjacency @ sample.astype(float)

    def flip(variable: int) -> None:
        change = -1.0 if sample[variable] else 1.0
        sample[variable] ^= np.uint8(1)
        start = adjacency.indptr[variable]
        stop = adjacency.indptr[variable + 1]
        local_fields[adjacency.indices[start:stop]] += (
            adjacency.data[start:stop] * change
        )

    for group in range(len(group_offsets) - 1):
        begin = int(group_offsets[group])
        end = int(group_offsets[group + 1])
        active = np.flatnonzero(sample[begin:end]) + begin
        if len(active) == 1:
            continue
        for variable in active:
            flip(int(variable))
        flip(begin + int(np.argmin(local_fields[begin:end])))

    for _ in range(sweeps):
        changed = False
        for group in range(len(group_offsets) - 1):
            begin = int(group_offsets[group])
            end = int(group_offsets[group + 1])
            current = begin + int(np.flatnonzero(sample[begin:end])[0])
            flip(current)
            best = begin + int(np.argmin(local_fields[begin:end]))
            flip(best)
            changed |= best != current
        if not changed:
            break

    if isinstance(bqm, CompactQubo):
        energy = bqm.energy(sample)
    else:
        energy = float(bqm.energies((sample.reshape(1, -1), labels))[0])
    return sample, violations, energy


def _scenario_batches(indices: Sequence[int], size: int) -> list[list[int]]:
    return [list(indices[start : start + size]) for start in range(0, len(indices), size)]


def _existing_dates(output: Path) -> set[str]:
    if not output.exists():
        return set()
    frame = pd.read_csv(output, usecols=["date"], dtype=str)
    return set(frame["date"])


def _append_result(output: Path, row: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists() and output.stat().st_size > 0
    with output.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        destination.flush()


def write_error_summary(output: Path) -> Path:
    """Write distribution statistics over all checkpointed daily errors."""

    frame = pd.read_csv(output)
    if frame.empty:
        raise ValueError("cannot summarize an empty backtest result")
    errors = frame["signed_margin_error"].to_numpy(dtype=float)
    shortfalls = errors[errors < 0.0]
    quantiles = np.quantile(errors, [0.01, 0.05, 0.50, 0.95, 0.99])
    summary = {
        "days": int(len(errors)),
        "mean_signed_margin_error": float(errors.mean()),
        "std_signed_margin_error": float(errors.std(ddof=1)) if len(errors) > 1 else 0.0,
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
        "total_runtime_seconds": float(frame["day_seconds"].sum()),
    }
    summary_path = output.with_suffix(".summary.json")
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return summary_path


def run_backtest(config: BacktestConfig) -> pd.DataFrame:
    config.validate()
    if not config.resume:
        for generated in (config.output, config.output.with_suffix(".summary.json")):
            if generated.exists():
                generated.unlink()
    solver = TorchBatchSBMSolver(config.device)
    pca_device = resolve_torch_device(config.device)
    correlation_requested = (
        solver.device if config.correlation_device == "auto" else config.correlation_device
    )
    correlation_device = resolve_torch_device(correlation_requested)
    if correlation_device != "cpu" and config.build_workers != 1:
        raise ValueError("GPU correlation construction requires build_workers=1")
    market = load_market_data(config)
    completed = _existing_dates(config.output) if config.resume else set()
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

    for day_position, date in enumerate(market.evaluation_dates):
        date_text = pd.Timestamp(date).date().isoformat()
        if date_text in completed:
            print(f"[{day_position + 1}/{len(market.evaluation_dates)}] {date_text}: resumed")
            continue
        day_started = time.perf_counter()
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

        best: tuple[float, int, float, int] | None = None
        build_seconds = 0.0
        solve_seconds = 0.0
        for scenario_batch in _scenario_batches(
            scenario_indices, config.scenario_batch_size
        ):
            batch_build_started = time.perf_counter()
            if config.build_workers == 1:
                problems = [
                    _scenario_problem(
                        scenario_index,
                        prepared,
                        pca,
                        scenario_grid,
                        shock_context,
                        market,
                        config,
                        correlation_device,
                    )
                    for scenario_index in scenario_batch
                ]
            else:
                with ThreadPoolExecutor(max_workers=config.build_workers) as executor:
                    problems = list(
                        executor.map(
                            lambda scenario_index: _scenario_problem(
                                scenario_index,
                                prepared,
                                pca,
                                scenario_grid,
                                shock_context,
                                market,
                                config,
                                correlation_device,
                            ),
                            scenario_batch,
                        )
                    )
            build_seconds += time.perf_counter() - batch_build_started
            seeds = [
                (
                    config.seed
                    + int(pd.Timestamp(date).strftime("%Y%m%d")) * 1_000_003
                    + problem.scenario_index * 97_409
                )
                % (2**63)
                for problem in problems
            ]
            solved: list[SBMSolveResult] = solver.solve_batch(
                [problem.bqm for problem in problems], config.sbm, seeds
            )
            solve_seconds += sum(result.solve_seconds for result in solved)
            for problem, result in zip(problems, solved):
                decoded, violations, decoded_energy = repair_one_hot(
                    problem.bqm,
                    result.sample,
                    problem.group_offsets,
                    config.decode_sweeps,
                )
                scenario_pnl = float(np.dot(problem.portfolio_linear, decoded))
                candidate = (
                    scenario_pnl,
                    problem.scenario_index,
                    decoded_energy,
                    violations,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
            del problems, solved

        if best is None:  # pragma: no cover - scenario validation prevents this
            raise RuntimeError("no scenario was solved")
        worst_pnl, selected_scenario, selected_energy, raw_violations = best
        realized_vector = market.simple_returns.loc[date].to_numpy(dtype=float)
        realized_pnl = float(np.dot(market.weights, realized_vector))
        margin = max(0.0, -worst_pnl)
        signed_error = (margin + realized_pnl) / market.gross_exposure
        row: dict[str, Any] = {
            "date": date_text,
            "calibration_start": prior.index[0].date().isoformat(),
            "calibration_end": prior.index[-1].date().isoformat(),
            "calibration_returns": len(prior),
            "assets": len(market.tickers),
            "scenarios": len(scenario_indices),
            "selected_scenario": selected_scenario,
            "selected_energy": selected_energy,
            "raw_one_hot_violations": raw_violations,
            "margin": margin,
            "worst_scenario_pnl": worst_pnl,
            "realized_pnl": realized_pnl,
            "realized_loss": -realized_pnl,
            "gross_exposure": market.gross_exposure,
            "signed_margin_error": signed_error,
            "pca_seconds": pca_seconds,
            "qubo_build_seconds": build_seconds,
            "solve_seconds": solve_seconds,
            "day_seconds": time.perf_counter() - day_started,
            "device": solver.device,
            "steps": config.sbm.steps,
            "runs": config.sbm.runs,
            "seed": config.seed,
        }
        _append_result(config.output, row)
        output_rows.append(row)
        print(
            f"[{day_position + 1}/{len(market.evaluation_dates)}] {date_text}: "
            f"margin={margin:.8g} realized={realized_pnl:.8g} "
            f"error={signed_error:.8g} scenario={selected_scenario} "
            f"time={row['day_seconds']:.2f}s",
            flush=True,
        )
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
    completed = _existing_dates(config.output) if config.resume else set()
    pending = [
        pd.Timestamp(date).date().isoformat()
        for date in market.evaluation_dates
        if pd.Timestamp(date).date().isoformat() not in completed
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
        for generated in (part, part.with_suffix(".summary.json")):
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
        frames.append(pd.read_csv(config.output))
    frames.extend(pd.read_csv(path) for path in part_paths)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date", kind="stable")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output.with_suffix(config.output.suffix + ".tmp")
    combined.to_csv(temporary, index=False, columns=RESULT_COLUMNS)
    temporary.replace(config.output)
    write_error_summary(config.output)
    for part_path in map(Path, part_paths):
        part_path.unlink(missing_ok=True)
        part_path.with_suffix(".summary.json").unlink(missing_ok=True)
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
        description="Run rolling scenario-QUBO margin backtesting with batched SBM."
    )
    parser.add_argument("subfolder", help="Market universe, e.g. assets_00100")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backtest-start", type=pd.Timestamp)
    parser.add_argument("--backtest-end", type=pd.Timestamp)
    parser.add_argument("--window", type=int, default=125)
    parser.add_argument("--ew-lambda", type=float, default=0.93)
    parser.add_argument("--grid-points", type=_grid_points, default=DEFAULT_GRID_POINTS[:2])
    parser.add_argument("--scenario-index", type=int, action="append")
    parser.add_argument("--z-bins", type=int, default=21)
    parser.add_argument("--nearest", type=int, default=100)
    parser.add_argument("--residual-sigma-range", type=float, default=5.0)
    parser.add_argument("--distance-inflation-alpha", type=float, default=0.50)
    parser.add_argument("--distance-inflation-power", type=float, default=2.0)
    parser.add_argument("--max-inflation-factor", type=float, default=5.0)
    parser.add_argument("--lambda-one-hot", type=float, default=1.0)
    parser.add_argument("--lambda-compat", type=float, default=0.1)
    parser.add_argument("--top-k-neighbors", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--devices",
        help="Comma-separated distinct devices for date sharding, e.g. cuda:0,cuda:1",
    )
    parser.add_argument("--pca-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--correlation-device", default="auto")
    parser.add_argument("--correlation-block-mib", type=int, default=256)
    parser.add_argument("--scenario-batch-size", type=int, default=1)
    parser.add_argument("--build-workers", type=int, default=1)
    parser.add_argument("--decode-sweeps", type=int, default=1)
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
        z_bins=args.z_bins,
        nearest=args.nearest,
        residual_sigma_range=args.residual_sigma_range,
        distance_inflation_alpha=args.distance_inflation_alpha,
        distance_inflation_power=args.distance_inflation_power,
        max_inflation_factor=args.max_inflation_factor,
        lambda_one_hot=args.lambda_one_hot,
        lambda_compat=args.lambda_compat,
        top_k_neighbors=args.top_k_neighbors,
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
        f"output={config.output}",
        flush=True,
    )
    if len(resolved_devices) == 1:
        run_backtest(replace(config, device=resolved_devices[0]))
    else:
        run_backtest_multi_device(config, resolved_devices)


if __name__ == "__main__":
    main()
