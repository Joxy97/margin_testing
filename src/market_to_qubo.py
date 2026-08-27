#!/usr/bin/env python3
"""Build (but do not solve) a scenario-conditioned portfolio QUBO.

The implementation follows the calculation cells in ``example_code.ipynb``.
It reads one subfolder of ``synthetic_market`` and writes a report plus one BQM
and CQM pair per requested scenario inside a stable project-root directory. For
example, ``qubo-assets_10`` contains:

* ``report.html`` - self-contained tables and plots
* ``bqms/scenario_0.bqm``, ... - Ocean BQM files
* ``cqms/scenario_0.cqm``, ... - matching Ocean CQM files
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import dimod
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_MARKET_ROOT = PROJECT_ROOT / "synthetic_market"
DEFAULT_GRID_POINTS = (21, 5, 5, 3)


@dataclass
class PreparedData:
    tickers: list[str]
    historical_close: pd.DataFrame
    backtest_close: pd.DataFrame
    historical_returns: pd.DataFrame
    historical_log_returns: pd.DataFrame
    backtest_returns: pd.DataFrame
    backtest_log_returns: pd.DataFrame
    historical_z: np.ndarray
    backtest_z: np.ndarray
    standardizer: StandardScaler


@dataclass
class PCAResult:
    factors: np.ndarray
    backtest_factors: np.ndarray
    loadings: np.ndarray
    eigenvalues: np.ndarray
    explained: np.ndarray
    weighted_mean: np.ndarray
    weights: np.ndarray
    solver: str


@dataclass
class ShockGridResult:
    selected_scenario: np.ndarray
    z_hat: np.ndarray
    residuals_recent: np.ndarray
    nearest_residuals: np.ndarray
    inflated_residuals: np.ndarray
    conditional_factors: np.ndarray
    factors_recent: np.ndarray
    scenario_distance: float
    inflation_factor: float
    synthetic_z_samples: np.ndarray
    asset_z_grids: list[np.ndarray]
    full_bin_edges: list[np.ndarray]
    kept_bin_edges: list[np.ndarray]
    max_vol_ranges: np.ndarray
    residual_ranges: np.ndarray


@dataclass(frozen=True)
class ShockGridContext:
    """Scenario-independent residual data reused across scenario exports."""

    factors_recent: np.ndarray
    residuals_recent: np.ndarray
    pc_scale: np.ndarray
    max_abs_z: np.ndarray
    fallback_sigma: np.ndarray


@dataclass(frozen=True)
class ModelStats:
    """Report-only model metadata that avoids retaining intermediate BQMs."""

    name: str
    variables: int
    interactions: int
    offset: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a scenario-conditioned portfolio QUBO from one "
            "synthetic_market subfolder. The QUBO is exported, never solved."
        )
    )
    parser.add_argument(
        "subfolder",
        help=(
            "Direct subfolder of synthetic_market, for example assets_00010. "
            "An absolute path to that subfolder is also accepted."
        ),
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        action="append",
        default=None,
        help=(
            "Export only this scenario index. Repeat the option to select "
            "multiple scenarios. By default every scenario is exported."
        ),
    )
    parser.add_argument("--components", type=int, default=2, metavar="K")
    parser.add_argument("--ew-lambda", type=float, default=0.93)
    parser.add_argument("--ew-window", type=int, default=125)
    parser.add_argument(
        "--grid-points",
        default=None,
        metavar="N1,N2,...",
        help=(
            "Odd grid sizes by PC. By default, use the first K values from "
            "the notebook list 11,5,5,3 (therefore 11,5 for K=2)."
        ),
    )
    parser.add_argument("--tail-density-gamma", type=float, default=1.0)
    parser.add_argument("--z-bins", type=int, default=21)
    parser.add_argument("--nearest", type=int, default=100)
    parser.add_argument("--residual-sigma-range", type=float, default=5.0)
    parser.add_argument("--distance-inflation-alpha", type=float, default=0.50)
    parser.add_argument("--distance-inflation-power", type=float, default=2.0)
    parser.add_argument("--max-inflation-factor", type=float, default=5.0)
    parser.add_argument("--lambda-one-hot", type=float, default=1.0)
    parser.add_argument("--lambda-compat", type=float, default=0.1)
    parser.add_argument("--top-k-neighbors", type=int, default=5)
    parser.add_argument("--visual-asset-index", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if args.components < 1:
        raise ValueError("--components must be at least 1")
    if not 0.0 < args.ew_lambda <= 1.0:
        raise ValueError("--ew-lambda must be in (0, 1]")
    if args.ew_window < 2:
        raise ValueError("--ew-window must be at least 2")
    if args.tail_density_gamma <= 0.0:
        raise ValueError("--tail-density-gamma must be positive")
    if args.z_bins < 1:
        raise ValueError("--z-bins must be positive")
    if args.nearest < 2:
        raise ValueError("--nearest must be at least 2")
    if args.residual_sigma_range <= 0.0:
        raise ValueError("--residual-sigma-range must be positive")
    if args.distance_inflation_alpha < 0.0:
        raise ValueError("--distance-inflation-alpha cannot be negative")
    if args.distance_inflation_power <= 0.0:
        raise ValueError("--distance-inflation-power must be positive")
    if args.max_inflation_factor < 1.0:
        raise ValueError("--max-inflation-factor must be at least 1")
    if args.lambda_one_hot < 0.0 or args.lambda_compat < 0.0:
        raise ValueError("QUBO penalty strengths cannot be negative")
    if args.top_k_neighbors < 0:
        raise ValueError("--top-k-neighbors cannot be negative")

    if args.grid_points is None:
        if args.components > len(DEFAULT_GRID_POINTS):
            raise ValueError(
                "--grid-points is required when --components is greater than "
                f"{len(DEFAULT_GRID_POINTS)}"
            )
        grid_points = DEFAULT_GRID_POINTS[: args.components]
    else:
        try:
            grid_points = tuple(
                int(value.strip()) for value in args.grid_points.split(",")
            )
        except ValueError as exc:
            raise ValueError("--grid-points must be comma-separated integers") from exc

    if len(grid_points) != args.components:
        raise ValueError(
            "--grid-points must contain exactly one value per PCA component"
        )
    if any(value < 1 or value % 2 == 0 for value in grid_points):
        raise ValueError("Every PC grid size must be a positive odd integer")
    return grid_points


def resolve_input_folder(value: str) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_absolute():
        folder = supplied.resolve()
    elif supplied.parts and supplied.parts[0].lower() == "synthetic_market":
        folder = (PROJECT_ROOT / supplied).resolve()
    else:
        folder = (SYNTHETIC_MARKET_ROOT / supplied).resolve()

    market_root = SYNTHETIC_MARKET_ROOT.resolve()
    try:
        relative = folder.relative_to(market_root)
    except ValueError as exc:
        raise ValueError(f"Input folder must be inside {market_root}") from exc
    if len(relative.parts) != 1:
        raise ValueError("Input must be a direct subfolder of synthetic_market")
    if not folder.is_dir():
        raise FileNotFoundError(f"Input subfolder does not exist: {folder}")
    return folder


def output_directory(folder: Path) -> Path:
    """Return the stable project-root output directory for an asset universe."""
    asset_match = re.fullmatch(r"assets_0*(\d+)", folder.name, flags=re.IGNORECASE)
    if asset_match:
        return PROJECT_ROOT / f"qubo-assets_{int(asset_match.group(1))}"

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", folder.name).strip("_")
    return PROJECT_ROOT / f"qubo-{safe_name or 'input'}"


def locate_file(folder: Path, candidates: Sequence[str], role: str) -> Path:
    for name in candidates:
        path = folder / name
        if path.is_file():
            return path
    expected = ", ".join(candidates)
    raise FileNotFoundError(f"No {role} CSV in {folder}; expected one of: {expected}")


def read_close_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(f"{path.name} must have a date column and asset columns")

    index_name = str(frame.columns[0])
    raw_index = frame.iloc[:, 0]
    parsed_index = pd.to_datetime(raw_index, errors="coerce")
    if parsed_index.isna().any():
        raise ValueError(f"{path.name}: column {index_name!r} contains invalid dates")

    prices = frame.iloc[:, 1:].copy()
    prices.columns = [str(column) for column in prices.columns]
    if len(set(prices.columns)) != len(prices.columns):
        raise ValueError(f"{path.name} has duplicate asset columns")
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices.index = pd.DatetimeIndex(parsed_index, name=index_name)
    prices = prices.ffill().dropna(axis=0, how="any")

    if len(prices) < 2:
        raise ValueError(f"{path.name} has fewer than two complete price rows")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(f"{path.name} prices must be finite and strictly positive")
    if not prices.index.is_monotonic_increasing:
        raise ValueError(f"{path.name} dates must be in increasing order")
    return prices


def calculate_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    simple = prices.pct_change(fill_method=None).ffill().dropna(axis=0, how="any")
    log_returns = np.log(prices / prices.shift(1)).ffill().dropna(axis=0, how="any")
    return simple, log_returns


def prepare_data(folder: Path) -> tuple[PreparedData, dict[str, Path]]:
    historical_path = locate_file(
        folder,
        ("historical_close.csv", "historical.csv"),
        "historical close-price",
    )
    backtest_path = locate_file(
        folder,
        ("backtest_close.csv", "backtesting_close.csv", "backtesting.csv"),
        "backtesting close-price",
    )
    portfolio_path = locate_file(folder, ("portfolio.csv",), "portfolio")

    historical_close = read_close_prices(historical_path)
    backtest_close = read_close_prices(backtest_path)
    if list(backtest_close.columns) != list(historical_close.columns):
        raise ValueError(
            "Historical and backtesting asset columns must match in name and order"
        )

    historical_returns, historical_log_returns = calculate_returns(historical_close)

    # Anchor the first backtesting return on the last historical close, retaining
    # every backtesting date without fitting anything on backtesting information.
    bridged = pd.concat([historical_close.iloc[[-1]], backtest_close], axis=0)
    backtest_returns, backtest_log_returns = calculate_returns(bridged)
    backtest_returns = backtest_returns.iloc[-len(backtest_close) :]
    backtest_log_returns = backtest_log_returns.iloc[-len(backtest_close) :]

    standardizer = StandardScaler()
    historical_z = standardizer.fit_transform(historical_log_returns)
    backtest_z = standardizer.transform(backtest_log_returns)

    if not np.isfinite(historical_z).all() or not np.isfinite(backtest_z).all():
        raise ValueError("Standardized returns contain non-finite values")

    prepared = PreparedData(
        tickers=list(historical_close.columns),
        historical_close=historical_close,
        backtest_close=backtest_close,
        historical_returns=historical_returns,
        historical_log_returns=historical_log_returns,
        backtest_returns=backtest_returns,
        backtest_log_returns=backtest_log_returns,
        historical_z=historical_z,
        backtest_z=backtest_z,
        standardizer=standardizer,
    )
    return prepared, {
        "historical": historical_path,
        "backtest": backtest_path,
        "portfolio": portfolio_path,
    }


def exponentially_weighted_pca(
    historical_z: np.ndarray,
    backtest_z: np.ndarray,
    components: int,
    decay: float,
    window: int,
) -> PCAResult:
    z_ew = historical_z[-window:]
    observations, assets = z_ew.shape
    if components > min(observations, assets):
        raise ValueError(
            f"--components={components} exceeds min(EW observations, assets)="
            f"{min(observations, assets)}"
        )

    weights = decay ** np.arange(observations - 1, -1, -1, dtype=float)
    weights /= weights.sum()
    weighted_mean = np.sum(weights[:, None] * z_ew, axis=0)
    centered = z_ew - weighted_mean

    # This is the notebook's eigendecomposition. For wide markets, use its
    # mathematically equivalent observation-space dual to avoid an N x N matrix.
    if assets <= 512:
        covariance = centered.T @ (weights[:, None] * centered)
        all_eigenvalues, all_eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(all_eigenvalues)[::-1]
        eigenvalues = np.maximum(all_eigenvalues[order[:components]], 0.0)
        loadings = all_eigenvectors[:, order[:components]].T
        total_variance = float(np.maximum(all_eigenvalues, 0.0).sum())
        solver = "asset-space symmetric eigendecomposition"
    else:
        weighted_centered = np.sqrt(weights[:, None]) * centered
        gram = weighted_centered @ weighted_centered.T
        all_eigenvalues, left_vectors = np.linalg.eigh(gram)
        order = np.argsort(all_eigenvalues)[::-1]
        eigenvalues = np.maximum(all_eigenvalues[order[:components]], 0.0)
        if np.any(eigenvalues <= np.finfo(float).eps):
            raise ValueError("Requested PCA components include a zero-variance mode")
        chosen_left = left_vectors[:, order[:components]]
        right_vectors = weighted_centered.T @ chosen_left
        right_vectors /= np.sqrt(eigenvalues)[None, :]
        loadings = right_vectors.T
        total_variance = float(np.maximum(all_eigenvalues, 0.0).sum())
        solver = "observation-space dual eigendecomposition"

    if total_variance <= 0.0:
        raise ValueError("EW PCA has no positive variance")
    explained = eigenvalues / total_variance
    factors = (historical_z - weighted_mean) @ loadings.T
    backtest_factors = (backtest_z - weighted_mean) @ loadings.T
    return PCAResult(
        factors=factors,
        backtest_factors=backtest_factors,
        loadings=loadings,
        eigenvalues=eigenvalues,
        explained=explained,
        weighted_mean=weighted_mean,
        weights=weights,
        solver=solver,
    )


def build_scenario_grid(
    pca: PCAResult,
    grid_points: Sequence[int],
    tail_density_gamma: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    pc_sigmas = np.sqrt(pca.eigenvalues)
    pc_grids: list[np.ndarray] = []
    for sigma, n_points in zip(pc_sigmas, grid_points):
        half_width = n_points // 2
        u = np.linspace(-1.0, 1.0, n_points)
        multipliers = half_width * np.sign(u) * (np.abs(u) ** tail_density_gamma)
        pc_grids.append(multipliers * sigma)

    # Construct the Cartesian product directly in notebook-compatible C order.
    # This avoids K full mesh arrays in addition to the final scenario matrix.
    dimensions = tuple(len(grid) for grid in pc_grids)
    scenario_count = int(np.prod(dimensions, dtype=np.int64))
    scenario_grid = np.empty((scenario_count, len(pc_grids)), dtype=float)
    for component, coordinates in enumerate(pc_grids):
        inner_repeats = int(np.prod(dimensions[component + 1 :], dtype=np.int64))
        outer_repeats = int(np.prod(dimensions[:component], dtype=np.int64))
        scenario_grid[:, component] = np.tile(
            np.repeat(coordinates, inner_repeats), outer_repeats
        )

    # Discretize each historical observation to its nearest coordinate on every
    # PC. The Cartesian cell count gives a well-defined "most samples" scenario.
    cell_coordinates = np.empty_like(pca.factors, dtype=np.int64)
    for component, coordinates in enumerate(pc_grids):
        distances = np.abs(pca.factors[:, component, None] - coordinates[None, :])
        cell_coordinates[:, component] = np.argmin(distances, axis=1)
    flat_cells = np.ravel_multi_index(
        cell_coordinates.T, dimensions
    )
    historical_counts = np.bincount(flat_cells, minlength=len(scenario_grid))
    return pc_grids, scenario_grid, historical_counts


def prepare_shock_grid_context(
    prepared: PreparedData,
    pca: PCAResult,
    ew_window: int,
) -> ShockGridContext:
    reconstructed = pca.weighted_mean + pca.factors @ pca.loadings
    residuals = prepared.historical_z - reconstructed
    recent_start = max(0, len(pca.factors) - ew_window)
    residuals_recent = residuals[recent_start:]
    return ShockGridContext(
        factors_recent=pca.factors[recent_start:],
        residuals_recent=residuals_recent,
        pc_scale=np.sqrt(np.maximum(pca.eigenvalues, 1e-12)),
        max_abs_z=np.nanmax(np.abs(prepared.historical_z), axis=0),
        fallback_sigma=np.nanstd(residuals_recent, axis=0, ddof=1),
    )


def create_z_shock_grids(
    prepared: PreparedData,
    pca: PCAResult,
    scenario_grid: np.ndarray,
    scenario_index: int,
    ew_window: int,
    nearest_requested: int,
    z_bins: int,
    residual_sigma_range: float,
    inflation_alpha: float,
    inflation_power: float,
    max_inflation: float,
    context: ShockGridContext | None = None,
) -> ShockGridResult:
    selected_scenario = scenario_grid[scenario_index]
    z_hat = pca.weighted_mean + selected_scenario @ pca.loadings
    if context is None:
        context = prepare_shock_grid_context(prepared, pca, ew_window)
    factors_recent = context.factors_recent
    residuals_recent = context.residuals_recent
    nearest_count = min(nearest_requested, len(factors_recent), ew_window)
    if nearest_count < 2:
        raise ValueError("At least two recent factor samples are required")

    scaled_differences = (factors_recent - selected_scenario) / context.pc_scale
    scaled_distances = np.linalg.norm(scaled_differences, axis=1)
    nearest_indices = np.argsort(scaled_distances)[:nearest_count]
    conditional_factors = factors_recent[nearest_indices]
    nearest_residuals = residuals_recent[nearest_indices]

    scenario_distance = float(scaled_distances[nearest_indices].mean())
    inflation_factor = 1.0 + inflation_alpha * scenario_distance**inflation_power
    inflation_factor = float(min(inflation_factor, max_inflation))
    inflated_residuals = nearest_residuals * inflation_factor
    synthetic_z_samples = z_hat + inflated_residuals

    local_sigma = np.nanstd(nearest_residuals, axis=0, ddof=1)
    local_sigma = np.where(
        np.isfinite(local_sigma) & (local_sigma > 1e-12),
        local_sigma,
        context.fallback_sigma,
    )
    local_sigma = np.where(
        np.isfinite(local_sigma) & (local_sigma > 1e-12), local_sigma, 1.0
    )
    inflated_sigma = local_sigma * inflation_factor

    z_low_max = z_hat - context.max_abs_z
    z_high_max = z_hat + context.max_abs_z
    all_bin_edges = np.linspace(z_low_max, z_high_max, z_bins + 1, axis=1)
    all_bin_centers = 0.5 * (all_bin_edges[:, :-1] + all_bin_edges[:, 1:])

    z_low_residual = z_hat - residual_sigma_range * inflated_sigma
    z_high_residual = z_hat + residual_sigma_range * inflated_sigma
    keep = (all_bin_edges[:, :-1] >= z_low_residual[:, None]) & (
        all_bin_edges[:, 1:] <= z_high_residual[:, None]
    )
    invalid_assets = np.flatnonzero(~keep.any(axis=1))
    if invalid_assets.size:
        ticker = prepared.tickers[int(invalid_assets[0])]
        raise ValueError(
            f"No valid z bins for asset {ticker!r}; increase "
            "--residual-sigma-range/--max-inflation-factor or reduce --z-bins"
        )

    asset_z_grids = [centers[mask] for centers, mask in zip(all_bin_centers, keep)]
    full_bin_edges = list(all_bin_edges)
    kept_bin_edges = [
        np.column_stack((edges[:-1][mask], edges[1:][mask]))
        for edges, mask in zip(all_bin_edges, keep)
    ]
    max_vol_ranges = np.column_stack((z_low_max, z_high_max))
    residual_ranges = np.column_stack((z_low_residual, z_high_residual))

    return ShockGridResult(
        selected_scenario=selected_scenario,
        z_hat=z_hat,
        residuals_recent=residuals_recent,
        nearest_residuals=nearest_residuals,
        inflated_residuals=inflated_residuals,
        conditional_factors=conditional_factors,
        factors_recent=factors_recent,
        scenario_distance=scenario_distance,
        inflation_factor=inflation_factor,
        synthetic_z_samples=synthetic_z_samples,
        asset_z_grids=asset_z_grids,
        full_bin_edges=full_bin_edges,
        kept_bin_edges=kept_bin_edges,
        max_vol_ranges=max_vol_ranges,
        residual_ranges=residual_ranges,
    )


def convert_z_to_returns(
    standardizer: StandardScaler, asset_z_grids: Sequence[np.ndarray]
) -> list[dict[str, np.ndarray]]:
    asset_grids: list[dict[str, np.ndarray]] = []
    for asset, z_grid in enumerate(asset_z_grids):
        log_return = standardizer.mean_[asset] + standardizer.scale_[asset] * z_grid
        asset_grids.append(
            {
                "z": z_grid,
                "log_return": log_return,
                "simple_return": np.expm1(log_return),
            }
        )
    return asset_grids


def read_portfolio(path: Path, tickers: Sequence[str]) -> tuple[pd.DataFrame, np.ndarray]:
    portfolio = pd.read_csv(path, dtype={"ticker": str})
    required = {"ticker", "weight"}
    missing_columns = required.difference(portfolio.columns)
    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: {', '.join(sorted(missing_columns))}"
        )
    portfolio["ticker"] = portfolio["ticker"].astype(str)
    portfolio["weight"] = pd.to_numeric(portfolio["weight"], errors="coerce")
    if portfolio["weight"].isna().any() or not np.isfinite(portfolio["weight"]).all():
        raise ValueError("Portfolio weights must be finite numbers")
    if "client_id" in portfolio and portfolio["client_id"].nunique(dropna=False) != 1:
        raise ValueError("portfolio.csv must describe a single client portfolio")

    ticker_set = set(tickers)
    extras = sorted(set(portfolio["ticker"]) - ticker_set)
    if extras:
        preview = ", ".join(extras[:10])
        raise ValueError(f"Portfolio contains assets absent from price data: {preview}")

    grouped = portfolio.groupby("ticker", sort=False)["weight"].sum()
    weights = grouped.reindex(tickers, fill_value=0.0).to_numpy(dtype=float)
    return portfolio, weights


def conditional_neighbors(
    residual_samples: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations, assets = residual_samples.shape
    if observations < 2:
        raise ValueError("At least two residual samples are required for compatibility")

    centered = residual_samples - residual_samples.mean(axis=0)
    variance = np.sum(centered * centered, axis=0) / (observations - 1)
    variance = np.maximum(variance, 1e-12)
    standard_deviation = np.sqrt(variance)
    normalized = centered / standard_deviation

    neighbor_count = min(top_k, max(assets - 1, 0))
    neighbor_indices = np.empty((assets, neighbor_count), dtype=np.int64)
    neighbor_correlations = np.empty((assets, neighbor_count), dtype=float)
    if neighbor_count == 0:
        return standard_deviation, neighbor_indices, neighbor_correlations

    # Calculate exact row blocks instead of retaining an O(N^2) correlation
    # matrix. This produces the notebook's top-|correlation| graph at much lower
    # peak memory for the wider synthetic markets.
    target_bytes = 128 * 1024 * 1024
    rows_per_block = max(1, min(assets, target_bytes // (8 * assets)))
    denominator = observations - 1
    for start in range(0, assets, rows_per_block):
        stop = min(start + rows_per_block, assets)
        correlations = normalized[:, start:stop].T @ normalized / denominator
        correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
        correlations = np.clip(correlations, -0.999, 0.999)
        absolute = np.abs(correlations)
        local_rows = np.arange(stop - start)
        absolute[local_rows, np.arange(start, stop)] = -np.inf

        if assets <= 512:
            chosen = np.argsort(absolute, axis=1)[:, -neighbor_count:]
        else:
            chosen = np.argpartition(
                absolute, kth=assets - neighbor_count, axis=1
            )[:, -neighbor_count:]
            chosen_values = np.take_along_axis(absolute, chosen, axis=1)
            chosen_order = np.argsort(chosen_values, axis=1)
            chosen = np.take_along_axis(chosen, chosen_order, axis=1)

        neighbor_indices[start:stop] = chosen
        neighbor_correlations[start:stop] = np.take_along_axis(
            correlations, chosen, axis=1
        )
    return standard_deviation, neighbor_indices, neighbor_correlations


def variable(asset: int, state: int) -> str:
    return f"x_{asset}_{state}"


def build_qubos(
    asset_grids: Sequence[dict[str, np.ndarray]],
    weights: np.ndarray,
    z_hat: np.ndarray,
    conditional_standard_deviation: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_correlations: np.ndarray,
    lambda_one_hot: float,
    lambda_compat: float,
) -> tuple[dimod.BinaryQuadraticModel, tuple[ModelStats, ModelStats, ModelStats], int]:
    """Build the same QUBO with vectorized coefficients and Ocean bulk loading.

    Structural and portfolio linear arrays are constructed independently, then
    added before the single BQM allocation. This is algebraically identical to
    ``bqm_structural + bqm_portfolio`` but avoids retaining and copying three
    graph-sized BQMs at once.
    """
    state_counts = np.fromiter(
        (len(grid["z"]) for grid in asset_grids),
        dtype=np.int64,
        count=len(asset_grids),
    )
    offsets = np.empty(len(state_counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(state_counts, out=offsets[1:])
    variable_count = int(offsets[-1])

    labels = [
        variable(asset, state)
        for asset, count in enumerate(state_counts)
        for state in range(int(count))
    ]
    state_residuals = [
        grid["z"] - z_hat[asset] for asset, grid in enumerate(asset_grids)
    ]
    variable_indices = [
        np.arange(offsets[asset], offsets[asset + 1])
        for asset in range(len(asset_grids))
    ]
    structural_linear = np.full(variable_count, -lambda_one_hot, dtype=float)
    portfolio_linear = np.concatenate(
        [
            weights[asset] * grid["simple_return"]
            for asset, grid in enumerate(asset_grids)
        ]
    ).astype(float, copy=False)

    one_hot_count = int(np.sum(state_counts * (state_counts - 1) // 2))
    edge_specs: list[tuple[int, int, float, int]] = []
    compatibility_edges = 0

    def compatibility_coefficients(asset: int, other: int, rho: float) -> np.ndarray:
        sigma_asset = conditional_standard_deviation[asset]
        sigma_other = conditional_standard_deviation[other]
        denominator = sigma_other**2 * max(1.0 - rho**2, 1e-12)
        asset_residuals = state_residuals[asset]
        expected_other = rho * sigma_other / sigma_asset * asset_residuals
        other_residuals = state_residuals[other]
        return (
            np.square(other_residuals[None, :] - expected_other[:, None])
            / denominator
        ) * lambda_compat

    # First pass determines the exact allocation size. Recomputing each small
    # state-pair block in the fill pass avoids a second graph-sized set of arrays.
    compatibility_count = 0
    for asset, neighbors in enumerate(neighbor_indices):
        for position, other_value in enumerate(neighbors):
            other = int(other_value)
            # Preserve the notebook's directed-neighbor filtering exactly.
            if other <= asset:
                continue
            rho = float(neighbor_correlations[asset, position])
            if abs(rho) < 1e-6:
                continue
            compatibility_edges += 1

            nonzero_count = int(
                np.count_nonzero(compatibility_coefficients(asset, other, rho))
            )
            if nonzero_count:
                edge_specs.append((asset, other, rho, nonzero_count))
                compatibility_count += nonzero_count

    quadratic_count = one_hot_count + compatibility_count
    quadratic_heads = np.empty(quadratic_count, dtype=np.int64)
    quadratic_tails = np.empty(quadratic_count, dtype=np.int64)
    quadratic_biases = np.empty(quadratic_count, dtype=float)

    triangle_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    cursor = 0
    for asset, count_value in enumerate(state_counts):
        count = int(count_value)
        if count not in triangle_cache:
            triangle_cache[count] = np.triu_indices(count, k=1)
        local_heads, local_tails = triangle_cache[count]
        next_cursor = cursor + len(local_heads)
        quadratic_heads[cursor:next_cursor] = offsets[asset] + local_heads
        quadratic_tails[cursor:next_cursor] = offsets[asset] + local_tails
        quadratic_biases[cursor:next_cursor] = 2.0 * lambda_one_hot
        cursor = next_cursor

    for asset, other, rho, nonzero_count in edge_specs:
        flat_biases = compatibility_coefficients(asset, other, rho).ravel()
        nonzero = flat_biases != 0.0
        next_cursor = cursor + nonzero_count
        asset_variables = variable_indices[asset]
        other_variables = variable_indices[other]
        quadratic_heads[cursor:next_cursor] = np.repeat(
            asset_variables, len(other_variables)
        )[nonzero]
        quadratic_tails[cursor:next_cursor] = np.tile(
            other_variables, len(asset_variables)
        )[nonzero]
        quadratic_biases[cursor:next_cursor] = flat_biases[nonzero]
        cursor = next_cursor

    offset = 0.0
    for _ in asset_grids:
        offset += lambda_one_hot
    bqm_total = dimod.BinaryQuadraticModel.from_numpy_vectors(
        structural_linear + portfolio_linear,
        (quadratic_heads, quadratic_tails, quadratic_biases),
        offset,
        dimod.BINARY,
        variable_order=labels,
    )

    interaction_count = len(bqm_total.quadratic)
    stats = (
        ModelStats(
            "portfolio-independent structural BQM",
            variable_count,
            interaction_count,
            offset,
        ),
        ModelStats("portfolio-dependent BQM", variable_count, 0, 0.0),
        ModelStats(
            "total QUBO BQM / CQM objective",
            variable_count,
            interaction_count,
            offset,
        ),
    )
    return bqm_total, stats, compatibility_edges


def figure_data_uri(figure: plt.Figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=145, bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def table_html(frame: pd.DataFrame, max_rows: int = 200) -> str:
    if len(frame) > max_rows:
        head_count = max_rows // 2
        shown = pd.concat([frame.head(head_count), frame.tail(max_rows - head_count)])
        note = (
            f'<p class="note">Showing the first {head_count} and last '
            f"{max_rows - head_count} of {len(frame):,} rows.</p>"
        )
    else:
        shown = frame
        note = ""
    return note + shown.to_html(index=False, border=0, classes="dataframe")


def describe_values(name: str, values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    return {
        "series": name,
        "observations": array.shape[0],
        "assets": array.shape[1],
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_figures(
    prepared: PreparedData,
    pca: PCAResult,
    pc_grids: Sequence[np.ndarray],
    scenario_grid: np.ndarray,
    scenario_index: int,
    shocks: ShockGridResult,
    asset_grids: Sequence[dict[str, np.ndarray]],
    visual_asset_index: int,
) -> list[tuple[str, str]]:
    figures: list[tuple[str, str]] = []

    if pca.factors.shape[1] >= 2:
        figure, axis = plt.subplots(figsize=(9, 7))
        axis.scatter(
            pca.factors[:, 0],
            pca.factors[:, 1],
            s=13,
            alpha=0.35,
            label="Historical samples",
        )
        axis.scatter(
            pca.backtest_factors[:, 0],
            pca.backtest_factors[:, 1],
            s=13,
            alpha=0.25,
            label="Backtesting samples (not used for calibration)",
        )
        for coordinate in pc_grids[0]:
            axis.axvline(coordinate, linestyle="--", linewidth=0.8, alpha=0.3)
        for coordinate in pc_grids[1]:
            axis.axhline(coordinate, linestyle="--", linewidth=0.8, alpha=0.3)
        axis.scatter(
            scenario_grid[scenario_index, 0],
            scenario_grid[scenario_index, 1],
            marker="x",
            s=150,
            linewidths=3,
            label=f"Detailed scenario {scenario_index}",
        )
        axis.set(xlabel="PC1", ylabel="PC2", title="Factor-space scenario grid")
        axis.legend(loc="best")
    else:
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.scatter(pca.factors[:, 0], np.zeros(len(pca.factors)), s=13, alpha=0.35)
        for coordinate in pc_grids[0]:
            axis.axvline(coordinate, linestyle="--", linewidth=0.8, alpha=0.4)
        axis.scatter(
            scenario_grid[scenario_index, 0], 0, marker="x", s=150, linewidths=3
        )
        axis.set(xlabel="PC1", yticks=[], title="Factor-space scenario grid")
    figures.append(("Factor-space discretization", figure_data_uri(figure)))

    figure, axis = plt.subplots(figsize=(7, 4))
    component_numbers = np.arange(1, len(pca.explained) + 1)
    axis.bar(component_numbers, 100.0 * pca.explained)
    axis.set(
        xlabel="Principal component",
        ylabel="Explained variance (%)",
        title="EW PCA retained-component variance",
        xticks=component_numbers,
    )
    figures.append(("EW PCA explained variance", figure_data_uri(figure)))

    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    factor_axis = axes[0]
    if pca.factors.shape[1] >= 2:
        factor_axis.scatter(
            shocks.factors_recent[:, 0],
            shocks.factors_recent[:, 1],
            s=18,
            alpha=0.35,
            label="Recent EW-window factors",
        )
        factor_axis.scatter(
            shocks.conditional_factors[:, 0],
            shocks.conditional_factors[:, 1],
            s=28,
            alpha=0.75,
            label="Nearest residual donors",
        )
        factor_axis.scatter(
            shocks.selected_scenario[0],
            shocks.selected_scenario[1],
            marker="x",
            s=140,
            linewidths=3,
            label="Detailed scenario",
        )
        factor_axis.set(xlabel="PC1", ylabel="PC2")
    else:
        factor_axis.scatter(
            shocks.factors_recent[:, 0],
            np.zeros(len(shocks.factors_recent)),
            s=18,
            alpha=0.35,
            label="Recent EW-window factors",
        )
        factor_axis.scatter(
            shocks.conditional_factors[:, 0],
            np.zeros(len(shocks.conditional_factors)),
            s=28,
            alpha=0.75,
            label="Nearest residual donors",
        )
        factor_axis.set(xlabel="PC1", yticks=[])
    factor_axis.set_title(
        "Residual donors\n"
        f"distance={shocks.scenario_distance:.2f}, "
        f"inflation={shocks.inflation_factor:.2f}"
    )
    factor_axis.legend(loc="best")

    asset = visual_asset_index
    ticker = prepared.tickers[asset]
    asset_axis = axes[1]
    samples = shocks.synthetic_z_samples[:, asset]
    asset_axis.hist(
        samples,
        bins=min(30, max(8, len(samples) // 4)),
        density=True,
        alpha=0.45,
        edgecolor="none",
        label="Scenario + inflated residuals",
    )
    for position, (left, right) in enumerate(shocks.kept_bin_edges[asset]):
        asset_axis.axvspan(
            left,
            right,
            color="tab:green",
            alpha=0.10,
            label="Kept whole bins" if position == 0 else None,
        )
    for edge in shocks.full_bin_edges[asset]:
        asset_axis.axvline(edge, color="gray", linewidth=0.7, alpha=0.35)
    low_residual, high_residual = shocks.residual_ranges[asset]
    asset_axis.axvline(
        low_residual,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label="Inflated residual border",
    )
    asset_axis.axvline(
        high_residual, color="tab:red", linestyle="--", linewidth=1.5
    )
    asset_axis.axvline(
        shocks.z_hat[asset],
        color="tab:blue",
        linestyle=":",
        linewidth=2,
        label="Scenario projection",
    )
    asset_axis.set(
        yticks=[],
        xlabel=f"{ticker} standardized z-return",
        title=f"{ticker}: z-shock grid ({len(asset_grids[asset]['z'])} states)",
    )
    asset_axis.legend(loc="best")
    figure.tight_layout()
    figures.append(("Conditional residual and asset-grid construction", figure_data_uri(figure)))

    figure, axis = plt.subplots(figsize=(8, 4))
    counts = [len(grid["z"]) for grid in asset_grids]
    bins = np.arange(min(counts) - 0.5, max(counts) + 1.5, 1.0)
    axis.hist(counts, bins=bins, rwidth=0.85)
    axis.set(
        xlabel="Kept states per asset",
        ylabel="Number of assets",
        title="QUBO state-count distribution",
    )
    figures.append(("Asset state counts", figure_data_uri(figure)))
    return figures


def build_report(
    folder: Path,
    paths: dict[str, Path],
    args: argparse.Namespace,
    grid_points: Sequence[int],
    prepared: PreparedData,
    pca: PCAResult,
    scenario_grid: np.ndarray,
    historical_counts: np.ndarray,
    scenario_index: int,
    exported_scenario_indices: Sequence[int],
    export_mode: str,
    shocks: ShockGridResult,
    asset_grids: Sequence[dict[str, np.ndarray]],
    portfolio: pd.DataFrame,
    weights: np.ndarray,
    model_stats: Sequence[ModelStats],
    compatibility_edges: int,
    scenario_model_summary: pd.DataFrame,
    figures: Sequence[tuple[str, str]],
) -> str:
    parameters = pd.DataFrame(
        [
            ("Input subfolder", folder.name),
            ("PCA components", args.components),
            ("EW lambda", args.ew_lambda),
            ("EW window", args.ew_window),
            ("PC grid sizes", ", ".join(map(str, grid_points))),
            ("Tail-density gamma", args.tail_density_gamma),
            ("Detailed report scenario", scenario_index),
            ("Scenario export mode", export_mode),
            ("Scenarios exported this run", len(exported_scenario_indices)),
            ("Candidate z bins", args.z_bins),
            ("Nearest residual donors", len(shocks.nearest_residuals)),
            ("Residual sigma range", args.residual_sigma_range),
            ("Inflation alpha", args.distance_inflation_alpha),
            ("Inflation power", args.distance_inflation_power),
            ("Max inflation", args.max_inflation_factor),
            ("One-hot penalty", args.lambda_one_hot),
            ("Compatibility penalty", args.lambda_compat),
            ("Top-k neighbors", args.top_k_neighbors),
        ],
        columns=["parameter", "value"],
    )
    input_summary = pd.DataFrame(
        [
            {
                "dataset": "historical",
                "file": paths["historical"].name,
                "price rows": len(prepared.historical_close),
                "return rows": len(prepared.historical_returns),
                "first date": prepared.historical_close.index[0].date(),
                "last date": prepared.historical_close.index[-1].date(),
                "assets": len(prepared.tickers),
            },
            {
                "dataset": "backtesting",
                "file": paths["backtest"].name,
                "price rows": len(prepared.backtest_close),
                "return rows": len(prepared.backtest_returns),
                "first date": prepared.backtest_close.index[0].date(),
                "last date": prepared.backtest_close.index[-1].date(),
                "assets": len(prepared.tickers),
            },
        ]
    )
    return_summary = pd.DataFrame(
        [
            describe_values("historical simple returns", prepared.historical_returns.values),
            describe_values(
                "historical log returns", prepared.historical_log_returns.values
            ),
            describe_values("historical z", prepared.historical_z),
            describe_values("backtesting simple returns", prepared.backtest_returns.values),
            describe_values(
                "backtesting log returns", prepared.backtest_log_returns.values
            ),
            describe_values("backtesting z", prepared.backtest_z),
        ]
    )

    pca_table = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.eigenvalues) + 1),
            "eigenvalue": pca.eigenvalues,
            "sigma": np.sqrt(pca.eigenvalues),
            "explained variance": pca.explained,
        }
    )
    scenario_columns = {
        "scenario index": np.arange(len(scenario_grid)),
        "historical samples": historical_counts,
        "exported this run": np.isin(
            np.arange(len(scenario_grid)), exported_scenario_indices
        ),
        "shown in detail": np.arange(len(scenario_grid)) == scenario_index,
    }
    for component in range(scenario_grid.shape[1]):
        scenario_columns[f"PC{component + 1}"] = scenario_grid[:, component]
    scenario_table = pd.DataFrame(scenario_columns)

    asset_table = pd.DataFrame(
        {
            "asset index": np.arange(len(prepared.tickers)),
            "ticker": prepared.tickers,
            "portfolio weight": weights,
            "scenario z": shocks.z_hat,
            "kept states": [len(grid["z"]) for grid in asset_grids],
            "z state min": [grid["z"].min() for grid in asset_grids],
            "z state max": [grid["z"].max() for grid in asset_grids],
            "simple return min": [
                grid["simple_return"].min() for grid in asset_grids
            ],
            "simple return max": [
                grid["simple_return"].max() for grid in asset_grids
            ],
        }
    )
    qubo_table = pd.DataFrame(
        [
            {
                "model": stats.name,
                "variables": stats.variables,
                "interactions": stats.interactions,
                "offset": stats.offset,
            }
            for stats in model_stats
        ]
    )

    sections = [
        "<h2>1. Run parameters</h2>" + table_html(parameters),
        "<h2>2. Input data</h2>"
        + table_html(input_summary)
        + table_html(return_summary)
        + '<p class="note">The first backtesting return is anchored on the last '
        "historical close. The historical scaler is applied to backtesting data; "
        "backtesting observations do not fit PCA or select the scenario.</p>",
        "<h2>3. Exponentially weighted PCA</h2>"
        + f"<p>Solver: {html.escape(pca.solver)}</p>"
        + table_html(pca_table),
        "<h2>4. Lambda-multiple PC scenario grid</h2>"
        + f"<p>{len(scenario_grid):,} scenarios. Scenario "
        f"<strong>{scenario_index}</strong> is shown in detail and has "
        f"{historical_counts[scenario_index]:,} historical samples in its "
        "nearest-grid cell.</p>"
        + table_html(scenario_table, max_rows=500),
        "<h2>5. Detailed-scenario z shocks and raw returns</h2>"
        + f"<p>Mean scaled scenario distance: {shocks.scenario_distance:.6g}; "
        f"residual inflation factor: {shocks.inflation_factor:.6g}.</p>"
        + table_html(asset_table),
        "<h2>6. Portfolio</h2>"
        + f"<p>Source: {html.escape(paths['portfolio'].name)}. Mapped portfolio "
        f"weight sum: {weights.sum():.12g}.</p>"
        + table_html(portfolio),
        "<h2>7. QUBO construction</h2>"
        + f"<p>Sparse residual compatibility asset edges used: "
        f"{compatibility_edges:,} for detailed scenario {scenario_index}. "
        "Each CQM contains the same unconstrained "
        "binary quadratic objective as the total BQM; one-hot behavior remains "
        "encoded by the notebook's quadratic penalty. No solver was called.</p>"
        + table_html(qubo_table)
        + "<h3>Per-scenario output files</h3>"
        + table_html(scenario_model_summary, max_rows=500),
        "<h2>8. Visual outputs</h2>",
    ]
    for title, uri in figures:
        sections.append(
            f"<h3>{html.escape(title)}</h3>"
            f'<img src="{uri}" alt="{html.escape(title)}">'
        )

    style = """
    body { font-family: system-ui, sans-serif; max-width: 1180px; margin: 2rem auto;
           padding: 0 1rem 3rem; color: #17202a; line-height: 1.45; }
    h1, h2, h3 { color: #17365d; }
    h2 { margin-top: 2.2rem; border-bottom: 2px solid #dbe5f1; padding-bottom: .3rem; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; font-size: .9rem; }
    th, td { border: 1px solid #d8dee6; padding: .38rem .5rem; text-align: right; }
    th { background: #edf3f9; position: sticky; top: 0; }
    td:first-child, th:first-child { text-align: left; }
    tr:nth-child(even) { background: #f8fafc; }
    img { display: block; max-width: 100%; height: auto; margin: .5rem auto 2rem; }
    .note { color: #52606d; font-size: .92rem; }
    code { background: #f1f3f5; padding: .1rem .25rem; }
    """
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>QUBO process outputs - {html.escape(folder.name)}</title>"
        f"<style>{style}</style></head><body>"
        f"<h1>Scenario-conditioned QUBO process outputs: "
        f"{html.escape(folder.name)}</h1>"
        "<p>This report records data preparation, EW PCA, scenario selection, "
        "asset shock discretization, portfolio mapping, and QUBO construction. "
        "The QUBO was not solved.</p>"
        + "".join(sections)
        + "</body></html>"
    )


def atomic_write_text(path: Path, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_model(path: Path, model_file: io.BufferedIOBase) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            model_file.seek(0)
            shutil.copyfileobj(model_file, destination)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def export_scenario_models(
    output_dir: Path,
    scenario_index: int,
    bqm: dimod.BinaryQuadraticModel,
) -> tuple[Path, Path]:
    """Serialize one scenario's BQM and equivalent CQM with bounded lifetime."""
    bqm_path = output_dir / "bqms" / f"scenario_{scenario_index}.bqm"
    cqm_path = output_dir / "cqms" / f"scenario_{scenario_index}.cqm"
    bqm_file = bqm.to_file()
    cqm = dimod.ConstrainedQuadraticModel.from_bqm(bqm)
    cqm_file = cqm.to_file()
    del cqm
    try:
        atomic_write_model(bqm_path, bqm_file)
        atomic_write_model(cqm_path, cqm_file)
    finally:
        bqm_file.close()
        cqm_file.close()
    return bqm_path, cqm_path


def main() -> None:
    args = parse_args()
    grid_points = validate_args(args)
    folder = resolve_input_folder(args.subfolder)
    print(f"[1/6] Reading and preparing {folder}", flush=True)
    prepared, paths = prepare_data(folder)
    if not 0 <= args.visual_asset_index < len(prepared.tickers):
        raise ValueError(
            f"--visual-asset-index must be in [0, {len(prepared.tickers) - 1}]"
        )

    print("[2/6] Fitting exponentially weighted PCA", flush=True)
    pca = exponentially_weighted_pca(
        prepared.historical_z,
        prepared.backtest_z,
        args.components,
        args.ew_lambda,
        args.ew_window,
    )

    print("[3/6] Discretizing PC scenario space", flush=True)
    pc_grids, scenario_grid, historical_counts = build_scenario_grid(
        pca, grid_points, args.tail_density_gamma
    )
    if args.scenario_index is None:
        scenario_indices = list(range(len(scenario_grid)))
        export_mode = "all scenarios (default)"
    else:
        scenario_indices = list(dict.fromkeys(args.scenario_index))
        export_mode = "explicit --scenario-index selection"
    invalid_scenarios = [
        index
        for index in scenario_indices
        if not 0 <= index < len(scenario_grid)
    ]
    if invalid_scenarios:
        raise ValueError(
            f"Scenario indices must be in [0, {len(scenario_grid) - 1}], got "
            + ", ".join(map(str, invalid_scenarios))
        )

    detail_scenario_index = max(
        scenario_indices, key=lambda index: historical_counts[index]
    )
    output_dir = output_directory(folder)
    output_dir.mkdir(parents=False, exist_ok=True)
    (output_dir / "bqms").mkdir(exist_ok=True)
    (output_dir / "cqms").mkdir(exist_ok=True)

    print("[4/6] Loading portfolio", flush=True)
    portfolio, weights = read_portfolio(paths["portfolio"], prepared.tickers)
    shock_context = prepare_shock_grid_context(prepared, pca, args.ew_window)

    print(
        f"[5/6] Building and exporting {len(scenario_indices)} scenario(s)",
        flush=True,
    )
    scenario_rows: list[dict[str, object]] = []
    detail_shocks: ShockGridResult | None = None
    detail_asset_grids: list[dict[str, np.ndarray]] | None = None
    detail_model_stats: tuple[ModelStats, ModelStats, ModelStats] | None = None
    detail_compatibility_edges: int | None = None

    for position, scenario_index in enumerate(scenario_indices, start=1):
        print(
            f"      scenario {scenario_index} ({position}/{len(scenario_indices)})",
            flush=True,
        )
        shocks = create_z_shock_grids(
            prepared,
            pca,
            scenario_grid,
            scenario_index,
            args.ew_window,
            args.nearest,
            args.z_bins,
            args.residual_sigma_range,
            args.distance_inflation_alpha,
            args.distance_inflation_power,
            args.max_inflation_factor,
            context=shock_context,
        )
        asset_grids = convert_z_to_returns(
            prepared.standardizer, shocks.asset_z_grids
        )
        conditional_std, neighbor_indices, neighbor_correlations = (
            conditional_neighbors(shocks.inflated_residuals, args.top_k_neighbors)
        )
        bqm_total, model_stats, compatibility_edges = build_qubos(
            asset_grids,
            weights,
            shocks.z_hat,
            conditional_std,
            neighbor_indices,
            neighbor_correlations,
            args.lambda_one_hot,
            args.lambda_compat,
        )
        bqm_path, cqm_path = export_scenario_models(
            output_dir, scenario_index, bqm_total
        )
        total_stats = model_stats[-1]
        scenario_rows.append(
            {
                "scenario index": scenario_index,
                "historical samples": int(historical_counts[scenario_index]),
                "variables": total_stats.variables,
                "interactions": total_stats.interactions,
                "compatibility edges": compatibility_edges,
                "BQM file": bqm_path.relative_to(output_dir).as_posix(),
                "CQM file": cqm_path.relative_to(output_dir).as_posix(),
            }
        )
        if scenario_index == detail_scenario_index:
            detail_shocks = shocks
            detail_asset_grids = asset_grids
            detail_model_stats = model_stats
            detail_compatibility_edges = compatibility_edges
        del bqm_total

    if (
        detail_shocks is None
        or detail_asset_grids is None
        or detail_model_stats is None
        or detail_compatibility_edges is None
    ):
        raise RuntimeError("Detailed report scenario was not constructed")

    scenario_model_summary = pd.DataFrame(scenario_rows)
    print("[6/6] Rendering self-contained process report", flush=True)
    figures = build_figures(
        prepared,
        pca,
        pc_grids,
        scenario_grid,
        detail_scenario_index,
        detail_shocks,
        detail_asset_grids,
        args.visual_asset_index,
    )
    report = build_report(
        folder,
        paths,
        args,
        grid_points,
        prepared,
        pca,
        scenario_grid,
        historical_counts,
        detail_scenario_index,
        scenario_indices,
        export_mode,
        detail_shocks,
        detail_asset_grids,
        portfolio,
        weights,
        detail_model_stats,
        detail_compatibility_edges,
        scenario_model_summary,
        figures,
    )

    report_path = output_dir / "report.html"
    atomic_write_text(report_path, report)

    print(
        "Done. No solver was called.\n"
        f"  Output directory: {output_dir}\n"
        f"  Scenarios exported: {len(scenario_indices)}\n"
        f"  Report: {report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
