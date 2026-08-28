"""
synthetic_market_generator.py

Generate synthetic close-price CSV files for the portfolio/backtest program.

Output (one directory per universe size):
    assets_00010/historical_close.csv
    assets_00010/backtest_close.csv
    assets_00010/portfolio.csv
    ...

CSV format:
    date,0,1,2,3,...
    2024-01-02,100.23,95.12,...
    ...

The historical interval is:
    [history_start, history_end)

The backtest interval is:
    one pre-start observation + [backtest_start, backtest_end]

Designed to scale comfortably to ~10,000 assets.
"""

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

UNIVERSE_SIZES = (10, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000)
N_ASSETS = max(UNIVERSE_SIZES)
N_SECTORS = 11

HISTORY_START = "2024-01-01"
HISTORY_END = "2024-12-31"      # EXCLUSIVE

BACKTEST_START = "2025-01-01"
BACKTEST_END = "2025-12-28"     # INCLUSIVE

OUTPUT_DIR = Path("synthetic_market")

SEED = 42

# Each scenario has one market-neutral portfolio holding its entire universe.
PORTFOLIO_GROSS_EXPOSURE = 1.0
PORTFOLIO_SEED = SEED + 1


# ============================================================
# Market parameters
# ============================================================

TRADING_DAYS_PER_YEAR: Final = 252

# Market
ANNUAL_MARKET_RETURN = 0.08
ANNUAL_MARKET_VOL = 0.18

# Sector factor
ANNUAL_SECTOR_VOL = 0.12

# Individual stocks
MEAN_IDIO_VOL = 0.20
STD_IDIO_VOL = 0.05

MEAN_MARKET_BETA = 1.0
STD_MARKET_BETA = 0.20

MEAN_SECTOR_BETA = 0.65
STD_SECTOR_BETA = 0.15

# Fat tails
STUDENT_T_DF = 6

# Initial stock prices
INITIAL_PRICE_MIN = 20.0
INITIAL_PRICE_MAX = 300.0

# Very simple stochastic volatility persistence
VOL_PERSISTENCE = 0.97
VOL_SHOCK_SCALE = 0.03


# ============================================================
# Helpers
# ============================================================

def student_t_unit_variance(
    rng: np.random.Generator,
    df: float,
    size: int | tuple[int, ...],
) -> np.ndarray:
    """
    Generate Student-t noise normalized to approximately unit variance.

    Var(T_df) = df / (df - 2)
    for df > 2.
    """
    if df <= 2:
        raise ValueError("Student-t degrees of freedom must be greater than 2.")
    x = rng.standard_t(df, size=size)

    return x / np.sqrt(df / (df - 2))


def previous_business_day(date: str | pd.Timestamp) -> pd.Timestamp:
    """Return the business day immediately preceding `date`."""
    date = pd.Timestamp(date)

    return pd.bdate_range(
        end=date - pd.Timedelta(days=1),
        periods=1,
    )[0]


def generate_portfolios(
    tickers: list[str],
    n_portfolios: int,
    positions_per_portfolio: int,
    gross_exposure: float = 1.0,
    seed: int = 43,
) -> pd.DataFrame:
    """Generate market-neutral portfolios in Experiment Lab long CSV format."""
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("tickers must be non-empty and unique.")
    if n_portfolios < 1:
        raise ValueError("n_portfolios must be at least 1.")
    if not 2 <= positions_per_portfolio <= len(tickers):
        raise ValueError(
            "positions_per_portfolio must be between 2 and the ticker count."
        )
    if not np.isfinite(gross_exposure) or gross_exposure <= 0:
        raise ValueError("gross_exposure must be a positive finite number.")

    rng = np.random.default_rng(seed)
    n_long = positions_per_portfolio // 2
    n_short = positions_per_portfolio - n_long
    side_gross = gross_exposure / 2.0
    records: list[tuple[str, str, float]] = []

    for portfolio_number in range(n_portfolios):
        selected = rng.choice(
            tickers, size=positions_per_portfolio, replace=False
        )
        long_weights = rng.dirichlet(np.ones(n_long)) * side_gross
        short_weights = -rng.dirichlet(np.ones(n_short)) * side_gross
        weights = np.concatenate((long_weights, short_weights))
        client_id = f"synthetic_{portfolio_number:04d}"
        records.extend(
            (client_id, str(ticker), float(weight))
            for ticker, weight in zip(selected, weights, strict=True)
        )

    portfolios = pd.DataFrame(
        records, columns=["client_id", "ticker", "weight"]
    )
    if portfolios.duplicated(["client_id", "ticker"]).any():
        raise RuntimeError("Generated duplicate client/ticker positions.")

    exposure = portfolios.assign(
        long=portfolios["weight"].clip(lower=0.0),
        short=-portfolios["weight"].clip(upper=0.0),
    ).groupby("client_id")[["long", "short"]].sum()
    if not np.allclose(exposure["long"], side_gross):
        raise RuntimeError("Generated portfolios have incorrect long exposure.")
    if not np.allclose(exposure["short"], side_gross):
        raise RuntimeError("Generated portfolios have incorrect short exposure.")
    return portfolios


# ============================================================
# Synthetic market
# ============================================================

def generate_market(
    n_assets: int,
    n_sectors: int,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily close prices.

    Model
    -----

        r_i,t =
              beta_market_i * market_t
            + beta_sector_i * sector_{s(i),t}
            + sigma_i,t * epsilon_i,t

    where:

        market_t  = common market shock
        sector_t  = sector-specific shock
        epsilon   = idiosyncratic Student-t shock

    Asset volatility evolves slowly through time to introduce
    volatility clustering.
    """

    if n_assets < 1:
        raise ValueError("n_assets must be at least 1.")
    if not 1 <= n_sectors <= n_assets:
        raise ValueError("n_sectors must be between 1 and n_assets.")

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must not be after end_date.")

    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(
        start=start,
        end=end,
        inclusive="both",
    )

    n_days = len(dates)

    if n_days < 1:
        raise ValueError("The requested interval contains no business days.")

    print("Generating market:")
    print(f"  Assets:       {n_assets:,}")
    print(f"  Sectors:      {n_sectors}")
    print(f"  Trading days: {n_days:,}")
    print()

    # --------------------------------------------------------
    # Asset metadata
    # --------------------------------------------------------

    sector = rng.integers(
        0,
        n_sectors,
        size=n_assets,
    )

    market_beta = rng.normal(
        MEAN_MARKET_BETA,
        STD_MARKET_BETA,
        size=n_assets,
    )

    market_beta = np.clip(
        market_beta,
        0.3,
        2.0,
    ).astype(np.float32)

    sector_beta = rng.normal(
        MEAN_SECTOR_BETA,
        STD_SECTOR_BETA,
        size=n_assets,
    )

    sector_beta = np.clip(
        sector_beta,
        0.1,
        1.5,
    ).astype(np.float32)

    annual_idio_vol = rng.normal(
        MEAN_IDIO_VOL,
        STD_IDIO_VOL,
        size=n_assets,
    )

    annual_idio_vol = np.clip(
        annual_idio_vol,
        0.05,
        0.60,
    ).astype(np.float32)

    daily_idio_vol = (
        annual_idio_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    ).astype(np.float32)

    # --------------------------------------------------------
    # Daily market parameters
    # --------------------------------------------------------

    market_daily_mu = (
        ANNUAL_MARKET_RETURN / TRADING_DAYS_PER_YEAR
    )

    market_daily_vol = (
        ANNUAL_MARKET_VOL / np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    sector_daily_vol = (
        ANNUAL_SECTOR_VOL / np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    # --------------------------------------------------------
    # Initial prices
    # --------------------------------------------------------

    initial_prices = rng.uniform(
        INITIAL_PRICE_MIN,
        INITIAL_PRICE_MAX,
        size=n_assets,
    ).astype(np.float32)

    # Store close prices.
    prices = np.empty(
        (n_days, n_assets),
        dtype=np.float32,
    )

    prices[0] = initial_prices

    # Mean-reverting log-volatility shared by the market. Modelling the log
    # multiplier keeps the state positive and lets shocks persist as intended.
    log_volatility_state = 0.0

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    for t in range(1, n_days):

        # ====================================================
        # Slow-moving volatility regime
        # ====================================================

        volatility_shock = rng.normal(
            0.0,
            VOL_SHOCK_SCALE,
        )

        log_volatility_state = (
            VOL_PERSISTENCE * log_volatility_state + volatility_shock
        )
        volatility_state = float(
            np.clip(np.exp(log_volatility_state), 0.5, 3.0)
        )

        # ====================================================
        # Market factor
        # ====================================================

        market_noise = student_t_unit_variance(
            rng,
            STUDENT_T_DF,
            1,
        )[0]

        market_return = (
            market_daily_mu
            + market_daily_vol
            * volatility_state
            * market_noise
        )

        # ====================================================
        # Sector factors
        # ====================================================

        sector_noise = student_t_unit_variance(
            rng,
            STUDENT_T_DF,
            n_sectors,
        )

        sector_returns = (
            sector_daily_vol
            * volatility_state
            * sector_noise
        )

        # ====================================================
        # Idiosyncratic stock shocks
        # ====================================================

        idio_noise = student_t_unit_variance(
            rng,
            STUDENT_T_DF,
            n_assets,
        ).astype(np.float32)

        # ====================================================
        # Combine factors
        # ====================================================

        log_returns = (
            market_beta * market_return
            + sector_beta * sector_returns[sector]
            + daily_idio_vol
            * volatility_state
            * idio_noise
        )

        # Prevent pathological one-day simulated returns.
        log_returns = np.clip(
            log_returns,
            -0.50,
            0.50,
        )

        # Log-return style price evolution keeps prices positive.
        prices[t] = prices[t - 1] * np.exp(log_returns)

        if t % 100 == 0:
            print(
                f"\rSimulating day {t:,}/{n_days - 1:,}",
                end="",
            )

    print("\nSimulation complete.")

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    tickers = [
        str(i)
        for i in range(n_assets)
    ]

    df = pd.DataFrame(
        prices,
        index=dates,
        columns=tickers,
    )

    df.index.name = "date"

    return df


# ============================================================
# Main
# ============================================================

def _validate_intervals(
    history_start: pd.Timestamp,
    history_end: pd.Timestamp,
    backtest_start: pd.Timestamp,
    backtest_end: pd.Timestamp,
) -> None:
    if history_start >= history_end:
        raise ValueError("HISTORY_START must be before exclusive HISTORY_END.")
    if backtest_start > backtest_end:
        raise ValueError("BACKTEST_START must not be after BACKTEST_END.")


def _validate_close_panel(frame: pd.DataFrame, name: str) -> None:
    if len(frame) < 2:
        raise ValueError(f"{name} requires at least two observations.")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be unique and increasing.")
    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite prices.")
    if (values <= 0).any():
        raise ValueError(f"{name} contains non-positive prices.")


def write_benchmark_scenarios(
    historical: pd.DataFrame,
    backtest: pd.DataFrame,
    universe_sizes: tuple[int, ...],
    output_dir: Path,
    portfolio_seed: int,
) -> pd.DataFrame:
    """Write nested-universe close panels and one full-universe portfolio each."""
    if not universe_sizes:
        raise ValueError("At least one universe size is required.")
    if tuple(sorted(set(universe_sizes))) != universe_sizes:
        raise ValueError("Universe sizes must be unique and strictly increasing.")
    if universe_sizes[0] < 2 or universe_sizes[-1] > historical.shape[1]:
        raise ValueError("Universe sizes must be between 2 and the asset count.")
    if list(historical.columns) != list(backtest.columns):
        raise ValueError("Historical and backtest ticker columns must match.")

    rows: list[dict[str, object]] = []
    previous_tickers: set[str] = set()
    for size in universe_sizes:
        tickers = list(historical.columns[:size])
        ticker_set = set(tickers)
        if not previous_tickers.issubset(ticker_set):
            raise RuntimeError("A benchmark universe does not include its predecessor.")

        scenario_name = f"assets_{size:05d}"
        scenario_dir = output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        history_slice = historical.loc[:, tickers]
        backtest_slice = backtest.loc[:, tickers]
        portfolio = generate_portfolios(
            tickers=tickers,
            n_portfolios=1,
            positions_per_portfolio=size,
            gross_exposure=PORTFOLIO_GROSS_EXPOSURE,
            seed=portfolio_seed + size,
        )

        history_slice.to_csv(
            scenario_dir / "historical_close.csv",
            date_format="%Y-%m-%d",
            float_format="%.8g",
        )
        backtest_slice.to_csv(
            scenario_dir / "backtest_close.csv",
            date_format="%Y-%m-%d",
            float_format="%.8g",
        )
        portfolio.to_csv(
            scenario_dir / "portfolio.csv",
            index=False,
            float_format="%.17g",
        )
        rows.append(
            {
                "scenario": scenario_name,
                "assets": size,
                "portfolio_positions": len(portfolio),
            }
        )
        previous_tickers = ticker_set
        print(f"  Wrote {scenario_name}: {size:,} assets, one portfolio")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    return manifest


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_start = pd.Timestamp(HISTORY_START)
    history_end = pd.Timestamp(HISTORY_END)

    backtest_start = pd.Timestamp(BACKTEST_START)
    backtest_end = pd.Timestamp(BACKTEST_END)
    _validate_intervals(history_start, history_end, backtest_start, backtest_end)

    # --------------------------------------------------------
    # Backtest CSV requires one pre-start close.
    # --------------------------------------------------------

    pre_backtest_date = previous_business_day(
        backtest_start
    )

    # --------------------------------------------------------
    # Determine complete simulation interval.
    # --------------------------------------------------------

    simulation_start = min(
        history_start,
        pre_backtest_date,
    )

    simulation_end = backtest_end

    print("=" * 60)
    print("Synthetic Equity Market Generator")
    print("=" * 60)

    print(f"Simulation interval: {simulation_start.date()}")
    print(f"                  -> {simulation_end.date()}")
    print()

    # --------------------------------------------------------
    # Generate one continuous market.
    # --------------------------------------------------------

    market = generate_market(
        n_assets=N_ASSETS,
        n_sectors=N_SECTORS,
        start_date=simulation_start,
        end_date=simulation_end,
        seed=SEED,
    )

    # ========================================================
    # Historical data
    #
    # History start = INCLUSIVE
    # History end   = EXCLUSIVE
    # ========================================================

    historical = market.loc[
        (market.index >= history_start)
        & (market.index < history_end)
    ].copy()

    # ========================================================
    # Backtest data
    #
    # Include one observation before BACKTEST_START.
    # BACKTEST_END is inclusive.
    # ========================================================

    backtest_main = market.loc[
        (market.index >= backtest_start)
        & (market.index <= backtest_end)
    ].copy()

    # Select from the generated index instead of assuming that the preceding
    # weekday is an available session (it may be a market holiday).
    available_before_start = market.loc[market.index < backtest_start]
    if available_before_start.empty:
        raise ValueError(
            "The simulation interval does not contain a close before "
            "BACKTEST_START."
        )
    pre_start = available_before_start.tail(1).copy()
    pre_backtest_date = pre_start.index[-1]

    backtest = pd.concat([
        pre_start,
        backtest_main,
    ])

    backtest = (
        backtest
        .sort_index()
        .loc[lambda x: ~x.index.duplicated()]
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    expected_tickers = [str(i) for i in range(N_ASSETS)]
    if list(historical.columns) != expected_tickers:
        raise RuntimeError("Generated ticker columns do not match the requested assets.")
    _validate_close_panel(historical, "Historical CSV")
    _validate_close_panel(backtest, "Backtest CSV")
    if backtest.index[0] >= backtest_start:
        raise RuntimeError("Backtest CSV is missing its pre-start close.")

    # --------------------------------------------------------
    # Write nested benchmark scenarios
    # --------------------------------------------------------

    print()
    print("Writing nested benchmark scenarios...")
    manifest = write_benchmark_scenarios(
        historical=historical,
        backtest=backtest,
        universe_sizes=UNIVERSE_SIZES,
        output_dir=OUTPUT_DIR,
        portfolio_seed=PORTFOLIO_SEED,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Finished")
    print("=" * 60)

    print()
    print(f"Output:    {OUTPUT_DIR}")
    print(f"Scenarios: {len(manifest)}")
    print(f"Universes: {', '.join(f'{size:,}' for size in UNIVERSE_SIZES)}")
    print(f"Historical shape (largest): {historical.shape}")
    print(
        f"Historical dates: "
        f"{historical.index.min().date()} -> "
        f"{historical.index.max().date()}"
    )
    print(
        f"Backtest dates: "
        f"{backtest.index.min().date()} -> "
        f"{backtest.index.max().date()}"
    )

    print()
    print(
        f"Backtest pre-start close: "
        f"{pre_backtest_date.date()}"
    )

    print()
    print("First few historical rows:")
    print(
        historical.iloc[:3, :5]
    )


if __name__ == "__main__":
    main()
