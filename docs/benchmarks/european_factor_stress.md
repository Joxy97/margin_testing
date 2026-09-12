# European options in the original factor-stress QUBO

For the extension to both exercise styles, safeguarded calibration and
boundary-aware search, see [american_factor_stress.md](american_factor_stress.md).
The European-only API described below remains available with its original scope.

This implements the fixed-IV European baseline from the
[shared formulation](https://chatgpt.com/share/6aa512ab-2638-83eb-9caa-57782d3229f1).
It uses the original PCA-plus-one-residual model and its integer-ball encoder.
American contracts and futures options are rejected at the option-book boundary.
The T0–T3 runners and application YAML composition are unchanged. Like the
original experimental factor model, this extension is a Python research API.

## Market inputs and exact valuation

`option_pricing.EuropeanOptionBook` receives:

- Canonical underlying `instruments` and aligned positive `spotPrices`.
- A tuple of existing `DerivativePosition` values containing
  `EquityOptionContract(exerciseStyle="E")`, including signed contract quantities.
- `marketPrices` in position order, in quote units before contract multipliers.
- One `valuationDate`, one `horizonDate`, and a finite `riskFreeRate`.

All marks and spots must come from the same valuation-date snapshot. The API
does not download prices or infer exercise style from a ticker. Supply actual
European contract metadata; do not relabel American equity options as European.
The caller must put stock return exposures and option monetary P&L in the same
currency/units. Stock `Portfolio.weights` remain return exposures, not shares:
a holding of 100 shares at spot 100 has exposure 10,000. Option quantities are
contract counts and the contract multiplier is applied exactly once.

Each option's IV is calibrated from its observed mark with the existing bounded
Newton–Raphson Black–Scholes inversion (IV range `1e-6` to `5`, up to 100
iterations). Invalid marks and failed calibration raise errors identifying the
position. Historical realized volatility is not substituted. Calibration arrays
are copied and immutable. Repricing holds each calibrated IV fixed at its strike,
including when the underlying moves. Rates and continuous dividend yields also
remain fixed. This version does not model volatility shocks, discrete dividends,
FX conversion, or funding/cash interest.

For `ell(z) = c + A z`, every option on underlying `i` sees the same
`S_i(z) = S_i0 * exp(ell_i(z))`. Both initial and horizon maturity use ACT/365
calendar days. Thus a Friday-to-Monday horizon removes three days of option time.
The underlying return scenario remains one observation interval from the input
PCA; changing the date gap does not automatically rescale that distribution.
Calibrate the underlying scenario horizon consistently when using this API.

```text
P(z) = sum_i exposure_i * expm1(ell_i(z))
     + sum_o quantity_o * multiplier_o
         * [BS(S_i(z), K_o, tau_horizon, r, q_o, IV_o) - marketPrice_o]
```

At zero log return and zero horizon, calibrated option P&L is zero to the IV
inversion tolerance. `z=0` instead means log return `c`, so its P&L generally is
nonzero. Horizon time decay is included in the constant as well as exact repricing.

An option must be unexpired on the valuation date. Expiration exactly on the
horizon uses intrinsic value. At an exactly at-the-money expiry expansion point,
the Taylor objective is undefined and raises rather than inventing Greeks.
Expiration before the horizon is rejected: its settlement requires a price at
expiry, which the single terminal scenario cannot supply.

## Objective and geometry

`risk_state_generator.EuropeanOptionFactorStressModel` combines a
`FactorStressModel` and a `EuropeanOptionBook`. `quadraticCoefficients()` supplies
the existing `QuadraticStressObjective` and `FactorStressQUBO.build` interfaces.
At the expansion spot `Sbar_i = S_i0 * exp(c_i)` and horizon maturity:

```text
D_i = sum_options_on_i quantity * multiplier * Sbar_i * delta
G_i = sum_options_on_i quantity * multiplier * (Sbar_i * delta + Sbar_i**2 * gamma)

p0 = exact mixed P&L at z=0
g  = A.T @ (exposure * exp(c) + D)
H  = A.T @ diag(exposure * exp(c) + G) @ A
```

Delta and gamma are analytic European Black–Scholes derivatives, evaluated with
the same IV, carry, and maturity as exact repricing. Calls, puts, longs and shorts
all contribute to the same aggregate objective. There is no per-option worst
scenario and no per-option binary variable. Option count changes coefficients
and repricing cost; binary count depends only on factor dimension and resolution.

`EuropeanOptionFactorStressModel.fromPCAGrid(grid, portfolio, options)` keeps the
PCA factors and recomputes the one residual direction from the combined local
log-return gradient `v = exposure * exp(c) + D`. It uses `R v / sqrt(v.T R v)`
through weighted observation products without constructing a dense covariance.
This also supports option-only books using zero stock exposures for their
underlyings. Zero local residual variance produces a zero direction. A
delta-neutral book can therefore lose residual gamma risk in this reduction;
the preserved property is local linear variance, not full nonlinear tail risk.
PCA calibration ending after the option valuation date is rejected.

Direct construction `EuropeanOptionFactorStressModel(equityModel, options)`
preserves the supplied directions, useful when comparing identical geometry.
Instrument order must match exactly. Neither construction adds an IV factor.

## Solving, repair, and reporting

The radius, product auxiliaries, slack encoding, penalty sufficiency argument,
and integer diagnostics are unchanged. Feed the quadratic objective to any
existing binary solver, then use `FactorStressRepair(model, encoding)` and
validate `encoding.diagnostics(repaired.sample)["encoding_feasible"]`.

Repair evaluates option neighborhoods with exact mixed P&L. The historical
stock-only exponential-increment optimization is selected by dispatch for
`FactorStressModel`. Projection, deterministic move ordering, auxiliary rebuild,
iteration limits, and pairwise neighborhoods retain their existing semantics.
Reported candidate margin is `max(0, -repaired.pnl)`; across candidates take the
largest exact margin. The example reports a continuous reference separately and
does not use it as a fallback candidate.

`solveRepricedReference` uses the model's explicit convexity capability. Any
nonzero option position disables the stock-only global certificate, including
long-only books: puts can be nonconvex in log spot. It returns a multistart local
reference without a global lower bound. Stock-only Taylor remainder formulas and
T0–T3 certificates must not be applied to this option model. QUBO optimality is
optimality of the quadratic surrogate, not of exact Black–Scholes P&L.

Vectorized repricing processes positions one at a time, so working storage does
not require a scenarios-by-options matrix. Repair has additional pricing cost
proportional to the option count. It runs in float64 on CPU; existing GPU binary
solvers can still solve the resulting QUBO. No GPU option-pricing kernel is added.

## Example and validation

```bash
PYTHONPATH=src python tools/example_european_factor_stress.py
PYTHONPATH=src python -m unittest tests.test_european_factor_stress \
  tests.test_factor_stress tests.test_factor_stress_repair
```

The offline example creates a synthetic underlying history and European marks,
fits PCA, calibrates IV, solves one binary QUBO with simulated annealing, repairs
the result, and prints both quadratic and exact P&L. It is a runnable composition
example, not a historical option backtest or a calibrated coverage claim.
Replace its synthetic snapshots with actual dated market inputs for research.

Tests cover call/put parity, calibration round trips, ACT/365 theta, signed
quantities/multipliers, analytic log-price derivatives, intrinsic expiry,
rejection of unsupported contracts/invalid snapshots, immutable arrays, mixed
residual variance, QUBO energy equivalence, variable-count invariance, exact
option repair, and the absence of invalid convexity certificates.
