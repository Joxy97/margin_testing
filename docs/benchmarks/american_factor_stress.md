# European and American options in the joint PCA–QUBO model

The new `OptionBook`, `OptionFactorStressModel`, and `solveOptionFactorStress`
research APIs extend the original equity/European path to vanilla equity and
spot-index calls and puts with either exercise style. The implementation follows
the supplied `american_pca_qubo_model.md` (12 September 2026), particularly
sections 5–8, 9.1, 10, 11.1–11.3, and 11.5. The original European API remains
available. Application YAML, T0–T3 runners, market conventions, and their existing
American tree pricing defaults retain their behavior.

The updated explanation is [the v2 PDF](../pca_greedy_and_joint_qubo_v2.pdf),
built from [its LaTeX source](../pca_greedy_and_joint_qubo_v2.tex). The original
PDF and source remain available for the European-only formulation.

## Supported financial conventions

Positions use `EquityOptionContract`, signed contract counts and exactly one
contract multiplier. Spots and marks form one dated snapshot. All option
currencies must match the book's `currency` (default USD); the caller supplies
stock exposures in that currency. Equity weights are monetary return exposures,
not share counts. Continuous dividend yield, fixed per-contract IV, and one
fixed book rate are used. American pricing requires nonnegative rate/yield.

The calculation revalues an outstanding frozen book at the horizon. ACT/365
removes calendar days from maturity. Expiry on the horizon uses payoff; expiry
before the horizon fails because settlement needs additional data. Underlying
PCA observations must match the stress horizon; the API does not automatically
rescale daily returns for weekends or longer horizons. Financing, discrete cash
dividends, intervening exercise/assignment, FX, stochastic IV and futures options
are outside this model's scope. Unsupported/materially unpriced positions fail
the calculation; positions are not silently omitted.

## Pricing and calibration

`option_pricing.ju_zhong.VanillaPriceContext` caches one immutable context per
contract/maturity. European contracts use Black–Scholes. American contracts use
the supplied Ju–Zhong boundary equation and stable correction coefficients.
Non-dividend calls and zero-rate puts use their European-equivalent branches.
Zero-rate dividend calls use the analytic zero-rate limit, not an epsilon rate.
`JuZhongPricingModel` provides the scalar `OptionPricingModel` adapter.

For fixed strike, maturity, rate, yield and IV, the boundary is independent of
scenario spot. Vectorized pricing reuses that boundary and evaluates only the
active branch. First and second log-spot derivatives differentiate the same
complete formula. They include the European contribution at the current spot
and positive European spot gamma for calls and puts. At an exercise boundary or
expiry strike (log-distance tolerance `1e-10`), requesting two-sided derivatives
raises `NonsmoothOptionError`. Price-only evaluation still works.

The supplied stable coefficients and corrected European derivative terms were
cross-checked against [QuantLib's official Ju engine source](https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/vanilla/juquadraticengine.cpp).
QuantLib is not a runtime dependency. Tests independently compare with the
existing CRR tree and the numerical examples in the supplied Markdown.

`OptionBook` calibrates with a maintained bracket and Brent's method, rebuilding
the current-date pricing context at every trial IV. The default IV bracket is
`(.01, 5.)`; `volatilityBounds` explicitly changes it. Necessary mark bounds are
checked first, including the deterministic stopping-value lower bound for
American options. Unbracketed marks, invalid trial contexts and a final quoted
price residual above `1e-10 * max(1, mark)` fail with the position index. An
American bracketed root is not reported as a proof of unique IV.

An American mark equal to intrinsic is rejected as unidentifiable by calibration.
For externally assigned IVs, supply `suppliedVolatilities=(value, None, ...)` in
position order. Positive supplied IVs are explicit model inputs, not inferred
values. Their pricing residuals remain in immutable `OptionCalibration` records
and in P&L relative to the actual observed mark. They do not bypass necessary
mark bounds. The original `EuropeanOptionBook` retains its historical inversion;
use `OptionBook` for the new safeguarded European/American calibration.

`model.domainDiagnostics(radius)` screens each held contract's entire marginal
spot interval. The continuation denominator minimum is checked analytically at
endpoints and its quadratic vertex. Economic bounds are checked at 65 log-spaced
spots and again at every evaluated scenario (tolerance `1e-7 * max(spot,strike)`).
These sampled economic checks are not an interval-wide proof. Unsafe denominator,
nonfinite value, nonpositive boundary premium and material pricing-bound
violations raise `OptionPricingError`. Values are never clipped and the backend
is never switched by scenario. This implementation fails unsupported domains;
it does not automatically recalibrate to a fallback tree.

## Shared geometry, objective and boundary search

`OptionFactorStressModel` reuses the existing mixed log-P&L gradient/Hessian
algebra. `fromPCAGrid` aligns the residual with stock plus European plus American
local log delta, using weighted observations instead of a dense residual
covariance. Instrument ordering and PCA/valuation-date validation remain in
force. Direct construction preserves the supplied `FactorStressModel` geometry.

If the center is nonsmooth, default PCA construction rejects undefined Greeks.
The caller can explicitly pass `localPnlGradient=...` to choose an alternative
residual alignment, or directly supply fixed geometry. The result records
`residualAlignment` as `mixed_delta`, `supplied`, or `fixed_geometry`. Such an
alternative does not make the center differentiable. This remains the original
single-residual-direction model: option delta neutrality can hide residual gamma
risk. The Markdown's proposed additional curvature mode is not enabled.

`solveOptionFactorStress` performs these steps on one fixed `A`, `c`, and radius:

1. Screen the pricing domain and identify reachable American/expiry boundaries.
2. Generate center, axis, nearby boundary-side and marginal-extreme seeds. Round
   and project them to the common integer lattice, deduplicating identical seeds.
3. Reprice and repair each seed. At smooth seed anchors build the translated
   quadratic objective and solve it with the supplied existing `BQMSolver`.
   Nonsmooth anchors retain price-only candidates and skip the derivative solve.
4. Check raw integer/product/slack feasibility. Project solver coordinates and
   rebuild auxiliaries **before repricing**; invalid raw scenarios can lie outside
   the screened pricing domain. Repair evaluates only feasible neighbors.
5. Select the lowest full nonlinear P&L over all retained repaired candidates.

Reachable-boundary diagnostics report whether seeds actually occupy each side
after rounding. A continuous crossing does not guarantee either lattice side;
missing sides stay visible. If all coordinates have zero loading and spot is at
a kink, every anchor is nonsmooth and the price-only result is still defined.
Marginal extreme seeds protect an expiry short option against a flat OTM Taylor
objective and a strict-improvement search that cannot escape that plateau.

Each translated polynomial gets its own sufficient penalty bound. Global integer
radius, product encoding, slack and variable count remain unchanged: `d=3`,
`k=8` still gives 122 binary variables. There are no binary exercise decisions.
Use `repairConfig=FactorStressRepairConfig(neighborhood="pairwise")` above three
dimensions. Every unique smooth lattice seed triggers a solve; large numbers of
distinct reachable boundaries therefore increase runtime. Option pricing is
float64 CPU; existing GPU solvers can solve each resulting QUBO.

The search uses global-ball multi-anchor Taylor proposals, without an additional
local trust ball. This is a heuristic, not the optional trust-region extension
in Markdown section 11.4. No stock-only convexity/Taylor certificates are applied
to a nonempty option position. Continuous reference calls still require smooth
points and do not provide global American/nonlinear certificates.

## Results and validation

`OptionStressSearchResult` retains the greatest model loss found, coordinates,
stressed spots, stock/European/American P&L, every repair result, raw solver
feasibility, smooth/nonsmooth anchor counts and boundary-side coverage. Each
repair record contains feasible rebuilt samples, projection/improvement P&L,
iterations, convergence and timings. The raw out-of-domain P&L is intentionally
not evaluated by this search; repair records start from projected coordinates.
Per-contract contexts, calibration residuals and domain diagnostics are available
from the immutable book and fixed model, and are archived by the example.

“Full nonlinear” means the complete selected pricing formula. Ju–Zhong remains an
approximation to American optimal stopping. Neither QUBO feasibility nor local
repair certifies the worst American loss. The example records CRR results at
800 and 1,600 steps using the same IV at the worst candidate, separately from
the reported model margin. Those comparisons are numerical checks, not bounds.

```bash
PYTHONPATH=src python -m unittest tests.test_american_factor_stress \
  tests.test_european_factor_stress tests.test_factor_stress tests.test_factor_stress_repair
PYTHONPATH=src python tools/example_option_factor_stress.py \
  --output docs/benchmarks/american_factor_stress_example.json
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
mkdir -p /tmp/margin_options_pdf
pdflatex -halt-on-error -output-directory=/tmp/margin_options_pdf \
  docs/pca_greedy_and_joint_qubo_v2.tex
cp /tmp/margin_options_pdf/pca_greedy_and_joint_qubo_v2.pdf docs/
```

Validation on 12 September 2026: 48 focused tests passed; the complete Python
suite ran 363 tests successfully with 9 skips. Tests cover calibration and
supplied-IV residuals, independent CRR convergence checks, zero-rate limits,
price/log-derivative parity, the documented slope mismatch, expiry plateaus,
mixed Hessians, residual alignment, QUBO energy/count invariants, domain-limited
repair and boundary-side search. GPU/native builds were not needed for this CPU
pricing/API change; GPU kernels were not modified or separately benchmarked.
The archived example is synthetic, not a historical exercise/assignment backtest.
