"""Pre-period portfolio designs and explicit nominal Gaussian coverage settings."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta, chi2

from risk_state_generator.factor_stress_model import FactorStressModel


def confidenceRadius(coverage: float, dimensions: int = 3) -> float:
    if isinstance(coverage, bool) or not 0 < coverage < 1:
        raise ValueError("nominal coverage must be strictly between zero and one")
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("dimensions must be a positive integer")
    return float(np.sqrt(chi2.ppf(coverage, dimensions)))


def breachUpperBound(breaches: int, dates: int) -> float:
    """One-sided 95% binomial bound; independent-date assumption is explicit."""
    if not 0 <= breaches <= dates or dates < 1:
        raise ValueError("invalid breach/date counts")
    return 1. if breaches == dates else float(beta.ppf(.95, breaches+1, dates-breaches))


def compactModel(model: FactorStressModel) -> FactorStressModel:
    """Remove zero exposures after deriving the residual direction on the full grid."""
    selected = model.exposures != 0
    if not np.any(selected):
        raise ValueError("portfolio must contain nonzero exposure")
    return FactorStressModel(tuple(np.asarray(model.instruments)[selected]), model.exposures[selected],
                            model.center[selected], model.directions[selected])


def constructPortfolios(prices, grid):
    """Use only the 126 initial closes; no evaluation-period return is inspected."""
    values = prices.loc[:, list(grid.instruments)].to_numpy(dtype=float)
    if values.shape[0] != 126 or grid.ew_window != 125:
        raise ValueError("portfolio design requires exactly 126 pre-period closes")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("design prices must be finite and positive")
    log_returns = np.diff(np.log(values), axis=0)
    eligible = (values.min(axis=0) >= 1.) & (values[-1] >= 5.) & (np.abs(log_returns).max(axis=0) <= np.log(2.))
    if eligible.sum() < 100:
        raise ValueError("fewer than 100 stocks pass pre-period price-quality filters")
    volatility = log_returns.std(axis=0)
    residual = (grid.residuals*grid.logReturnScale).std(axis=0)
    pca = np.abs(grid.logReturnScale*grid.loadings[0])
    def order(score):
        return np.array(sorted(np.flatnonzero(eligible), key=lambda i: (-score[i], str(grid.instruments[i]))))
    vol, resid, factor = order(volatility), order(residual), order(pca)
    residual_alternatives = np.array([i for i in resid if i not in set(vol[:10])])[:10]
    factor_alternatives = np.array([i for i in factor if i not in set(vol[:10]) | set(residual_alternatives)])[:10]
    definitions = [
        ("p01", "High-volatility 50 long", vol[:50], [], 1., 0.),
        ("p02", "High-volatility 10 long", vol[:10], [], 1., 0.),
        ("p03", "Residual-volatility 10 long", residual_alternatives, [], 1., 0.),
        ("p04", "PCA-factor concentrated 10 long", factor_alternatives, [], 1., 0.),
        ("p05", "High-volatility 20 short", [], vol[:20], 0., 1.),
        ("p06", "Residual-volatility 10 short", [], residual_alternatives, 0., 1.),
        ("p07", "High-volatility 20/20 long-short", vol[:20], vol[20:40], 1., 1.),
        ("p08", "Concentrated 5/5 long-short", vol[:5], vol[5:10], 1., 1.),
        ("p09", "Residual-volatility 20/20 long-short", resid[:20], resid[20:40], 2., 2.),
        ("p10", "Short-biased 10/30 long-short", factor[:10], [], 1., 2.),
    ]
    # Keep the two sides disjoint for the mixed factor/residual design.
    definitions[-1] = (*definitions[-1][:3], np.array([i for i in resid if i not in set(factor[:10])])[:30], 1., 2.)
    rows, weights = [], []
    simple = np.expm1(log_returns)
    for identifier, name, long, short, long_gross, short_gross in definitions:
        weight = np.zeros(len(grid.instruments))
        if len(long):
            weight[np.asarray(long)] = long_gross/len(long)
        if len(short):
            weight[np.asarray(short)] = -short_gross/len(short)
        pnl = simple @ weight
        rows.append(dict(id=identifier, name=name, positions=int(np.count_nonzero(weight)),
            gross=float(np.abs(weight).sum()), net=float(weight.sum()),
            long_exposure=long_gross, short_exposure=short_gross,
            design_pnl_std=float(pnl.std()), design_worst_loss=max(0., -float(pnl.min())),
            design_best_gain=float(pnl.max())))
        weights.append(weight)
    return rows, np.array(weights), eligible


def configurations():
    """Primary three seeds; paired single-seed sensitivity probes."""
    rows = []
    for coverage in (.99, .999, .9995, .9999):
        rows.append(dict(id=f"c{round(coverage*10000):04d}_p1", coverage=coverage,
                         radius=confidenceRadius(coverage), multiplier=1., repeats=3 if coverage == .9995 else 1,
                         primary=coverage == .9995))
    for multiplier in (10., 100.):
        rows.append(dict(id=f"c9995_p{int(multiplier)}", coverage=.9995,
                         radius=confidenceRadius(.9995), multiplier=multiplier, repeats=1, primary=False))
    return rows
