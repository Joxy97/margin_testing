"""European/American factor stress with explicit pricing-domain diagnostics."""

from dataclasses import dataclass

import numpy as np

from option_pricing.option_book import OptionBook
from .european_option_factor_stress_model import EuropeanOptionFactorStressModel
from .factor_stress_model import FactorStressModel


@dataclass(frozen=True)
class OptionFactorStressModel(EuropeanOptionFactorStressModel):
    """Reuse shared mixed-P&L algebra and delta-aligned residual construction."""

    options: OptionBook
    residualAlignment: str = "fixed_geometry"

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.options, OptionBook):
            raise TypeError("options must be an OptionBook")
        if self.residualAlignment not in ("mixed_delta", "supplied", "fixed_geometry"):
            raise ValueError("invalid residualAlignment")

    @classmethod
    def fromPCAGrid(cls, grid, portfolio, options, *, localPnlGradient=None):
        """An explicit alignment vector permits a nonsmooth center without Greeks.

        The vector selects geometry only; derivative requests at a kink still
        fail, and search uses other smooth anchors plus full nonlinear values.
        """
        if localPnlGradient is None:
            model = super().fromPCAGrid(grid, portfolio, options)
            return cls(model.equityModel, options, "mixed_delta")
        if tuple(grid.instruments) != options.instruments or portfolio.instruments != options.instruments:
            raise ValueError("PCA, portfolio and option instruments must have identical order")
        if grid.calibrationEndDate > options.valuationDate:
            raise ValueError("PCA calibration must not use prices after option valuationDate")
        equity = FactorStressModel.fromPCAGrid(grid, portfolio, localPnlGradient=localPnlGradient)
        return cls(equity, options, "supplied")

    def domainDiagnostics(self, radius: float) -> tuple[dict, ...]:
        if isinstance(radius, bool) or not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be positive and finite")
        diagnostics = []
        for index, (position, context, underlying) in enumerate(zip(
                self.options.positions, self.options.contexts, self.options._underlyingIndices)):
            if not position.quantity:
                continue
            row = self.equityModel.directions[underlying]
            norm = float(np.linalg.norm(row))
            center_spot = float(self.options.spotPrices[underlying]*np.exp(self.equityModel.center[underlying]))
            width = radius*norm
            item = context.screenInterval(float(center_spot*np.exp(-width)), float(center_spot*np.exp(width)))
            distance = None if context.kink is None else float(np.log(context.kink/center_spot))
            item.update(position=index, boundary_log_distance=distance,
                        boundary_reachable=distance is not None and abs(distance) <= width,
                        loading_norm=norm)
            diagnostics.append(item)
        return tuple(diagnostics)
