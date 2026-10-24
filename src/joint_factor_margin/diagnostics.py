"""Existing references and bounds, with explicit applicability statuses."""

from functools import singledispatchmethod
from typing import Protocol, TYPE_CHECKING

from option_pricing import NonsmoothOptionError
from risk_state_generator import FactorStressModel
from margin_calculator.optimization import QuadraticStressObjective
from margin_calculator.optimization.factor_stress_reference import (
    solveQuadraticReference, solveRepricedReference, solveGlobalQuadraticReference,
)
from margin_calculator.optimization.factor_stress_bounds import (
    taylor_ball_bound, lattice_discretization_bound, margin_bracket,
)
from .values import SearchResult

if TYPE_CHECKING:
    from .scenarios import StressModel
    from .settings import DiagnosticsConfig
    from margin_calculator.optimization import FactorStressQUBOConfig


class DiagnosticEvaluator(Protocol):
    def evaluate(self, model: "StressModel", selected: SearchResult) -> dict: ...


class ExistingDiagnostics:
    def __init__(self, config: "DiagnosticsConfig", encoding: "FactorStressQUBOConfig", cudaBounds=None):
        self.config, self.encoding, self.cudaBounds = config, encoding, cudaBounds

    def evaluate(self, model: "StressModel", selected: SearchResult) -> dict:
        result = {"references": {"status": "disabled"}, "bounds": {"status": "disabled"}}
        if self.config.references:
            try:
                objective = QuadraticStressObjective(*model.quadraticCoefficients())
                result["references"] = {
                    "status": "computed",
                    "quadratic": solveQuadraticReference(objective, self.encoding.radius),
                    "repriced": solveRepricedReference(model, self.encoding.radius,
                        boundCoordinates=self.config.boundCoordinates),
                }
            except NonsmoothOptionError as error:
                result["references"] = {"status": "unsupported", "reason": str(error)}
        if self.config.bounds:
            result["bounds"] = self._bounds(model, selected)
        return result

    @singledispatchmethod
    def _bounds(self, model, selected):
        return {"status": "unsupported", "reason": "Existing stock Taylor bounds do not apply to option books"}

    @_bounds.register(FactorStressModel)
    def _(self, model, selected):
        objective = QuadraticStressObjective(*model.quadraticCoefficients())
        quadratic = solveGlobalQuadraticReference(objective, self.encoding.radius)
        if self.config.boundsBackend == "cuda":
            if self.cudaBounds is None:
                from margin_calculator.optimization.factor_stress_cuda_bounds import CUDABallBounds
                self.cudaBounds = CUDABallBounds(self.config.cudaDevice)
            values = self.cudaBounds.diagnostics(model, self.encoding.radius, selected.coordinates,
                quadratic, self.encoding.bitsPerCoordinate, scope="legacy_reduced")
            return {"status": "computed", "quadratic": quadratic, **values,
                    "execution": self.cudaBounds.metadata()}
        return {"status": "computed", "quadratic": quadratic,
                "taylor": taylor_ball_bound(model, self.encoding.radius),
                "lattice": lattice_discretization_bound(model, self.encoding.radius, self.encoding.bitsPerCoordinate),
                "marginBracket": margin_bracket(model, self.encoding.radius, selected.coordinates, quadratic)}
