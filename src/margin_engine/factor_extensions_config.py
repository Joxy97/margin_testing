"""Versioned opt-in research configuration, separate from application YAML."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import yaml

from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolverConfig


def _mapping(value, path):
    if not isinstance(value, Mapping):
        raise TypeError(f'{path} must be a mapping')
    return dict(value)


def _keys(value, allowed, path):
    unknown = value.keys()-set(allowed)
    if unknown:
        raise ValueError(f'{path}: unknown keys {sorted(unknown, key=str)}')


def _number(value, name, lower, upper=np.inf, *, integer=False, inclusive=False):
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        raise TypeError(f'{name} must be {"an integer" if integer else "numeric"}')
    if not np.isfinite(value) or not (lower <= value if inclusive else lower < value) or value > upper:
        raise ValueError(f'{name} is outside its valid range')


def _booleans(instance, names):
    for name in names:
        if type(getattr(instance, name)) is not bool:
            raise TypeError(f'{name} must be boolean')


def _section(cls, value, path):
    value = _mapping(value, path)
    _keys(value, [f.name for f in fields(cls)], path)
    return cls(**value)


@dataclass(frozen=True)
class ExtensionReferenceConfig:
    global_quadratic_trs: bool = False
    taylor_bracket: bool = False
    full_residual_convex_bound: bool = False
    exact_repricing: bool = True
    certificate_kind: str = 'numerical'
    dtype: str = 'float64'

    def __post_init__(self):
        _booleans(self, ('global_quadratic_trs', 'taylor_bracket', 'full_residual_convex_bound', 'exact_repricing'))
        if self.dtype != 'float64' or self.certificate_kind != 'numerical':
            raise ValueError('reference supports only float64 numerical certificates')
        if self.taylor_bracket and not self.global_quadratic_trs:
            raise ValueError('taylor_bracket requires global_quadratic_trs')


@dataclass(frozen=True)
class ExtensionEstimationConfig:
    dependence_window: int = 125
    legacy_decay: float = .93
    common_components: int = 2
    rank_tolerance: float = 1e-12

    def __post_init__(self):
        _number(self.dependence_window, 'dependence_window', 1, integer=True)
        _number(self.legacy_decay, 'legacy_decay', 0, 1)
        _number(self.common_components, 'common_components', 0, 2, integer=True)
        _number(self.rank_tolerance, 'rank_tolerance', 0, .01)


@dataclass(frozen=True)
class ExtensionStressConfig:
    radius_mode: str = 'fixed'
    radius: float | None = 3.
    nominal_region_content: str | None = None
    probability_dimension: int | None = None

    def __post_init__(self):
        if self.radius_mode == 'fixed':
            _number(self.radius, 'radius', 0)
            if self.nominal_region_content is not None or self.probability_dimension is not None:
                raise ValueError('fixed radius has no probability label')
        elif self.radius_mode == 'legacy_gaussian':
            if self.radius is not None:
                raise ValueError('legacy_gaussian requires radius: null')
            _number(self.probability_dimension, 'probability_dimension', 0, integer=True)
            if not isinstance(self.nominal_region_content, str):
                raise TypeError('nominal_region_content must be an exact probability string')
            try:
                probability = Decimal(self.nominal_region_content)
            except InvalidOperation as error:
                raise ValueError('invalid nominal_region_content') from error
            if not probability.is_finite() or not 0 < probability < 1 or not 0 < float(probability) < 1:
                raise ValueError('nominal_region_content must be representable strictly between zero and one')
        else:
            raise ValueError('radius_mode must be fixed or legacy_gaussian')

    def resolvedRadius(self, encoded_dimension, active_dimension, mode):
        if self.radius_mode == 'fixed':
            return float(self.radius)
        if self.probability_dimension != encoded_dimension:
            raise ValueError('probability_dimension must match the specified coordinate model')
        if mode == 'extended' and active_dimension != encoded_dimension:
            raise ValueError('rank-deficient extended model requires an explicitly fixed geometric radius')
        from scipy.stats import chi2
        return float(np.sqrt(chi2.ppf(float(self.nominal_region_content), self.probability_dimension)))


@dataclass(frozen=True)
class ExtensionEncodingConfig:
    enabled: bool = False
    bits_per_coordinate: int = 8
    penalty_multiplier: float = 1.
    repair: bool = True
    repair_max_steps: int = 1024

    def __post_init__(self):
        _booleans(self, ('enabled', 'repair'))
        _number(self.bits_per_coordinate, 'bits_per_coordinate', 2, 8, integer=True, inclusive=True)
        _number(self.penalty_multiplier, 'penalty_multiplier', 0, inclusive=True)
        _number(self.repair_max_steps, 'repair_max_steps', 0, integer=True, inclusive=True)


@dataclass(frozen=True)
class FactorStressExtensionsConfig:
    schema_version: int = 1
    mode: str = 'legacy'
    reference: ExtensionReferenceConfig = ExtensionReferenceConfig()
    estimation: ExtensionEstimationConfig = ExtensionEstimationConfig()
    stress: ExtensionStressConfig = ExtensionStressConfig()
    encoding: ExtensionEncodingConfig = ExtensionEncodingConfig()

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('factor_stress_extensions.schema_version must be 1')
        if self.mode not in ('legacy', 'extended'):
            raise ValueError('factor_stress_extensions.mode must be legacy or extended')
        for name, cls in (('reference', ExtensionReferenceConfig), ('estimation', ExtensionEstimationConfig),
                          ('stress', ExtensionStressConfig), ('encoding', ExtensionEncodingConfig)):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f'{name} must be {cls.__name__}')

    @classmethod
    def fromMapping(cls, value):
        value = _mapping(value, 'factor_stress_extensions')
        _keys(value, [f.name for f in fields(cls)], 'factor_stress_extensions')
        for name, section in (('reference', ExtensionReferenceConfig), ('estimation', ExtensionEstimationConfig),
                              ('stress', ExtensionStressConfig), ('encoding', ExtensionEncodingConfig)):
            value[name] = _section(section, value.get(name, {}), f'factor_stress_extensions.{name}')
        return cls(**value)


@dataclass(frozen=True)
class FactorExtensionsExperimentConfig:
    market_path: Path
    margin_date: date
    portfolio_path: Path | None
    seed: int
    extensions: FactorStressExtensionsConfig
    solver: BQMSolverConfig
    baseline_archive: Path | None
    decision_cutoff_utc: str
    prior_close_available_time_utc: str

    @classmethod
    def fromYaml(cls, path):
        path = Path(path).resolve()
        return cls.fromMapping(yaml.safe_load(path.read_text()), path.parent)

    @classmethod
    def fromMapping(cls, value, base):
        from datetime import datetime, time, timezone

        value = _mapping(value, 'experiment')
        _keys(value, ('market', 'seed', 'factor_stress_extensions', 'solver', 'baseline_archive'), 'experiment')
        market = _mapping(value.get('market'), 'market')
        _keys(market, ('prices_csv', 'margin_date', 'portfolio_csv', 'decision_cutoff_utc',
                       'prior_close_available_time_utc'), 'market')
        def resolve(item):
            if not isinstance(item, str) or not item:
                raise TypeError('configured paths must be nonempty strings')
            return (Path(base)/item).resolve()
        day = date.fromisoformat(str(market['margin_date']))
        cutoff = datetime.fromisoformat(market['decision_cutoff_utc'].replace('Z', '+00:00'))
        if cutoff.tzinfo is None or cutoff.utcoffset().total_seconds() != 0 or cutoff.date() != day:
            raise ValueError('decision_cutoff_utc must be UTC on margin_date')
        available_time = time.fromisoformat(market['prior_close_available_time_utc'])
        if available_time.tzinfo is not None:
            raise ValueError('prior_close_available_time_utc uses an implicit UTC timezone')
        seed = value.get('seed', 20260910)
        _number(seed, 'seed', 0, 2**32-1, integer=True, inclusive=True)
        solver = _mapping(value.get('solver', {'type': 'lib_simulated_annealing',
            'solverParameters': {'runs': 8, 'sweeps': 100}}), 'solver')
        _keys(solver, ('type', 'constructorParameters', 'solverParameters'), 'solver')
        if not isinstance(solver.get('type'), str):
            raise TypeError('solver.type must be a string')
        constructor = _mapping(solver.get('constructorParameters', {}), 'constructorParameters')
        parameters = _mapping(solver.get('solverParameters', {}), 'solverParameters')
        if 'seed' in parameters:
            raise ValueError('configure the paired seed at experiment.seed')
        return cls(resolve(market['prices_csv']), day,
            resolve(market['portfolio_csv']) if market.get('portfolio_csv') else None, seed,
            FactorStressExtensionsConfig.fromMapping(value.get('factor_stress_extensions', {})),
            BQMSolverConfig(solver['type'], constructor, parameters),
            resolve(value['baseline_archive']) if value.get('baseline_archive') else None,
            cutoff.astimezone(timezone.utc).isoformat(), available_time.isoformat())
