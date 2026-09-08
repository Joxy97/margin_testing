#!/usr/bin/env python3
"""Fit a sparsity-conditioned polynomial scheduler on the BiqMac sweep."""

from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fit_torch_svl_polynomial import FEATURES, matrix

SOLVER_START = FEATURES.index("runs")
BASE_NAMES = list(FEATURES)
INTERACTION_NAMES = [f"density * {name}" for name in FEATURES[SOLVER_START:]]
EXPANDED_NAMES = BASE_NAMES + ["density^2"] + INTERACTION_NAMES


def expand(x: np.ndarray) -> np.ndarray:
    density = x[:, FEATURES.index("density") : FEATURES.index("density") + 1]
    return np.column_stack((x, density ** 2, density * x[:, SOLVER_START:]))


def make_model(alpha: float) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        transformer=StandardScaler(),
    )


def run(raw: Path) -> None:
    with raw.open(newline="") as source:
        rows = list(csv.DictReader(source))
    train = [row for row in rows if row["split"] == "train"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    x = expand(matrix(train)); y = np.asarray([float(row["quality"]) for row in train])
    groups = np.asarray([row["instance"] for row in train])
    candidates = []
    for alpha in (.01, .1, 1., 10., 100., 1_000., 10_000.):
        errors = []
        for fit_indices, validation_indices in GroupKFold(4).split(x, y, groups):
            fitted = make_model(alpha).fit(x[fit_indices], y[fit_indices])
            errors.extend(abs(y[validation_indices] - fitted.predict(x[validation_indices])))
        candidates.append((float(np.mean(errors)), alpha))
    cv_mae, alpha = min(candidates)
    fitted = make_model(alpha).fit(x, y)
    xe = expand(matrix(evaluation)); observed = np.asarray([float(row["quality"]) for row in evaluation])
    predicted = fitted.predict(xe)
    scheduler = []; correlations = []
    for instance in sorted({row["instance"] for row in evaluation}):
        indices = [i for i, row in enumerate(evaluation) if row["instance"] == instance]
        chosen = max(indices, key=lambda i: predicted[i]); oracle = max(indices, key=lambda i: observed[i])
        correlation = spearmanr(observed[indices], predicted[indices]).statistic
        correlations.append(float(correlation) if np.isfinite(correlation) else 0.0)
        row = evaluation[chosen]
        scheduler.append({
            "instance": instance, "stratum": row["stratum"], "density": row["density"],
            "chosen_configuration": row["configuration_id"],
            **{name: row[name] for name in ("runs","steps","dt","mass","damping","temperature","integrator",
               "transverse_field_initial","transverse_field_final","problem_scale_initial","problem_scale_final")},
            "predicted_quality": predicted[chosen], "actual_quality": observed[chosen],
            "oracle_quality": observed[oracle], "quality_regret": observed[oracle] - observed[chosen],
        })
    metrics = {
        "shape": "linear main effects + density^2 + density_by_every_solver_parameter",
        "ridge_alpha": alpha, "train_group_cv_mae": cv_mae,
        "evaluation_mae": mean_absolute_error(observed, predicted),
        "evaluation_rmse": mean_squared_error(observed, predicted) ** .5,
        "evaluation_r2": r2_score(observed, predicted),
        "mean_within_instance_spearman": float(np.mean(correlations)),
        "scheduler_mean_regret": float(np.mean([row["quality_regret"] for row in scheduler])),
        "scheduler_max_regret": float(np.max([row["quality_regret"] for row in scheduler])),
        "distinct_scheduler_choices": len({row["chosen_configuration"] for row in scheduler}),
        "choices_by_stratum": {
            stratum: sorted({row["chosen_configuration"] for row in scheduler if row["stratum"] == stratum})
            for stratum in sorted({row["stratum"] for row in scheduler})
        },
    }
    output = raw.parent
    joblib.dump({"model": fitted, "base_features": FEATURES, "expanded_features": EXPANDED_NAMES},
                output / "sparsity_quality_scheduler.joblib")
    (output / "sparsity_scheduler_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (output / "sparsity_scheduler_evaluation.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=scheduler[0]); writer.writeheader(); writer.writerows(scheduler)
    coefficients = fitted.regressor_.named_steps["ridge"].coef_
    with (output / "sparsity_scheduler_terms.csv").open("w", newline="") as destination:
        writer = csv.writer(destination); writer.writerow(("term", "standardized_coefficient", "absolute_coefficient"))
        writer.writerows(sorted(zip(EXPANDED_NAMES, coefficients, abs(coefficients)), key=lambda item: -item[2]))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("results", type=Path)
    arguments = parser.parse_args(); run(arguments.results.resolve())
