#!/usr/bin/env python3
"""Select, fit, and evaluate an instance-held-out polynomial quality model."""

from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURES=("vertices","density","weight_mean","weight_std","negative_weight_fraction","runs","steps","dt",
          "mass","damping","temperature","integrator_code","transverse_field_initial","transverse_field_final",
          "problem_scale_initial","problem_scale_final")

def matrix(rows):
    return np.asarray([[float(r[f]) if f!="integrator_code" else float(r["integrator"]=="weak_order_2") for f in FEATURES] for r in rows])

def model(degree,alpha):
    reg=make_pipeline(SimpleImputer(),StandardScaler(),PolynomialFeatures(degree,include_bias=False),Ridge(alpha=alpha))
    return TransformedTargetRegressor(regressor=reg,transformer=StandardScaler())

def run(raw: Path) -> None:
    with raw.open(newline="") as source: rows=list(csv.DictReader(source))
    train=[r for r in rows if r["split"]=="train"]; test=[r for r in rows if r["split"]=="evaluation"]
    x=matrix(train); y=np.asarray([float(r["quality"]) for r in train]); groups=np.asarray([r["instance"] for r in train])
    candidates=[]
    for degree in (1,2,3):
        for alpha in (.01,.1,1.,10.,100.,1_000.,10_000.,100_000.,1_000_000.):
            errors=[]
            for a,b in GroupKFold(4).split(x,y,groups):
                fitted=model(degree,alpha).fit(x[a],y[a]); errors.extend(abs(y[b]-fitted.predict(x[b])))
            candidates.append((float(np.mean(errors)),degree,alpha))
    cv_mae,degree,alpha=min(candidates); fitted=model(degree,alpha).fit(x,y)
    xt=matrix(test); yt=np.asarray([float(r["quality"]) for r in test]); prediction=fitted.predict(xt)
    metrics={"selected_degree":degree,"ridge_alpha":alpha,"train_group_cv_mae":cv_mae,
             "evaluation_mae":mean_absolute_error(yt,prediction),"evaluation_rmse":mean_squared_error(yt,prediction)**.5,
             "evaluation_r2":r2_score(yt,prediction),"train_instances":len(set(groups)),
             "evaluation_instances":len({r["instance"] for r in test}),"train_rows":len(train),"evaluation_rows":len(test)}
    metrics["evaluation_by_stratum"] = {
        stratum: {"rows": len(indices), "mae": mean_absolute_error(yt[indices], prediction[indices]),
                  "rmse": mean_squared_error(yt[indices], prediction[indices]) ** .5}
        for stratum in sorted({r["stratum"] for r in test})
        for indices in [[i for i, r in enumerate(test) if r["stratum"] == stratum]]
    }
    scheduler=[]; rank_correlations=[]
    for name in sorted({r["instance"] for r in test}):
        indices=[i for i,r in enumerate(test) if r["instance"]==name]; chosen=max(indices,key=lambda i:prediction[i]); oracle=max(indices,key=lambda i:yt[i])
        correlation=spearmanr(yt[indices], prediction[indices]).statistic
        rank_correlations.append(float(correlation) if np.isfinite(correlation) else 0.0)
        scheduler.append({"instance":name,"stratum":test[chosen]["stratum"],"chosen_configuration":test[chosen]["configuration_id"],
                          **{key: test[chosen][key] for key in ("runs","steps","dt","mass","damping","temperature","integrator",
                              "transverse_field_initial","transverse_field_final","problem_scale_initial","problem_scale_final")},
                          "predicted_quality":prediction[chosen],"actual_quality":yt[chosen],"oracle_quality":yt[oracle],
                          "quality_regret":yt[oracle]-yt[chosen]})
    metrics["scheduler_mean_regret"]=float(np.mean([r["quality_regret"] for r in scheduler]))
    metrics["scheduler_max_regret"]=float(np.max([r["quality_regret"] for r in scheduler]))
    metrics["mean_within_instance_spearman"] = float(np.mean(rank_correlations))
    metrics["distinct_scheduler_choices"] = len({r["chosen_configuration"] for r in scheduler})
    train_by_configuration = {
        configuration: np.mean([float(r["quality"]) for r in train if r["configuration_id"] == configuration])
        for configuration in {r["configuration_id"] for r in train}
    }
    global_choice = max(train_by_configuration, key=train_by_configuration.get)
    global_regrets = []
    for name in sorted({r["instance"] for r in test}):
        instance_rows = [r for r in test if r["instance"] == name]
        oracle = max(float(r["quality"]) for r in instance_rows)
        chosen = next(float(r["quality"]) for r in instance_rows if r["configuration_id"] == global_choice)
        global_regrets.append(oracle - chosen)
    metrics["global_training_choice"] = global_choice
    metrics["global_choice_mean_regret"] = float(np.mean(global_regrets))
    output=raw.parent
    joblib.dump({"model": fitted, "features": FEATURES}, output/"quality_polynomial.joblib")
    polynomial=fitted.regressor_.named_steps["polynomialfeatures"]
    coefficients=fitted.regressor_.named_steps["ridge"].coef_
    with (output/"model_terms.csv").open("w",newline="") as dest:
        writer=csv.writer(dest); writer.writerow(("term","standardized_coefficient","absolute_coefficient"))
        writer.writerows(sorted(zip(polynomial.get_feature_names_out(FEATURES),coefficients,abs(coefficients)),key=lambda row:-row[2]))
    (output/"model_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    with (output/"scheduler_evaluation.csv").open("w",newline="") as dest:
        writer=csv.DictWriter(dest,fieldnames=scheduler[0]); writer.writeheader(); writer.writerows(scheduler)
    with (output/"degree_selection.csv").open("w",newline="") as dest:
        writer=csv.writer(dest); writer.writerow(("cv_mae","degree","alpha")); writer.writerows(sorted(candidates))
    fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(yt,prediction,c=[float(r["density"]) for r in test],cmap="viridis",alpha=.75)
    lo=min(yt.min(),prediction.min()); hi=max(yt.max(),prediction.max()); ax.plot([lo,hi],[lo,hi],"k--")
    ax.set(xlabel="Observed quality",ylabel="Predicted quality",title=f"Held-out BiqMac instances: degree {degree} ridge polynomial")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output/"predicted_vs_observed.png",dpi=180); plt.close(fig)
    print(json.dumps(metrics,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("results",type=Path); args=parser.parse_args(); run(args.results.resolve())
