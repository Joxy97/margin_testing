"""Check completed portfolio returns and calibration dates against source prices."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from benchmark_factor_stress import writeJson
from sweep_factor_stress import loadModel


def audit(root, group):
    settings = json.loads((root/"settings.json").read_text())
    if hashlib.sha256(group.read_bytes()).hexdigest() != settings["group_sha256"]:
        raise ValueError("source price file differs from the remote run")
    prices = pd.read_csv(group, index_col=0).sort_index()
    dates = list(prices.index)
    if dates[126:] != settings["dates"] or settings["design_cutoff"] != dates[125]:
        raise ValueError("evaluation dates or portfolio selection cutoff differ")
    completed = []
    for metadata in settings["portfolios"]:
        directory = root/metadata["id"]
        if not (directory/"local_verification.json").exists():
            continue
        marker = directory/"market_data_verification.json"
        if marker.exists() and json.loads(marker.read_text()).get("raw_samples_repriced") == settings["expected_trials_per_portfolio"]:
            completed.append(metadata["id"])
            continue
        portfolio = pd.read_csv(directory/"portfolio.csv", float_precision="round_trip").set_index("ticker").weight
        portfolio = portfolio[portfolio != 0]
        values = prices.loc[:, portfolio.index].to_numpy()
        pnls = (values[126:]/values[125:-1]-1) @ portfolio.to_numpy()
        days = json.loads((directory/"days.json").read_text())["days"]
        np.testing.assert_allclose(pnls, [d["realized_pnl"] for d in days], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(np.maximum(0, -pnls), [d["realized_loss"] for d in days], atol=1e-12, rtol=1e-12)
        for index, day in enumerate(days, start=126):
            if (day["date"] != dates[index] or day["calibration_start"] != dates[index-125]
                    or day["calibration_end"] != dates[index-1] or day["calibration_observations"] != 125):
                raise ValueError("PCA window does not contain exactly the prior 125 returns")
        daily = pd.read_csv(directory/"daily_margins.csv")
        if not np.array_equal((daily.realized_loss > daily.margin).to_numpy(), daily.breach.to_numpy()):
            raise ValueError("daily breach CSV does not compare margin directly with realized loss")
        models, sample_ids = {}, set()
        for path in sorted((directory/"batches").glob("*.json")):
            for row in json.loads(path.read_text())["rows"]:
                if row["id"] in sample_ids:
                    raise ValueError("duplicate raw sample")
                sample_ids.add(row["id"])
                if row["day"] not in models:
                    models[row["day"]] = loadModel(directory/"models"/f"{row['day']:03d}.npz")[0]
                model = models[row["day"]]
                raw = np.load(directory/"samples"/(row["id"]+".npy"), allow_pickle=False)
                # Decode the 24 coordinate bits directly, independently of the
                # quadratization builder and the recorded coordinate vector.
                integers = raw[:24].reshape(3, 8) @ (1 << np.arange(8))-127
                coordinates = integers*(row["radius"]/127)
                np.testing.assert_allclose(coordinates, row["coordinates"], atol=1e-14, rtol=0)
                np.testing.assert_allclose(float(model.pnl(coordinates)), row["raw_repriced_pnl"], atol=1e-12, rtol=1e-12)
        if len(sample_ids) != settings["expected_trials_per_portfolio"]:
            raise ValueError("raw sample coverage is incomplete")
        result = dict(status="passed", dates=len(days), independent_realized_pnl=True,
                      clipped_loss=True, calibration_window=125, selection_excludes_evaluation_period=True,
                      raw_samples_repriced=len(sample_ids))
        writeJson(directory/"market_data_verification.json", result)
        completed.append(metadata["id"])
    writeJson(root/"market_data_verification.json", dict(status="passed", portfolios=completed,
        group_sha256=settings["group_sha256"]))
    print(json.dumps(dict(stage="market_data_verified", portfolios=completed)))
    return len(completed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        count = audit(args.root.resolve(), args.group.resolve())
        if not args.watch or count == 10:
            break
        time.sleep(20)
