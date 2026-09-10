"""Fetch each completed remote portfolio, verify it locally, and publish plots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from report_risky_factor_backtests import reportPortfolio, reportAll
from benchmark_factor_stress import writeJson


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path("experiments/group1_risky_portfolios_20260910")
REMOTE_ROOT = "/workspace/group1-risky-portfolios-20260910"
SSH = ["ssh", "-S", "/tmp/group1-factor-ssh", "-o", "BatchMode=yes", "-p", "28732", "root@159.48.242.5"]


def remote(command):
    return subprocess.run([*SSH, command], check=True, capture_output=True, text=True).stdout


def fetch(identifier):
    if identifier not in {f"p{i:02d}" for i in range(1, 11)}:
        raise ValueError("unexpected portfolio identifier")
    archive = f"/tmp/risky-factor-{identifier}.tar.gz"
    names = [str(EXPERIMENT/"results"/identifier)]
    for name in ("settings.json", "portfolios.csv", "shared_timing.json", "shared_preparation_stages.csv", "large_asset_moves.csv"):
        relative = str(EXPERIMENT/"results"/name)
        if remote("test -f "+shlex.quote(REMOTE_ROOT+"/"+relative)+" && echo yes || true").strip() == "yes":
            names.append(relative)
    command = "cd "+shlex.quote(REMOTE_ROOT)+" && "+shlex.join(["tar", "-czf", archive, *names])+" && "+shlex.join(["sha256sum", archive])
    checksum = remote(command).split()[0]
    subprocess.run(["scp", "-o", "ControlPath=/tmp/group1-factor-ssh", "-o", "BatchMode=yes", "-P", "28732",
                    "root@159.48.242.5:"+archive, archive], check=True)
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest() != checksum:
        raise ValueError("portfolio transfer checksum mismatch")
    permitted = (ROOT/EXPERIMENT/"results").resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not (ROOT/member.name).resolve().is_relative_to(permitted) or not (member.isdir() or member.isfile()):
                raise ValueError("unsafe archive member")
        tar.extractall(ROOT, filter="data")
    reportPortfolio(permitted, identifier)
    reportAll(permitted)
    writeJson(permitted/identifier/"local_verification.json", dict(status="passed", archive_sha256=checksum,
        portfolio=identifier, sample_verification="all raw/repaired samples reloaded and repriced locally",
        chart="margin_loss.png", sensitivity_chart="confidence_penalty.png"))
    print(json.dumps(dict(stage="fetched_and_plotted", portfolio=identifier,
                         directory=str(permitted/identifier))), flush=True)


def main():
    output = ROOT/EXPERIMENT/"results"
    output.mkdir(parents=True, exist_ok=True)
    fetched = {f"p{i:02d}" for i in range(1, 11) if (output/f"p{i:02d}"/"local_verification.json").exists()}
    prefix = REMOTE_ROOT+"/"+str(EXPERIMENT)
    while len(fetched) < 10:
        try:
            state = json.loads(remote("cat "+shlex.quote(prefix+"/results/status.json")))
            raw = remote("cat "+shlex.quote(prefix+"/results/portfolio_timings.json")+" 2>/dev/null || true")
            timing = json.loads(raw) if raw.strip() else {"portfolios": []}
            completed = [r["portfolio"] for r in timing["portfolios"]]
            for identifier in completed:
                if identifier not in fetched:
                    fetch(identifier)
                    fetched.add(identifier)
            writeJson(output/"portfolio_timings.json", timing)
            writeJson(output/"remote_status.json", state)
            progress = dict(stage="tracking", fetched=sorted(fetched), remote=state)
            if state.get("stage") == "portfolio":
                status = remote("cat "+shlex.quote(prefix+"/results/"+state["portfolio"]+"/status.json")+" 2>/dev/null || true")
                if status.strip():
                    detail = json.loads(status)
                    progress["current_jobs"] = detail.get("completed_jobs")
                    progress["total_jobs"] = detail.get("total_jobs")
            print(json.dumps(progress), flush=True)
            if len(fetched) == 10:
                break
            running = remote("pgrep -f '^/workspace/margin-sweep-venv/bin/python tools/run_risky_factor_backtests.py' || true").strip()
            if not running:
                print(remote("tail -n 35 "+shlex.quote(prefix+"/run.log")), flush=True)
                raise RuntimeError("remote runner stopped before all portfolios completed")
        except subprocess.CalledProcessError as error:
            print(json.dumps(dict(stage="connection_retry", error=str(error))), flush=True)
        time.sleep(25)
    # The final runner summary is published immediately after its last portfolio.
    for name in ("timings.json", "status.json", "portfolio_timings.json"):
        content = remote("cat "+shlex.quote(prefix+"/results/"+name))
        writeJson(output/name, json.loads(content))
    (ROOT/EXPERIMENT/"run.log").write_text(remote("cat "+shlex.quote(prefix+"/run.log")))
    reportAll(output)
    print(json.dumps(dict(stage="all_portfolios_fetched", portfolios=sorted(fetched))), flush=True)


if __name__ == "__main__":
    main()
