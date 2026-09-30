"""Six representatives, one seed each, unchanged settings, local CPU only."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'benchmark_results').mkdir(exist_ok=True)
if list((ROOT/'benchmark_results/runtime_v2_release_smoke').glob('*/experiment.json')):
    raise SystemExit('Existing acceptance smoke found; use run_benchmark.py --resume for its directories.')
results=[]
with (ROOT/'benchmark_results/runtime_v2_release_smoke.log').open('a',encoding='utf-8') as log:
    for n,density in ((200,'sparse'),(200,'dense'),(500,'sparse'),(500,'dense'),(1000,'sparse'),(1000,'dense')):
        command=[sys.executable,str(ROOT/'run_benchmark.py'),str(n),density,'0.05',
                 '--instances','representative','--runs','1','--device','cpu',
                 '--output-root','benchmark_results/runtime_v2_release_smoke']
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=log)
        results.append(dict(n=n,density=density,exit_code=result.returncode))
        if result.returncode not in (0,1):break
(ROOT/'benchmark_results/runtime_v2_release_smoke_exits.json').write_text(
    json.dumps(results,indent=2)+'\n',encoding='utf-8')
sys.exit(2 if len(results)!=6 else int(any(r['exit_code'] for r in results)))
