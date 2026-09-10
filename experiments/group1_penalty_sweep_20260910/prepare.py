"""Capture the same 105 portfolio objectives for every timed solver trial."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from margin_engine import MarginApplicationConfig
from risk_state_generator import RiskStateGenerationContext


HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    output = HERE / 'qubos'
    output.mkdir(exist_ok=True)
    started = perf_counter()
    app = MarginApplicationConfig.fromYaml(HERE / 'configuration.yaml')
    engine = app.createEngine()
    data = engine.getPortfolioMarketData(app.portfolio, app.marginDate)
    generator = engine.riskStateGenerator
    request = generator.createDataRequest(app.portfolio, app.marginDate)
    context = RiskStateGenerationContext(data, request, app.marginDate)
    visitor = engine.marginCalculator.bqmVisitor
    records = []
    for index, state in enumerate(generator.getRiskStates(context)):
        before = perf_counter()
        objective = visitor.createBQM(state, app.portfolio, {'lambdaOneHot': 0., 'lambdaCompat': 0.})
        factors = state.correlations
        offsets = objective.groupOffsets
        heads = offsets[factors.firstAssets] + factors.firstStates
        tails = offsets[factors.secondAssets] + factors.secondStates
        normalized = visitor.normalizedCorrelationCoefficients(factors)
        path = output / f'{index:03d}.npz'
        np.savez(path, linear=objective.linear, offsets=offsets, heads=heads,
                 tails=tails, raw=factors.coefficients, normalized=normalized,
                 seed=np.array(objective.seedOffset, dtype=np.uint64))
        record = dict(scenario=index, variables=objective.variableCount, groups=len(offsets)-1,
                      raw_correlation_edges=len(heads), retained_correlation_edges=int(np.count_nonzero(normalized)),
                      raw_max=float(factors.coefficients.max(initial=0.)),
                      greedy_margin=max(0., -sum(float(objective.linear[a:b].min()) for a,b in zip(offsets[:-1],offsets[1:]))),
                      snapshot_seconds=perf_counter()-before, elapsed_seconds=perf_counter()-started,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        records.append(record)
        (HERE/'preparation_status.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    if len(records)!=105 or any(r['groups']!=8590 for r in records):
        raise ValueError('Expected 105 QUBOs with all 8590 stocks')
    (HERE/'qubos_manifest.json').write_text(json.dumps(dict(scenarios=records,
        seconds=perf_counter()-started, torch=torch.__version__, cuda=torch.version.cuda,
        hardware=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]),indent=2)+'\n')


if __name__=='__main__':
    main()
