"""Recheck the delivered smoke artifacts without invoking any optimizer."""
import collections
import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qubo_benchmark.model import Problem
from qubo_benchmark.runtime.common import read,source_identity,filehash
from qubo_benchmark.runtime.storage import load_attempts,primary

def main():
    snapshot=source_identity()['code_snapshot_hash']
    root=ROOT/'benchmark_results/runtime_v2_release_smoke'
    groups=[];scores=0;captures=0
    for path in sorted(root.glob('*/experiment.json')):
        experiment=read(path);core=experiment['core'];directory=path.parent
        assert core['source']['code_snapshot_hash']==snapshot,directory
        assert core['seeds']==[0] and len(core['solvers'])==23
        assert filehash(directory/'BENCHMARK_DEFINITIONS.txt')==filehash(ROOT/'BENCHMARK_DEFINITIONS.txt')
        attempts=load_attempts(directory);records=list(primary(attempts).values())
        assert len(attempts)==len(records)==len(experiment['execution_plan'])==23
        problems={i['entry']['instance_id']:Problem.load(ROOT/'benchmark_data/qubo37/normalized'/
                  (i['entry']['instance_id']+'.npz')) for i in core['instances']}
        for attempt in attempts:
            run=attempt['run'];problem=problems[run['instance_id']]
            for solution in attempt['solutions']:
                # The runner already independently scored every captured vector.
                # Keep this extra acceptance audit bounded to each saved best.
                if solution['valid'] and solution['role']=='best_in_budget':
                    assert len(solution['bitstring'])==problem.n
                    assert problem.score([int(b) for b in solution['bitstring']])==solution['independent_objective']
                    captures+=1
            if run['in_budget_score_available']:
                best=next(s for s in attempt['solutions'] if s['solution_id']==run['best_solution_id'])
                assert best['within_budget'] and best['completed_elapsed_s']<=core['budget_s']
                assert best['independent_objective']==run['best_objective_within_budget']
                scores+=1
            assert run['parameters_json']==core['solvers'][run['solver_id']]
        missing=[dict(solver=a['run']['solver_id'],status=a['run']['result_status'],
                      actual_solve_wall_s=a['run']['actual_solve_wall_s'])
                 for a in records if a['run']['result_status']!='completed']
        groups.append(dict(n=core['n'],density=core['density'],directory=str(directory.relative_to(ROOT)),
                           attempts=len(attempts),statuses=dict(collections.Counter(a['run']['result_status'] for a in records)),
                           limitations=missing))
    assert len(groups)==6
    report=dict(code_snapshot_hash=snapshot,groups=groups,attempts=sum(g['attempts'] for g in groups),
                independently_rechecked_best_solution_records=captures,eligible_best_solutions=scores)
    output=ROOT/'benchmark_results/runtime_v2_release_smoke_audit.json'
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
