"""Deadline-bounded binary-solver penalty sweep with frozen SAC inference."""

import argparse
import csv
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import signal
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
SOLVERS = {'SBM':'torch_sbm','SVL':'torch_svl','TRF':'torch_transverse_route'}


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def make_problem(arrays, variant, strength, visitor):
    from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem
    coefficients = arrays['raw'] if variant=='legacy' else arrays['normalized']
    retained = coefficients != 0.
    heads, tails = arrays['heads'][retained], arrays['tails'][retained]
    biases = .1 * coefficients[retained]
    bound = np.abs(arrays['linear']).copy()
    np.add.at(bound, heads, np.abs(biases))
    np.add.at(bound, tails, np.abs(biases))
    bound = float(bound.max(initial=0.))
    penalty = max(1., float(np.nextafter(1.01*bound, np.inf))) if strength=='bound' else float(strength)
    template = visitor.oneHotTopology(tuple(np.diff(arrays['offsets']).tolist()), penalty)
    problem = QUBOProblem(arrays['linear']-penalty,
        np.concatenate((template.heads, heads)), np.concatenate((template.tails, tails)),
        np.concatenate((template.biases,biases)), offset=penalty*(len(arrays['offsets'])-1),
        groupOffsets=arrays['offsets'], seedOffset=int(arrays['seed']))
    return problem, penalty, bound


def parameters(name, model, bank, actor, problem, steps, runs, features=None):
    import torch
    arm = 0
    features = np.array([math.log1p(problem.variableCount)/math.log1p(10000),
        min(1., 2*problem.interactionCount/max(1,problem.variableCount*(problem.variableCount-1))),
        float(np.median(np.abs(problem.quadraticBiases[::max(1,problem.interactionCount//1024)]))>10),
        1.,0.,0.,0.,0.],dtype=np.float32) if features is None else np.asarray(features,dtype=np.float32)
    if model=='sac':
        with torch.inference_mode():
            arm=int(actor(torch.tensor(features)).argmax())
    supplied=dict(bank[name][arm])
    # Keep learned algorithm parameters; bound execution resources identically.
    supplied.update(steps=steps,runs=runs,run_batch_size=runs,dtype='float32',
                    seed=20260910,energy_chunk_size=65536)
    if name=='TRF':
        supplied.update(matrix_format='sparse',max_variables=250000,
                        candidate_interval=steps,candidate_batch_size=max(2*runs,2),
                        deduplicate_candidates=True,cuda_graph=True,graph_steps=16)
    return supplied,arm,features.tolist()


def worker(gpu, indices, inbox, events, output):
    try:
        import torch
        from benchmark_biqmac_sac import network
        from margin_calculator.optimization.portfolio_risk_state_bqm_visitor import PortfolioRiskStateBQMVisitor
        from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem
        from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolverFactory
        from margin_calculator.optimization.optimization_solver.bqm_solver.candidate_selection import CandidateSelection
        from margin_calculator.optimization.optimization_solver.bqm_solver.torch_candidates import TorchCandidateAccumulator
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(gpu)
        bank=json.loads((HERE/'sac_action_bank.json').read_text())
        saved=torch.load(HERE/'sac_selected_actors.pt',map_location='cpu',weights_only=True)
        actors={name:network(len(bank[name])) for name in SOLVERS}
        for name,actor in actors.items():
            actor.load_state_dict(saved[name]); actor.eval()
        solvers={name:BQMSolverFactory.create(kind,{'device':f'cuda:{gpu}'}) for name,kind in SOLVERS.items()}
        visitor=PortfolioRiskStateBQMVisitor()
        cache={}
        for index in indices:
            with np.load(HERE/'qubos'/f'{index:03d}.npz') as data:
                cache[index]={name:data[name] for name in data.files}
        # Freeze the reference features across penalties and correlation variants
        # so the sweep isolates their effect rather than changing solver controls.
        reference_features={}
        for index in indices:
            reference,_,_=make_problem(cache[index],'pruned','1',visitor)
            _,_,reference_features[index]=parameters('SBM','default',bank,actors['SBM'],reference,16,1)
        del reference
        tiny=QUBOProblem(np.array([-1.,-1.]),np.array([0]),np.array([1]),np.array([2.]))
        for name,solver in solvers.items():
            supplied,_,_=parameters(name,'default',bank,actors[name],tiny,16,1)
            solver.solve(tiny,supplied)
        torch.cuda.synchronize()
        metrics={}
        original_repair=CandidateSelection._repairCandidate
        original_model=CandidateSelection._repairModel
        original_add=TorchCandidateAccumulator.add
        def timed_repair(*args):
            started=time.perf_counter(); metrics['repair_calls']+=1
            try: return original_repair(*args)
            finally: metrics['repair_seconds']+=time.perf_counter()-started
        def timed_model(*args):
            started=time.perf_counter()
            try: return original_model(*args)
            finally: metrics['repair_model_seconds']+=time.perf_counter()-started
        def counted_add(accumulator,samples):
            selected=samples[:,accumulator.groupVariables].long()
            prefix=torch.cat((torch.zeros((len(samples),1),device=samples.device,dtype=torch.int64),selected.cumsum(dim=1)),dim=1)
            counts=prefix[:,accumulator.groupOffsets[1:]]-prefix[:,accumulator.groupOffsets[:-1]]
            metrics['raw_candidates']+=len(samples)
            metrics['raw_feasible']+=int((counts==1).all(dim=1).sum())
            metrics['violated_groups']+=int((counts!=1).sum())
            metrics['checked_groups']+=counts.numel()
            return original_add(accumulator,samples)
        CandidateSelection._repairCandidate=staticmethod(timed_repair)
        CandidateSelection._repairModel=staticmethod(timed_model)
        TorchCandidateAccumulator.add=counted_add
        def expired(*_args): raise TimeoutError('Per-scenario time allowance expired')
        signal.signal(signal.SIGALRM,expired)
        events.put(dict(type='ready',gpu=gpu,scenarios=len(indices)))
        with (Path(output)/f'worker_{gpu}.jsonl').open('a',buffering=1) as stream:
            while True:
                trial=inbox.get()
                if trial is None: break
                deadline=trial['deadline']; name=trial['solver']
                for position,index in enumerate(indices):
                    remaining=deadline-time.monotonic()
                    if remaining<=.5: break
                    allowance=max(.1,(remaining-.5)/(len(indices)-position))
                    metrics.update(repair_calls=0,repair_seconds=0.,repair_model_seconds=0.,raw_candidates=0,raw_feasible=0,
                                   violated_groups=0,checked_groups=0)
                    row=dict(type='result',trial=trial['id'],scenario=index,gpu=gpu,
                             solver=name,variant=trial['variant'],penalty_setting=trial['strength'],
                             policy=trial['policy'],steps=trial['steps'],runs=trial['runs'],allowance_seconds=allowance)
                    started=time.perf_counter()
                    signal.setitimer(signal.ITIMER_REAL,allowance)
                    try:
                        problem,penalty,bound=make_problem(cache[index],trial['variant'],trial['strength'],visitor)
                        supplied,arm,features=parameters(name,trial['policy'],bank,actors[name],problem,trial['steps'],trial['runs'],reference_features[index])
                        row.update(lambdaOneHot=penalty,sufficient_bound=bound,arm=arm,features=features,parameters=supplied,
                                   variables=problem.variableCount,interactions=problem.interactionCount)
                        torch.cuda.synchronize(); solve_started=time.perf_counter()
                        result=solvers[name].solve(problem,supplied)
                        torch.cuda.synchronize()
                        solve_seconds=time.perf_counter()-solve_started
                        bits=np.asarray(result.sample,dtype=np.uint8)
                        counts=np.add.reduceat(bits.astype(np.int64),cache[index]['offsets'][:-1])
                        if not np.all(counts==1): raise ValueError('Returned solution violates one-hot constraints')
                        energy=problem.energy(bits)
                        if not math.isclose(energy,result.energy,rel_tol=1e-9,abs_tol=1e-6):
                            raise ValueError('Original QUBO scoring mismatch')
                        elapsed=time.perf_counter()-started
                        row.update(status='solved' if time.monotonic()<=deadline else 'late',
                                   energy=energy,loss=max(0.,-float(cache[index]['linear']@bits)),solve_seconds=solve_seconds,
                                   total_seconds=elapsed,**metrics)
                        np.save(Path(output)/f"{trial['id']}_{index:03d}_sample.npy",bits)
                    except TimeoutError:
                        row.update(status='timeout',total_seconds=time.perf_counter()-started,**metrics)
                    except Exception:
                        row.update(status='error',error=traceback.format_exc(),total_seconds=time.perf_counter()-started,**metrics)
                    finally:
                        signal.setitimer(signal.ITIMER_REAL,0.)
                    stream.write(json.dumps(row)+'\n'); events.put(row)
                events.put(dict(type='done',gpu=gpu,trial=trial['id']))
    except Exception:
        events.put(dict(type='fatal',gpu=gpu,error=traceback.format_exc()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--pilot-first-eight',action='store_true')
    parser.add_argument('--pilot-full-grid',action='store_true')
    parser.add_argument('--steps',type=int,default=256)
    parser.add_argument('--runs',type=int,default=1)
    parser.add_argument('--budget',type=float,default=300.)
    parser.add_argument('--policy',choices=('default','sac','selected'),default='sac')
    parser.add_argument('--output',default='results')
    args=parser.parse_args()
    output=HERE/args.output; output.mkdir(exist_ok=True)
    indices=(list(range(8)) if args.pilot_first_eight else [0,15,30,45,60,75,90,104]) if args.pilot else list(range(105))
    ctx=mp.get_context('spawn'); events=ctx.Queue()
    inboxes=[ctx.Queue() for _ in range(8)]
    processes=[ctx.Process(target=worker,args=(gpu,indices[gpu::8],inboxes[gpu],events,str(output))) for gpu in range(8)]
    for process in processes: process.start()
    policies=json.loads((HERE/'selected_policies.json').read_text()) if args.policy=='selected' else {name:args.policy for name in SOLVERS}
    trials=([dict(solver=name,variant='pruned',strength='1',policy=policy)
             for policy in ('default','sac') for name in SOLVERS] if args.pilot and not args.pilot_full_grid else
            [dict(solver=name,variant=variant,strength=strength,policy=policies[name])
             for variant,strength in [('legacy','1'),('pruned','1'),('pruned','10'),('pruned','bound')]
             for name in SOLVERS])
    summaries=[]; all_rows=[]
    try:
        ready=set()
        while len(ready)<8:
            event=events.get(timeout=300)
            if event['type']=='fatal': raise RuntimeError(event['error'])
            ready.add(event['gpu'])
        for number,trial in enumerate(trials):
            trial.update(id=f"{number:02d}_{trial['solver']}_{trial['variant']}_{trial['strength']}_{trial['policy']}",
                         steps=args.steps,runs=args.runs,deadline=time.monotonic()+args.budget)
            start=time.monotonic(); rows=[]; done=set()
            atomic(output/'status.json',dict(state='running',trial=trial,completed_trials=summaries))
            for inbox in inboxes: inbox.put(trial)
            while len(done)<8 and time.monotonic()<trial['deadline']:
                try: event=events.get(timeout=min(1.,max(.01,trial['deadline']-time.monotonic())))
                except queue.Empty: continue
                if event['type']=='fatal': raise RuntimeError(event['error'])
                if event['type']=='done': done.add(event['gpu'])
                if event['type']=='result':
                    rows.append(event); all_rows.append(event)
                    atomic(output/'status.json',dict(state='running',trial=trial,finished=len(rows),
                        solved=sum(r['status']=='solved' for r in rows),elapsed_seconds=time.monotonic()-start,
                        completed_trials=summaries))
            wall=time.monotonic()-start
            solved=[r for r in rows if r['status']=='solved']
            summary=dict(trial=trial['id'],solver=trial['solver'],variant=trial['variant'],
                penalty_setting=trial['strength'],policy=trial['policy'],steps=args.steps,runs=args.runs,
                requested=len(indices),solved=len(solved),timeouts=sum(r['status']=='timeout' for r in rows),
                errors=sum(r['status']=='error' for r in rows),wall_seconds=wall,budget_seconds=args.budget,
                complete=len(solved)==len(indices),margin=max((r['loss'] for r in solved),default=None),
                repair_calls=sum(r['repair_calls'] for r in rows),repair_seconds=sum(r['repair_seconds'] for r in rows),
                repair_model_seconds=sum(r['repair_model_seconds'] for r in rows),
                raw_candidates=sum(r['raw_candidates'] for r in rows),raw_feasible=sum(r['raw_feasible'] for r in rows),
                violated_group_fraction=sum(r['violated_groups'] for r in rows)/max(1,sum(r['checked_groups'] for r in rows)))
            summaries.append(summary); atomic(output/'summary.json',summaries); atomic(output/'rows.json',all_rows)
            with (output/'summary.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(summary)); writer.writeheader(); writer.writerows(summaries)
            print(json.dumps(summary),flush=True)
            if len(done)<8:
                raise TimeoutError('Trial exhausted global deadline; terminating only this sweep\'s workers')
        atomic(output/'status.json',dict(state='complete',completed_trials=summaries))
    finally:
        for inbox in inboxes: inbox.put(None)
        for process in processes:
            process.join(timeout=1)
            if process.is_alive(): process.terminate()
        for process in processes: process.join(timeout=2)


if __name__=='__main__': main()
