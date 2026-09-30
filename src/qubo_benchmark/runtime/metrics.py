"""Independent exact scoring and observed target times. No optimizer imports."""
import hashlib
from ..model import binaryVector

def quality(value, reference, proven=False):
    gap=value-reference
    alarm=bool(proven and gap<0)
    threshold=reference+.01*abs(reference) if reference else 0
    return dict(signed_gap=gap,signed_gap_percent=100*gap/abs(reference) if reference else None,
                near_target_objective=threshold,reference_beaten=gap<0,
                optimum_validation_alarm=alarm,success_1pct=value<=threshold and not alarm,
                success_reference=value<=reference and not alarm,reference_equal=value==reference,
                exact_optimum_reached=(value==reference and not alarm) if proven else None)

def evaluate(problem, events, budget, reference, status, error=None):
    proven=status=='published_proven_optimum'
    solutions=[];trace=[];best=None;first=None;best_time=None;best_id=None
    hits=dict(ttt_1pct_s=None,ttt_reference_s=None,ttt_optimum_s=None)
    invalid=False;alarm=False;previous=-1.;last=None;raw=[];final=None;initial=None
    for index,event in enumerate(events):
        elapsed=event['elapsed_s']
        if elapsed<previous:raise ValueError('Nonmonotonic observation timestamps')
        previous=elapsed
        text=event['bitstring'];valid=True
        try:
            if not isinstance(text,str) or any(c not in '01' for c in text):raise ValueError('Nonbinary bitstring')
            vector=binaryVector([int(c) for c in text],problem.n)
            score=problem.score(vector)
        except (ValueError,TypeError):
            valid=False;score=None;invalid=True
        eligible=bool(valid and 0<=elapsed<=budget)
        sid=f'c{index:08d}'
        q=quality(score,reference,proven) if valid else {}
        alarm |= bool(eligible and q.get('optimum_validation_alarm'))
        if eligible:
            first=elapsed if first is None else first
            if best is None or score<best:best,best_time,best_id=score,elapsed,sid
            if not q['optimum_validation_alarm']:
                for key,condition in [('ttt_1pct_s',q['success_1pct']),('ttt_reference_s',q['success_reference']),
                                      ('ttt_optimum_s',q['exact_optimum_reached'])]:
                    if condition and hits[key] is None:hits[key]=elapsed
        role=event.get('phase','checkpoint')
        if role=='final_returned':final=dict(objective=score,within=eligible,id=sid)
        if role=='initial' and initial is None:initial=sid
        solution=dict(solution_id=sid,role=role,n_variables=problem.n,index_mapping='bit i = original index i (source label i+1)',
            bitstring=text,independent_objective=score,completed_elapsed_s=elapsed,within_budget=eligible,valid=valid,
            solution_sha256=hashlib.sha256(text.encode('ascii',errors='replace')).hexdigest())
        solutions.append(solution)
        trace.append(dict(event_index=index,solution_id=sid,elapsed_s=elapsed,iteration=event.get('iteration'),
            evaluation_count=None,phase=role,solver_reported_energy=event.get('raw'),
            independently_evaluated_objective=score,independently_validated_best_so_far=best,
            signed_gap=q.get('signed_gap'),signed_gap_percent=q.get('signed_gap_percent'),
            near_target_hit=eligible and q.get('success_1pct',False),
            reference_target_hit=eligible and q.get('success_reference',False),within_budget=eligible,
            timestamp_kind='host_observed_after_immutable_capture',valid=valid))
        if isinstance(event.get('raw'),(int,float)):raw.append(event['raw'])
        last=elapsed
    history=any(e.get('phase')!='final_returned' for e in events)
    failure=error is not None or invalid or alarm
    result_status=(error or ('invalid_output' if invalid else 'validation_alarm' if alarm else
                            'completed' if best is not None else 'no_in_budget_candidate'))
    q=quality(best,reference,proven) if best is not None else quality(reference,reference,proven)
    if best is None:
        q.update(signed_gap=None,signed_gap_percent=None,reference_beaten=False,success_1pct=False,
                 success_reference=False,reference_equal=False,exact_optimum_reached=False if proven else None)
    if failure:
        q.update(success_1pct=False,success_reference=False,exact_optimum_reached=False if proven else None)
    if failure or not history:
        hits={k:None for k in hits}
    q['optimum_validation_alarm']=alarm
    reason=('invalid_run' if failure else 'no_in_budget_candidate' if best is None else
            'history_unavailable' if not history else 'not_reached')
    record=dict(q,**hits,best_objective_within_budget=best,raw_binary_min_objective=best,
        solver_reported_raw_min=min(raw) if raw else None,final_returned_objective=final['objective'] if final else None,
        final_returned_within_budget=final['within'] if final else None,postprocessed_objective=None,
        candidate_valid=not invalid if events else None,in_budget_score_available=best is not None,
        result_status=result_status,score_audit_status='failed' if invalid or alarm else 'passed',
        history_available=history,history_mode='observed_checkpoints' if history else 'final_return_only',
        time_to_first_candidate_s=first,time_to_best_s=best_time,last_candidate_elapsed_s=last,
        near_target_event=hits['ttt_1pct_s'] is not None,reference_target_event=hits['ttt_reference_s'] is not None,
        near_target_time_status='observed' if hits['ttt_1pct_s'] is not None else reason,
        reference_time_status='observed' if hits['ttt_reference_s'] is not None else reason,
        optimum_time_status=('not_applicable' if not proven else 'observed' if hits['ttt_optimum_s'] is not None else reason),
        initial_solution_id=initial,best_solution_id=best_id,final_solution_id=final['id'] if final else None,
        solution_sha256=next((s['solution_sha256'] for s in solutions if s['solution_id']==best_id),None),
        trace_event_count=len(trace),trace_truncated=False)
    if best_id is not None:
        selected=dict(next(s for s in solutions if s['solution_id']==best_id),role='best_in_budget',solution_id='best')
        solutions.append(selected);record['best_solution_id']='best'
    return record,solutions,trace
