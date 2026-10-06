"""Objective-only storage parity and one-state temporal sampling contracts."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from qubo_benchmark.model import Problem
from qubo_benchmark.runtime.continuous_worker import ObjectiveScorer, ObjectiveValidationError, ScalarObserver, CHECKPOINTS, run_continuous_trial
from qubo_benchmark.runtime.compact_metrics import aggregate_energies
from qubo_benchmark.runtime.worker import Observer
from qubo_solvers.observation import SolveInterrupted, observing, capture
from qubo_solvers.backends.problem import QUBOProblem
from qubo_solvers.backends.torch_candidates import TorchCandidateAccumulator


def source():
    return Problem(2,np.array([0,1]),np.array([0,1]),np.array([-10,-1]),3)


def test_original_source_scoring_does_not_trust_reported_energy():
    scorer=ObjectiveScorer(source(),'cpu')
    assert float(scorer.score(torch.tensor([[1,1]]),torch.tensor([999.])))==-8
    assert scorer.matrix.dtype==torch.float32  # Proven integer mantissa range.


def test_large_integer_scores_use_float64_and_preserve_offset():
    problem=Problem(2,np.array([0,1]),np.array([0,1]),np.array([100000001,-100000000]),7)
    scorer=ObjectiveScorer(problem,'cpu')
    assert scorer.matrix.dtype==torch.float64
    assert float(scorer.score(torch.ones((1,2))))==8


def test_late_capture_does_not_retroactively_improve_checkpoint():
    clock=[0]
    observer=ScalarObserver(ObjectiveScorer(source(),'cpu'),clock=lambda:clock[0])
    clock[0]=10_000_000
    observer.capture(torch.tensor([[0,0]]),torch.tensor([-999.]))
    clock[0]=60_000_000
    observer.capture(torch.tensor([[1,0]]))
    assert observer.energies[0]==3
    clock[0]=110_000_000
    observer.poll()
    assert observer.energies[1]==-7
    assert np.isnan(observer.energies[2:]).all()


def test_deadline_stores_only_last_completed_source_value():
    clock=[0]
    observer=ScalarObserver(ObjectiveScorer(source(),'cpu'),clock=lambda:clock[0])
    clock[0]=10_000_000
    observer.capture(torch.tensor([[1,0]]))
    clock[0]=20_100_000_000
    with pytest.raises(SolveInterrupted,match='deadline'):
        observer.capture(torch.tensor([[1,1]]))
    np.testing.assert_array_equal(observer.finish(),np.full(9,-7.))
    assert not observer.preparation


def test_minimal_energies_reproduce_legacy_vector_trace_metrics_exactly():
    clock=[0];events=[]
    old=Observer(20.,lambda message:events.extend(message.get('events',[])),clock=lambda:clock[0])
    new=ScalarObserver(ObjectiveScorer(source(),'cpu'),clock=lambda:clock[0])
    for seconds,bits in [(.01,[0,0]),(.08,[1,0]),(.8,[1,1])]:
        clock[0]=int(seconds*1e9)
        sample=torch.tensor([bits])
        old.capture(sample,torch.tensor([999.]))
        new.capture(sample,torch.tensor([999.]))
    clock[0]=20_000_000_000
    expected=[]
    for checkpoint in CHECKPOINTS:
        eligible=[source().score([int(v) for v in e['bitstring']]) for e in events
                  if e['elapsed_s']<=checkpoint]
        expected.append(min(eligible) if eligible else np.nan)
    np.testing.assert_array_equal(new.finish(),expected)
    assert aggregate_energies([expected],CHECKPOINTS,-8,'BKS')==aggregate_energies([new.energies],CHECKPOINTS,-8,'BKS')


def test_scalar_accumulator_avoids_host_candidate_capture():
    problem=QUBOProblem(linear=np.array([-10.,-1.]),quadraticHeads=np.array([],dtype=np.uint32),
                       quadraticTails=np.array([],dtype=np.uint32),quadraticBiases=np.array([]),offset=3)
    clock=[0]
    observer=ScalarObserver(ObjectiveScorer(source(),'cpu',native=False),clock=lambda:clock[0])
    accumulator=TorchCandidateAccumulator(torch,problem,'cpu',8,None)
    with observing(observer):
        clock[0]=10_000_000
        accumulator.add(torch.tensor([[1,0],[1,1]],dtype=torch.bool))
        assert accumulator.best is None  # Old host-vector path was not used.
        assert accumulator.result()==((1,1),-8.)
    clock[0]=20_000_000_000
    np.testing.assert_array_equal(observer.finish(),np.full(9,-8.))


def test_schedule_uses_wall_clock_only_when_explicitly_enabled():
    clock=[0]
    observer=ScalarObserver(ObjectiveScorer(source(),'cpu'),clock=lambda:clock[0])
    clock[0]=10_000_000_000
    assert observer.schedule_fraction(.2)==.2
    observer.wall_schedule=True
    assert observer.schedule_fraction(.2)==.5


@pytest.mark.parametrize('tensor', [False, True])
def test_optimum_before_first_checkpoint_stops_and_fills_without_waiting(monkeypatch, tensor):
    clock = [0]
    observer = ScalarObserver(ObjectiveScorer(source(), 'cpu'), clock=lambda: clock[0],
                              reference=-8, reference_type='OPTIMUM')
    clock[0] = 10_000_000
    with pytest.raises(SolveInterrupted, match='optimum'):
        if tensor:
            observer.capture(torch.tensor([[1, 1]]), torch.tensor([999.]))
        else:
            observer.capture_objectives(-8.)
    monkeypatch.setattr('qubo_benchmark.runtime.continuous_worker.time.sleep',
                        lambda delay: pytest.fail('optimum finish must not sleep'))
    np.testing.assert_array_equal(observer.finish(), np.full(9, -8.))
    assert observer.time_to_optimum_s == .01 and observer.elapsed() == .01


@pytest.mark.parametrize('initial', [None, 3.])
def test_optimum_between_checkpoints_preserves_earlier_checkpoint(initial):
    clock = [0]
    observer = ScalarObserver(ObjectiveScorer(source(), 'cpu'), clock=lambda: clock[0],
                              reference=-8, reference_type='OPTIMUM')
    if initial is not None:
        clock[0] = 10_000_000
        observer.capture_objectives(initial)
    clock[0] = 80_000_000
    with pytest.raises(SolveInterrupted, match='optimum'):
        observer.capture_objectives(-8.)
    expected = [np.nan if initial is None else initial] + [-8.] * 8
    np.testing.assert_array_equal(observer.finish(), expected)
    assert observer.time_to_optimum_s == .08


def test_optimum_time_includes_scalar_copy_and_never_back_credits(monkeypatch):
    clock = [0]
    observer = ScalarObserver(ObjectiveScorer(source(), 'cpu'), clock=lambda: clock[0],
                              reference=-8, reference_type='OPTIMUM')
    clock[0] = 10_000_000
    observer.capture_objectives(3.)
    original = torch.Tensor.cpu
    def completed_copy(value, *args, **kwargs):
        clock[0] = 60_000_000
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', completed_copy)
    with pytest.raises(SolveInterrupted, match='optimum'):
        observer.capture(torch.tensor([[1, 1]]))
    np.testing.assert_array_equal(observer.finish(), [3.] + [-8.] * 8)
    assert observer.time_to_optimum_s == .06


def test_candidate_at_checkpoint_boundary_does_not_revise_recorded_value():
    clock = [0]
    observer = ScalarObserver(ObjectiveScorer(source(), 'cpu'), clock=lambda: clock[0],
                              reference=-8, reference_type='OPTIMUM')
    clock[0] = 10_000_000
    observer.capture_objectives(3.)
    clock[0] = 100_000_000
    with pytest.raises(SolveInterrupted, match='optimum'):
        observer.capture_objectives(-8.)
    np.testing.assert_array_equal(observer.finish(), [3., 3.] + [-8.] * 7)
    assert observer.time_to_optimum_s == .1


def test_scalar_validation_finishing_after_deadline_cannot_stop_on_optimum(monkeypatch):
    clock = [0]
    observer = ScalarObserver(ObjectiveScorer(source(), 'cpu'), clock=lambda: clock[0],
                              reference=-8, reference_type='OPTIMUM')
    clock[0] = 10_000_000
    observer.capture_objectives(3.)
    original = torch.Tensor.cpu
    def late_copy(value, *args, **kwargs):
        clock[0] = 20_100_000_000
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', late_copy)
    with pytest.raises(SolveInterrupted, match='deadline'):
        observer.capture(torch.tensor([[1, 1]]))
    np.testing.assert_array_equal(observer.finish(), np.full(9, 3.))
    assert observer.time_to_optimum_s is None


def test_bks_hit_and_improvement_continue_and_never_record_optimum_time():
    clock = [0]
    observer = ScalarObserver(ObjectiveScorer(source(), 'cpu'), clock=lambda: clock[0],
                              reference=-7, reference_type='BKS')
    clock[0] = 10_000_000
    observer.capture_objectives(-7.)
    clock[0] = 80_000_000
    observer.capture(torch.tensor([[1, 1]]))
    assert observer.time_to_optimum_s is None and observer.cursor == 1
    clock[0] = 20_000_000_000
    np.testing.assert_array_equal(observer.finish(), [-7.] + [-8.] * 8)


def test_suboptimum_score_raises_validation_alarm_instead_of_hit():
    clock = [0]
    observer = ScalarObserver(ObjectiveScorer(source(), 'cpu'), clock=lambda: clock[0],
                              reference=-7, reference_type='OPTIMUM')
    clock[0] = 10_000_000
    with pytest.raises(ObjectiveValidationError, match='below the proven optimum'):
        observer.capture(torch.tensor([[1, 1]]), torch.tensor([-7.]))
    assert observer.certified is None and observer.time_to_optimum_s is None


def test_optimum_tolerance_and_first_hit_time_survive_disabled_early_stop():
    clock = [0]
    observer = ScalarObserver(ObjectiveScorer(source(), 'cpu'), clock=lambda: clock[0],
                              reference=-8, reference_type='OPTIMUM', tolerance=.1,
                              stop_on_optimum=False)
    clock[0] = 10_000_000
    observer.capture_objectives(-7.95)
    clock[0] = 80_000_000
    observer.capture_objectives(-8.05)
    clock[0] = 20_000_000_000
    np.testing.assert_array_equal(observer.finish(), [-7.95] + [-8.05] * 8)
    assert observer.time_to_optimum_s == .01


@pytest.mark.parametrize('graph_stream', [None, object()])
def test_trial_optimum_uses_job_metadata_and_returns_before_budget(monkeypatch, graph_stream):
    import qubo_solvers
    from qubo_solvers.backends.result import BQMOptimizationResult
    from qubo_solvers.observation import current
    worker = SimpleNamespace(torch=torch, device='cpu', cache_key='optimum', native=False,
                             source=source(), compact=object(), trials=0)
    if graph_stream is not None:
        worker._continuous_graph_stream = graph_stream
    class Backend:
        def solve(self, problem, parameters):
            assert current().cuda_graph_stream is graph_stream
            assert not {'reference', 'reference_type', 'tolerance', 'stop_on_optimum'} & parameters.keys()
            return BQMOptimizationResult((1, 1), 999.)
    monkeypatch.setattr(qubo_solvers, 'create_bqm_solver', lambda name, parameters: Backend())
    job = dict(solver='lib_altermagnet', parameters={'dtype': 'float32'}, seed=0,
               reference=-8, reference_type='OPTIMUM', tolerance=0.)
    result = run_continuous_trial(worker, job)
    assert result['status'] == 'complete' and result['stop_reason'] == 'optimum'
    assert result['energies'] == [-8.] * 9
    assert 0 <= result['time_to_optimum_s'] <= result['actual_solve_wall_s'] < 1.
    assert worker.trials == 1
    alarm = run_continuous_trial(worker, dict(job, reference=-7))
    assert alarm['status'] == alarm['stop_reason'] == 'validation_error'
    assert alarm['time_to_optimum_s'] is None
    assert 'below the proven optimum' in alarm['error']


def test_failed_trial_clears_a_previously_observed_optimum_time(monkeypatch):
    import qubo_solvers
    from qubo_solvers.observation import current
    worker = SimpleNamespace(torch=torch, device='cpu', cache_key='failed optimum', native=False,
                             source=source(), compact=object(), trials=0)
    class Backend:
        def solve(self, problem, parameters):
            current().capture_objectives(-8.)
            assert current().time_to_optimum_s is not None
            raise ValueError('failure after first hit')
    monkeypatch.setattr(qubo_solvers, 'create_bqm_solver', lambda name, parameters: Backend())
    result = run_continuous_trial(worker, dict(solver='lib_altermagnet', parameters={'dtype': 'float32'},
        seed=0, reference=-8, reference_type='OPTIMUM', stop_on_optimum=False))
    assert result['status'] == 'error' and result['time_to_optimum_s'] is None
    assert 'failure after first hit' in result['error']


def test_compact_to_sbm_cache_transition_scores_direct_device_capture(monkeypatch):
    """SBM's existing snapshot hook needs an operator even on compact inputs."""
    from qubo_solvers.backends.result import BQMOptimizationResult
    import qubo_solvers

    problem = Problem(2, np.array([0, 1, 0]), np.array([0, 1, 1]),
                      np.array([-4, -1, 3]), 7)
    compact = QUBOProblem([-4., -1.], [0], [1], [6.], offset=7.)
    worker = SimpleNamespace(torch=torch, device='cpu', cache_key='same compact input',
                             native=False, source=problem, compact=compact, trials=0)

    class Backend:
        def __init__(self, name):
            self.name = name

        def solve(self, source, parameters):
            if self.name == 'lib_simulated_bifurcation':
                # Deliberately false reported energy. Original offset and the
                # pair's symmetric factor determine the observed objective.
                capture(torch.tensor([[1, 1]]), torch.tensor([-999.]))
                return BQMOptimizationResult((1, 1), 8.)
            accumulator = TorchCandidateAccumulator(torch, source, 'cpu', 8, None)
            accumulator.add(torch.tensor([[1, 0], [0, 1]], dtype=torch.bool))
            return BQMOptimizationResult(*accumulator.result())

    monkeypatch.setattr(qubo_solvers, 'create_bqm_solver', lambda name, parameters: Backend(name))
    job = dict(solver='lib_altermagnet', parameters={'dtype': 'float32'},
               seed=0, budget=.02, checkpoints=(.02,))
    compact_result = run_continuous_trial(worker, job)
    assert compact_result['status'] == 'complete' and compact_result['error'] is None
    assert compact_result['energies'] == [3.]
    assert worker._continuous_scorer.matrix is None
    compact_key = worker._continuous_score_key
    sbm_result = run_continuous_trial(worker, dict(job, solver='lib_simulated_bifurcation'))
    assert sbm_result['status'] == 'complete' and sbm_result['error'] is None
    assert sbm_result['energies'] == [8.]
    assert worker._continuous_scorer.matrix is not None
    assert worker._continuous_score_key != compact_key


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gpu_scalar_copies_have_constant_dimension(monkeypatch):
    clock=[0]
    observer=ScalarObserver(ObjectiveScorer(source(),'cuda:0'),clock=lambda:clock[0])
    copies=[]
    original=torch.Tensor.cpu
    def checked(value,*args,**kwargs):
        copies.append(value.numel())
        return original(value,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'cpu',checked)
    observer.capture(torch.ones((16,2),device='cuda:0'))
    observer._checkpoint(.05)
    assert copies and set(copies)=={1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_gpu_optimum_stop_copies_only_scalar_after_completed_scoring(monkeypatch):
    clock = [0]
    observer = ScalarObserver(ObjectiveScorer(source(), 'cuda:0'), clock=lambda: clock[0],
                              reference=-8, reference_type='OPTIMUM')
    copies = []
    original = torch.Tensor.cpu
    def checked(value, *args, **kwargs):
        copies.append(value.numel())
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', checked)
    clock[0] = 10_000_000
    with pytest.raises(SolveInterrupted, match='optimum'):
        observer.capture(torch.ones((16, 2), device='cuda:0'), torch.full((16,), 999., device='cuda:0'))
    np.testing.assert_array_equal(observer.finish(), np.full(9, -8.))
    assert copies and set(copies) == {1}
    assert observer.time_to_optimum_s == .01
