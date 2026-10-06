"""Quality-aware admission and calibration without running optimization."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from qubo_benchmark.runtime.continuous_calibration import (
    GIB, admission_capacity, calibrate_solver, calibration_cache_key,
    calibration_options, calibration_seeds, compare_quality, estimate_worker_memory)


SOLVER = 'lib_exchange_cascade'


def prepared(devices=('cuda:0',)):
    return dict(settings={SOLVER: dict(dtype='float32', runs=16, run_batch_size=8)},
        seeds=list(range(100)), entry=dict(binary_variables=200,
            nonzero_offdiagonal_pairs=1000, actual_density_percent=5.),
        manifest=dict(source={'snapshot': 'source-hash'}, configuration='configuration-hash',
            inputs={'problem': 'input-hash'}, catalog='catalog-hash',
            problems=[dict(id='problem', n=200, reference=100., reference_type='BKS')],
            environment={'hardware_id': 'host-hash', 'packages': {'torch': 'version'}}),
        environment={'hardware': {'gpus': [dict(device=device, uuid=f'GPU-{index}')
                         for index,device in enumerate(devices)]}},
        gpu_available_bytes={device: 24*GIB for device in devices},
        host_available_bytes=128*GIB, cpu_allowed=16,
        capacity=dict(workers_per_device={device: 4 for device in devices},
                      maximum_workers_estimate=4*len(devices)))


class FakeRunner:
    def __init__(self, transform=None, rate=None):
        self.calls = []
        self.transform = transform
        self.rate = rate or (lambda device,count,seeds: count)

    def __call__(self, solver, device, count, seeds, calibration=False):
        assert calibration and solver == SOLVER
        self.calls.append((device, count, tuple(seeds)))
        rows = [dict(seed=seed, status='complete', energies=[110.,108.,106.,104.,102.,101.,100.,100.,100.],
            actual_solve_wall_s=20., time_to_optimum_s=None,
            gpu_reserved_peak_bytes=512*1024**2, host_rss_peak_bytes=512*1024**2)
            for seed in seeds]
        if self.transform:
            self.transform(device, count, seeds, rows)
        return rows, len(seeds)/self.rate(device,count,seeds)


def test_highest_integer_candidate_and_disjoint_fresh_holdout():
    runner = FakeRunner()
    counts,report = calibrate_solver(prepared(), SimpleNamespace(), SOLVER, ['cuda:0'], runner)
    assert counts == {'cuda:0': 4}
    assert [count for _,count,_ in runner.calls] == [1,2,3,4,1,4]
    screening,holdout = report['screening_seeds'],report['holdout_seeds']
    assert not set(screening) & set(holdout)
    assert not (set(screening) | set(holdout)) & set(range(100))
    assert all(seeds == tuple(screening) for _,_,seeds in runner.calls[:4])
    assert all(seeds == tuple(holdout) for _,_,seeds in runner.calls[4:])
    assert report['options']['budget_seconds'] == 20.
    assert 'statistical guarantee' in report['limitation']
    json.dumps(report, allow_nan=False)


def test_early_quality_regression_rejects_even_equal_final_and_fast_throughput():
    def regress(device,count,seeds,rows):
        if count > 1:
            for row in rows:
                row['energies'][0] += 1.
    counts,report = calibrate_solver(prepared(), SimpleNamespace(), SOLVER, ['cuda:0'], FakeRunner(regress))
    assert counts == {'cuda:0': 1}
    assert 'paired_incumbent_regression' in report['devices']['cuda:0']['screening'][0]['comparison']['reasons']
    assert report['devices']['cuda:0']['holdout'] is None


def test_holdout_failure_falls_to_one_without_reselecting_screening_runner_up():
    screen,held = calibration_seeds(range(100))
    def regress(device,count,seeds,rows):
        if seeds == held and count == 4:
            rows[0]['energies'][-1] = 100.5
    runner = FakeRunner(regress)
    counts,report = calibrate_solver(prepared(), SimpleNamespace(), SOLVER, ['cuda:0'], runner)
    assert counts == {'cuda:0': 1}
    assert [count for _,count,seeds in runner.calls if seeds == tuple(held)] == [1,4]
    assert report['devices']['cuda:0']['reason'] == 'holdout_rejected_fallback_one'


def test_unhelpful_extra_memory_does_not_select_higher_count():
    runner = FakeRunner(rate=lambda device,count,seeds: 1.09 if count > 1 else 1.)
    counts,report = calibrate_solver(prepared(), SimpleNamespace(), SOLVER, ['cuda:0'], runner)
    assert counts['cuda:0'] == 1
    assert all('insufficient_throughput_gain' in result['comparison']['reasons']
               for result in report['devices']['cuda:0']['screening'])


def test_ten_percent_minimum_is_inclusive():
    runner = FakeRunner(rate=lambda device,count,seeds: 1.1 if count == 2 else 1.)
    assert calibrate_solver(prepared(), SimpleNamespace(), SOLVER, ['cuda:0'], runner)[0]['cuda:0'] == 2


def test_worker_counts_are_selected_independently_on_each_gpu():
    runner = FakeRunner(rate=lambda device,count,seeds: count if device == 'cuda:0' else 1.)
    counts,_ = calibrate_solver(prepared(('cuda:0','cuda:1')), SimpleNamespace(), SOLVER,
                              ['cuda:0','cuda:1'], runner)
    assert counts == {'cuda:0': 4, 'cuda:1': 1}


@pytest.mark.parametrize('failure', ['missing_row', 'duplicate_row', 'status', 'missing_final'])
def test_missing_or_failed_attempt_cannot_admit_concurrency(failure):
    def fail(device,count,seeds,rows):
        if count == 1:
            return
        if failure == 'missing_row':
            rows.pop()
        elif failure == 'duplicate_row':
            rows[-1] = deepcopy(rows[0])
        elif failure == 'status':
            rows[0]['status'] = 'oom'
        else:
            rows[0]['energies'][-1] = None
    assert calibrate_solver(prepared(), SimpleNamespace(), SOLVER, ['cuda:0'], FakeRunner(fail))[0]['cuda:0'] == 1


def test_shared_unobserved_early_checkpoint_is_labeled_and_does_not_block():
    def missing(device,count,seeds,rows):
        for row in rows:
            row['energies'][0] = None
    counts,report = calibrate_solver(prepared(), SimpleNamespace(), SOLVER, ['cuda:0'], FakeRunner(missing))
    assert counts['cuda:0'] == 4
    checkpoint = report['devices']['cuda:0']['screening'][0]['comparison']['checkpoints'][0]
    assert not checkpoint['comparable'] and checkpoint['unobserved_both_seeds'] == 4
    assert checkpoint['baseline_mean_signed_gap'] is None


def test_new_missing_candidate_at_observed_early_checkpoint_rejects():
    def missing(device,count,seeds,rows):
        if count > 1:
            rows[0]['energies'][0] = None
    counts,report = calibrate_solver(prepared(), SimpleNamespace(), SOLVER, ['cuda:0'], FakeRunner(missing))
    assert counts['cuda:0'] == 1
    assert 'missing_observed_comparable' in report['devices']['cuda:0']['screening'][0]['comparison']['reasons']


def test_improved_early_observability_can_pass():
    def improved(device,count,seeds,rows):
        if count == 1:
            rows[0]['energies'][0] = None
    assert calibrate_solver(prepared(), SimpleNamespace(), SOLVER, ['cuda:0'], FakeRunner(improved))[0]['cuda:0'] == 4


def test_unavailable_baseline_final_leaves_one_and_reports_uncertainty():
    def missing(device,count,seeds,rows):
        for row in rows:
            row['energies'] = [None]*9
    runner = FakeRunner(missing)
    counts,report = calibrate_solver(prepared(), SimpleNamespace(), SOLVER, ['cuda:0'], runner)
    assert counts['cuda:0'] == 1 and len(runner.calls) == 1
    assert report['devices']['cuda:0']['reason'] == 'baseline_quality_unavailable'


@pytest.mark.parametrize(('before','after','reason'), [
    ([100.,130.], [100.5,120.], 'exact_hit_regression'),
    ([101.,130.], [102.,120.], 'one_percent_hit_regression'),
    ([110.,110.], [111.,110.], 'mean_signed_gap_regression')])
def test_independent_quality_gates_cannot_be_hidden_by_averaging(before,after,reason):
    def wave(values):
        return dict(errors=[], rows=[dict(seed=index, energies=[value]*9)
                                      for index,value in enumerate(values)])
    result = compare_quality(wave(before), wave(after), [0,1], 100., strict_pairs=False)
    assert not result['accepted'] and reason in result['reasons']


def test_optimum_alarm_uses_shared_reference_bks_improvements_remain_valid():
    wave = dict(errors=[], rows=[dict(seed=0,energies=[99.]*9)])
    assert compare_quality(wave,wave,[0],100.,reference_type='BKS')['accepted']
    assert compare_quality(wave,wave,[0],100.,reference_type='OPTIMUM')['reasons'] == ['below_proven_optimum']


def test_cpu_and_disabled_autotune_never_run_calibration():
    def forbidden(*args,**kwargs):
        raise AssertionError('must not execute a wave')
    assert calibrate_solver(prepared(('cpu',)), SimpleNamespace(), SOLVER, ['cpu'], forbidden)[0] == {'cpu':1}
    assert calibrate_solver(prepared(), SimpleNamespace(no_autotune=True), SOLVER, ['cuda:0'], forbidden)[0] == {'cuda:0':1}


def test_sample_options_do_not_silently_expand_expensive_calibration():
    with pytest.raises(ValueError,match='cover max_auto_workers'):
        calibration_options(SimpleNamespace(max_auto_workers=8))
    result = calibration_options(SimpleNamespace(max_auto_workers=8,calibration_seeds=8,
                                                  calibration_holdout_seeds=8))
    assert result['screening_seeds'] == 8
    with pytest.raises(ValueError,match='positive integer'):
        calibration_options(SimpleNamespace(max_auto_workers=0))
    with pytest.raises(ValueError,match='at least 1.1'):
        calibration_options(SimpleNamespace(calibration_min_speedup=1.01))


def test_seed_generation_skips_nonstandard_campaign_seeds():
    screening,holdout = calibration_seeds([2**31,2**31+3])
    assert screening == [2**31+1,2**31+2,2**31+4,2**31+5]
    assert len(set(screening+holdout)) == 8


def test_cache_key_tracks_frozen_identities_and_options_but_not_free_memory():
    original = prepared()
    options = calibration_options(SimpleNamespace())
    expected = calibration_cache_key(original,SOLVER,['cuda:0'],options)
    variations = []
    for field in ('source','inputs','configuration'):
        changed = deepcopy(original)
        changed['manifest'][field] = 'changed'
        variations.append(changed)
    changed = deepcopy(original)
    changed['settings'][SOLVER]['runs'] += 1
    variations.append(changed)
    changed = deepcopy(original)
    changed['environment']['hardware']['gpus'][0]['uuid'] = 'different GPU'
    variations.append(changed)
    assert all(calibration_cache_key(value,SOLVER,['cuda:0'],options) != expected for value in variations)
    changed = deepcopy(original)
    changed['gpu_available_bytes']['cuda:0'] = GIB
    assert calibration_cache_key(changed,SOLVER,['cuda:0'],options) == expected
    assert calibration_cache_key(original,SOLVER,['cuda:0'],dict(options,strict_pairs=False)) != expected


def test_cache_key_binds_semantic_execution_but_excludes_post_calibration_worker_plan():
    value = prepared()
    value['manifest']['execution'] = dict(stop_on_optimum=True,budget_seconds=20.,
        cpu_threads=1,tf32=False,checkpoint_policy='completed independent score',
        natural_return_policy='carry result',optimum_time_policy='observed completion',
        bks_policy='continue',wall_schedule_solvers=[SOLVER],scheduling_policy='homogeneous')
    options = calibration_options(SimpleNamespace())
    expected = calibration_cache_key(value,SOLVER,['cuda:0'],options)
    changed = deepcopy(value)
    changed['manifest']['execution']['stop_on_optimum'] = False
    assert calibration_cache_key(changed,SOLVER,['cuda:0'],options) != expected
    changed = deepcopy(value)
    changed['manifest']['execution']['worker_plan'] = {SOLVER:{'cuda:0':4}}
    changed['manifest']['execution']['calibration_evidence'] = 'report-file'
    assert calibration_cache_key(changed,SOLVER,['cuda:0'],options) == expected


def test_memory_estimate_accounts_for_independent_fp64_original_score_and_large_padding():
    parameters = dict(dtype='float32',runs=16,run_batch_size=8)
    estimate = estimate_worker_memory(10000,50000000,parameters,True,SOLVER,
                                       internal_coordinates=16384)
    assert estimate['internal_coordinates'] == 16384
    assert estimate['search_matrix_bytes'] == 6*4*16384**2
    assert estimate['independent_scoring_bytes'] == 8*10000**2
    assert estimate['gpu_per_worker_estimate'] > estimate['search_matrix_bytes']+8*10000**2+GIB
    assert estimate['memory_fraction'] == .8
    with pytest.raises(ValueError,match='truncate'):
        estimate_worker_memory(10000,1,parameters,True,SOLVER,internal_coordinates=9999)


def test_physics_padding_and_sparse_score_storage_follow_actual_dimensions():
    parameters = dict(dtype='float32',runs=16,run_batch_size=8)
    assert estimate_worker_memory(200,1000,parameters,False,'lib_altermagnet')['internal_coordinates'] == 225
    estimate = estimate_worker_memory(200,1000,parameters,True,SOLVER)
    assert estimate['independent_scoring_bytes'] == 16*2000+8*201


def test_admission_accounts_for_shared_host_memory_and_cpu_across_gpus():
    value = prepared(('cuda:0','cuda:1'))
    value.pop('capacity')
    value['cpu_allowed'] = 5
    capacity = admission_capacity(value,SOLVER,['cuda:0','cuda:1'])
    assert capacity['workers_per_device'] == {'cuda:0':2,'cuda:1':2}
    assert capacity['maximum_workers_estimate'] == 4
    assert capacity['cpu_worker_limit'] == 5
    value['gpu_available_bytes']['cuda:0'] = 0
    assert admission_capacity(value,SOLVER,['cuda:0','cuda:1'])['workers_per_device']['cuda:0'] == 0


def test_cpu_admission_counts_tensor_and_scoring_storage_in_host_memory():
    value = prepared(('cpu',))
    value.pop('capacity')
    cpu = admission_capacity(value,SOLVER,['cpu'])
    value['gpu_available_bytes'] = {'cuda:0':24*GIB}
    gpu = admission_capacity(value,SOLVER,['cuda:0'])
    assert cpu['host_per_worker_estimate'] == gpu['host_per_worker_estimate']+gpu['gpu_per_worker_estimate']-GIB


def test_actual_baseline_memory_caps_candidates_before_launch():
    value = prepared()
    value['capacity'].update(host_available_bytes=128*GIB,
        gpu_available_bytes={'cuda:0':24*GIB},device_count=1,
        gpu_per_worker_estimate=2*GIB,host_per_worker_estimate=2*GIB)
    def memory(device,count,seeds,rows):
        for row in rows:
            row['gpu_reserved_peak_bytes'] = 18*GIB
    runner = FakeRunner(memory)
    counts,report = calibrate_solver(value,SimpleNamespace(),SOLVER,['cuda:0'],runner)
    assert counts['cuda:0'] == 1 and len(runner.calls) == 1
    assert report['devices']['cuda:0']['admitted_maximum'] == 1


@pytest.mark.parametrize('field', ['gpu_reserved_peak_bytes','host_rss_peak_bytes'])
def test_missing_baseline_peak_measurement_keeps_one(field):
    def missing(device,count,seeds,rows):
        rows[0][field] = None
    runner = FakeRunner(missing)
    counts,report = calibrate_solver(prepared(),SimpleNamespace(),SOLVER,['cuda:0'],runner)
    assert counts['cuda:0'] == 1 and len(runner.calls) == 1
    assert report['devices']['cuda:0']['reason'] == 'baseline_memory_measurement_unavailable'


def test_known_zero_peaks_are_distinct_from_missing_measurements():
    def zero(device,count,seeds,rows):
        for row in rows:
            row['gpu_reserved_peak_bytes'] = row['host_rss_peak_bytes'] = 0
    counts,report = calibrate_solver(prepared(),SimpleNamespace(),SOLVER,['cuda:0'],FakeRunner(zero))
    assert counts['cuda:0'] == 4
    assert report['devices']['cuda:0']['baseline_memory_peaks'] == dict(gpu=0.,host=0.,available=True)


def test_runner_failure_records_error_and_leaves_baseline_count():
    def broken(*args,**kwargs):
        raise RuntimeError('owned worker failed')
    counts,report = calibrate_solver(prepared(),SimpleNamespace(),SOLVER,['cuda:0'],broken)
    assert counts['cuda:0'] == 1
    assert report['devices']['cuda:0']['baseline']['errors'] == ['RuntimeError: owned worker failed']


@pytest.mark.parametrize('phase', ['baseline','screening','holdout'])
def test_fatal_calibration_errors_propagate_in_every_phase(phase):
    class FatalCalibrationError(RuntimeError):
        fatal_calibration = True
    failure = FatalCalibrationError('unsafe shutdown or changed immutable identity')
    _,held = calibration_seeds(range(100))
    healthy = FakeRunner()
    def runner(solver,device,count,seeds,calibration=False):
        fatal = ((phase == 'baseline' and count == 1) or
                 (phase == 'screening' and count > 1) or
                 (phase == 'holdout' and seeds == held))
        if fatal:
            raise failure
        return healthy(solver,device,count,seeds,calibration=calibration)
    with pytest.raises(FatalCalibrationError) as caught:
        calibrate_solver(prepared(),SimpleNamespace(),SOLVER,['cuda:0'],runner)
    assert caught.value is failure


def test_normal_candidate_oom_rejects_without_becoming_fatal():
    healthy = FakeRunner()
    def runner(solver,device,count,seeds,calibration=False):
        if count > 1:
            raise MemoryError('ordinary candidate OOM')
        return healthy(solver,device,count,seeds,calibration=calibration)
    counts,report = calibrate_solver(prepared(),SimpleNamespace(),SOLVER,['cuda:0'],runner)
    assert counts['cuda:0'] == 1
    assert len(report['devices']['cuda:0']['screening']) == 3
    assert all(result['wave']['errors'] == ['MemoryError: ordinary candidate OOM']
               and not result['comparison']['accepted']
               for result in report['devices']['cuda:0']['screening'])


def test_calibration_does_not_mutate_prepared_settings_or_campaign():
    value = prepared()
    original = deepcopy(value)
    calibrate_solver(value,SimpleNamespace(),SOLVER,['cuda:0'],FakeRunner())
    assert value == original


@pytest.mark.parametrize('bad_rows', [[None], [dict(seed=2**31,energies=None)],
                                     [dict(seed=[],energies=[100.]*9)]])
def test_malformed_runner_rows_reject_instead_of_escaping(bad_rows):
    def malformed(*args,**kwargs):
        return bad_rows,20.
    counts,report = calibrate_solver(prepared(),SimpleNamespace(),SOLVER,['cuda:0'],malformed)
    assert counts['cuda:0'] == 1
    assert report['devices']['cuda:0']['baseline']['errors']
    json.dumps(report,allow_nan=False)


def test_measured_baseline_cannot_invent_one_admitted_slot():
    value = prepared()
    value['capacity'].update(gpu_available_bytes={'cuda:0':2*GIB},device_count=1,
                              gpu_per_worker_estimate=GIB)
    def exceeds(device,count,seeds,rows):
        for row in rows:
            row['gpu_reserved_peak_bytes'] = 3*GIB
    with pytest.raises(ValueError,match='single-worker baseline exceeds'):
        calibrate_solver(value,SimpleNamespace(),SOLVER,['cuda:0'],FakeRunner(exceeds))


def test_candidate_memory_is_checked_even_when_quality_and_throughput_pass():
    value = prepared()
    value['capacity'].update(gpu_available_bytes={'cuda:0':24*GIB},device_count=1,
                              gpu_per_worker_estimate=GIB)
    def exceeds(device,count,seeds,rows):
        if count > 1:
            for row in rows:
                row['gpu_reserved_peak_bytes'] = 18*GIB
    counts,report = calibrate_solver(value,SimpleNamespace(),SOLVER,['cuda:0'],FakeRunner(exceeds))
    assert counts['cuda:0'] == 1
    assert 'candidate_memory_not_admitted' in report['devices']['cuda:0']['screening'][0]['comparison']['reasons']


def test_holdout_baseline_rechecks_memory_before_selected_candidate_launch():
    value = prepared()
    value['capacity'].update(gpu_available_bytes={'cuda:0':24*GIB},device_count=1,
                              gpu_per_worker_estimate=GIB)
    _,held = calibration_seeds(range(100))
    def exceeds(device,count,seeds,rows):
        if seeds == held:
            for row in rows:
                row['gpu_reserved_peak_bytes'] = 18*GIB
    runner = FakeRunner(exceeds)
    counts,report = calibrate_solver(value,SimpleNamespace(),SOLVER,['cuda:0'],runner)
    assert counts['cuda:0'] == 1
    assert [count for _,count,seeds in runner.calls if seeds == tuple(held)] == [1]
    assert report['devices']['cuda:0']['holdout']['candidate'] is None
