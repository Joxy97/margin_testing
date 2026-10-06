"""Paired quality checks for continuous-mode worker concurrency.

This module neither executes solvers nor imports Torch. The supplied wave
runner owns process readiness, the unchanged 20-second solve, and calibration
peak-memory measurements. Its small report is separate from production metrics.
Few paired seeds can reject observed regressions; they cannot prove that
concurrency preserves the distribution of solution quality.
"""
from __future__ import annotations

import hashlib
import json
import math


CHECKPOINTS = (.05, .1, .2, .5, 1., 2., 5., 10., 20.)
POLICY_VERSION = 1
GIB = 1024**3
MEMORY_FRACTION = .8
NATIVE_SOLVERS = frozenset((
    'lib_simulated_annealing', 'lib_greedy_local_search',
    'lib_spin_vector_langevin', 'lib_angular_annealing', 'lib_transverse_route',
    'lib_easy_axis_annealing', 'lib_spin_coherent_annealing',
    'lib_vector_amplitude_annealing', 'lib_mean_field_annealing',
    'lib_tap_annealing', 'lib_spherical_annealing', 'lib_contact_annealing',
    'lib_replica_annealing', 'lib_heat_bath_annealing', 'lib_tabu_search',
    'lib_random_search', 'lib_exchange_cascade'))
PHYSICS_SOLVERS = frozenset((
    'lib_altermagnet', 'lib_dynamical_geometry', 'lib_geometric',
    'lib_phonon_exchange', 'lib_supersymmetric'))
LIMITATION = ('A small paired screening and held-out sample detects observed '
              'regressions only; it does not establish unchanged quality, a '
              'statistical guarantee, or globally optimal concurrency.')


def _digest(value):
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _positive_integer(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    if maximum is not None and value > maximum:
        raise ValueError(f'{name} must be at most {maximum}')
    return value


def calibration_options(args):
    """Resolve calibration controls without changing solver parameters."""
    maximum = getattr(args, 'max_auto_workers', None)
    maximum = _positive_integer(4 if maximum is None else maximum,
                                'max_auto_workers', 100)
    requested = getattr(args, 'calibration_seeds', None)
    screening = 4 if requested is None else _positive_integer(
        requested, 'calibration_seeds')
    holdout = getattr(args, 'calibration_holdout_seeds', None)
    holdout = 4 if holdout is None else _positive_integer(
        holdout, 'calibration_holdout_seeds')
    if min(screening, holdout) < maximum:
        raise ValueError('Screening and holdout seeds must each cover max_auto_workers')
    speedup = float(getattr(args, 'calibration_min_speedup', 1.1))
    tolerance = float(getattr(args, 'calibration_gap_tolerance', 0.))
    if not math.isfinite(speedup) or speedup < 1.1:
        raise ValueError('calibration_min_speedup must be finite and at least 1.1')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('calibration_gap_tolerance must be finite and nonnegative')
    strict = getattr(args, 'calibration_strict_pairs', True)
    if not isinstance(strict, bool):
        raise ValueError('calibration_strict_pairs must be boolean')
    return dict(max_workers=maximum, screening_seeds=screening,
                holdout_seeds=holdout, min_speedup=speedup,
                gap_tolerance=tolerance, strict_pairs=strict,
                budget_seconds=20., checkpoints=list(CHECKPOINTS))


def calibration_seeds(campaign_seeds, screening_count=4, holdout_count=4):
    """Deterministic disjoint high seeds, also outside nonstandard campaigns."""
    _positive_integer(screening_count, 'screening_count')
    _positive_integer(holdout_count, 'holdout_count')
    excluded = set(campaign_seeds)
    values, current = [], 2**31
    while len(values) < screening_count + holdout_count:
        if current not in excluded:
            values.append(current)
        current += 1
    return values[:screening_count], values[screening_count:]


def calibration_cache_key(prepared, solver, devices, options):
    """Bind frozen evidence to data, code, config, devices and quality policy.

    Current free memory is deliberately excluded: saved counts are restored,
    not retuned on resume. A controller must still validate current admission.
    """
    manifest = prepared['manifest']
    env = prepared.get('environment', {})
    hardware = env.get('hardware', {})
    gpus = hardware.get('gpus', [])
    semantic_execution_keys = (
        'stop_on_optimum', 'budget_seconds', 'cpu_threads', 'tf32',
        'checkpoint_policy', 'natural_return_policy', 'optimum_time_policy',
        'bks_policy', 'wall_schedule_solvers', 'scheduling_policy', 'worker_mode')
    execution = manifest.get('execution', {})
    execution = {key: execution[key] for key in semantic_execution_keys if key in execution}
    identities = {device: next((gpu for gpu in gpus if gpu.get('device') == device),
                               dict(device=device, hardware_id=manifest.get(
                                   'environment', {}).get('hardware_id')))
                  for device in devices}
    return _digest(dict(policy_version=POLICY_VERSION, solver=solver,
        parameters=prepared['settings'][solver], source=manifest.get('source'),
        inputs=manifest.get('inputs'), configuration=manifest.get('configuration'),
        catalog=manifest.get('catalog'), problems=manifest.get('problems'),
        environment=manifest.get('environment'), execution=execution, devices=identities,
        campaign_seeds=prepared.get('seeds', []), options=options))


def estimate_worker_memory(n, coo_count, parameters, native, solver,
                           *, internal_coordinates=None):
    """Conservative host/device estimates including the independent scorer.

    COO is the original upper triangle. Search matrices keep the solver's native
    dense/sparse and padding policy. A dense FP64 original-objective operator is
    admitted independently for native methods and SBM, even when a particular
    integer input permits the scorer's smaller proven-exact FP32 representation.
    Context/library allowances and 20% global headroom supplement tensor storage;
    these estimates are not hard bounds on CUDA pools or external workloads.
    """
    _positive_integer(n, 'n')
    if isinstance(coo_count, bool) or not isinstance(coo_count, int) or coo_count < 0:
        raise ValueError('coo_count must be a nonnegative integer')
    dtype = parameters.get('dtype', 'float32')
    if dtype not in ('float32', 'float64'):
        raise ValueError('Memory admission supports float32 or float64')
    item = 4 if dtype == 'float32' else 8
    runs = _positive_integer(parameters.get('runs', 16), 'runs')
    batch = min(runs, _positive_integer(parameters.get('run_batch_size') or runs,
                                       'run_batch_size'))
    size = n
    if solver == 'lib_altermagnet':
        size = (math.isqrt(n - 1) + 1)**2
    elif solver in ('lib_dynamical_geometry', 'lib_phonon_exchange'):
        size = n + n % 2
    if internal_coordinates is not None:
        size = _positive_integer(internal_coordinates, 'internal_coordinates')
        if size < n:
            raise ValueError('Internal matrix dimension cannot truncate the problem')
    symmetric_nnz = min(n*n, 2*coo_count)
    score_needed = native or solver == 'lib_simulated_bifurcation'
    score_bytes = (8*n*n if symmetric_nnz >= .2*n*n else
                   16*symmetric_nnz + 8*(n+1)) if score_needed else 0
    source_bytes = 32*coo_count + 8*n
    if native:
        search_bytes = 6*item*size*size
        workspace = item*((108+12*parameters.get('replicas', 0))*batch*size + 288*batch)
        host_buffers = 3*source_bytes + n*n*(8+item)
    elif solver in PHYSICS_SOLVERS:
        dense = (parameters.get('matrix_format', 'auto') != 'sparse' and
                 size <= parameters.get('max_dense_variables', size))
        search_bytes = (2*item*size*size if dense else
                        2*(8+item)*symmetric_nnz + 16*(size+1))
        workspace = item*size*batch*(128+32*parameters.get('oscillator_points', 0))
        workspace += min(coo_count, parameters.get('energy_chunk_size', 8192))*batch*32
        host_buffers = 5*source_bytes
    else:
        # Compact SBM retains sparse source arrays and integration/scoring scratch.
        search_bytes = 2*(8+item)*symmetric_nnz + 16*(size+1)
        workspace = item*size*batch*(16+parameters.get('noise_chunk_size', 0))
        workspace += min(coo_count, parameters.get('energy_chunk_size', 1_000_000))*batch*32
        host_buffers = 3*source_bytes
    library_workspace = max(64*1024**2, 2*item*size*batch)
    graph_pool_estimate = (search_bytes+workspace if parameters.get('graph_block', 0) else 0)
    tensor_bytes = search_bytes+score_bytes+workspace+library_workspace+graph_pool_estimate
    return dict(original_variables=n, internal_coordinates=size,
        source_coo_count=coo_count, independent_scoring_bytes=score_bytes,
        search_matrix_bytes=search_bytes, workspace_bytes=workspace,
        library_workspace_bytes=library_workspace, graph_pool_estimate_bytes=graph_pool_estimate,
        cuda_context_reserve_bytes=GIB, host_framework_reserve_bytes=GIB,
        gpu_per_worker_estimate=tensor_bytes+GIB,
        host_per_worker_estimate=host_buffers+workspace+GIB,
        memory_fraction=MEMORY_FRACTION,
        note='Estimates include original scoring and process/library reserves; '
             'allocator/context peaks and external load can still exceed them.')


def admission_capacity(prepared, solver, devices, *, gpu_available_bytes=None,
                       host_available_bytes=None, cpu_allowed=None):
    """Admit per-solver workers without creating a parent CUDA context.

    Share host and CPU admission equally across selected production devices.
    Calibration executes devices separately; production may execute them together.
    The controller supplies CUDA free bytes from its hardware discovery.
    """
    devices = list(devices)
    if not devices or len(devices) != len(set(devices)):
        raise ValueError('Admission requires unique selected devices')
    entry = prepared['entry']
    n = entry['binary_variables']
    coo = prepared.get('coo_count')
    if coo is None:
        pairs = entry.get('nonzero_offdiagonal_pairs')
        if pairs is None:
            pairs = math.ceil(n*(n-1)/2*entry.get('actual_density_percent', 100.)/100.)
        coo = int(pairs)+n  # Count all possible diagonals conservatively.
    estimate = estimate_worker_memory(n, int(coo), prepared['settings'][solver],
                                      solver in NATIVE_SOLVERS, solver)
    if devices == ['cpu']:
        # The same matrix/scorer/dynamics tensors reside in RAM on CPU. Count
        # their storage there, without charging a nonexistent CUDA context.
        estimate['host_per_worker_estimate'] += estimate['gpu_per_worker_estimate']-GIB
    if host_available_bytes is None:
        host_available_bytes = prepared.get('host_available_bytes')
    if cpu_allowed is None:
        cpu_allowed = prepared.get('cpu_allowed')
    if host_available_bytes is None or cpu_allowed is None:
        from .selection import cpu_capacity, host_memory_available
        if host_available_bytes is None:
            host_available_bytes = host_memory_available()
        if cpu_allowed is None:
            cpu_allowed = cpu_capacity()['effective_cpus']
    gpu_available_bytes = (prepared.get('gpu_available_bytes', {}) if
        gpu_available_bytes is None else gpu_available_bytes)
    if host_available_bytes < 0 or not math.isfinite(cpu_allowed) or cpu_allowed <= 0:
        raise ValueError('Available host memory and CPU capacity must be valid')
    host_limit = int(host_available_bytes*MEMORY_FRACTION)//estimate['host_per_worker_estimate']
    # Preserve the runtime's one-lane-per-selected-GPU floor. A small quota can
    # time-share those minimal lanes; every additional lane must fit the quota.
    cpu_limit = max(len(devices), int(cpu_allowed))
    shared_limit = min(host_limit, cpu_limit)
    per_device = {}
    for device in devices:
        limit = shared_limit//len(devices)
        free = gpu_available_bytes.get(device)
        if device.startswith('cuda'):
            if free is None or free < 0:
                raise ValueError(f'CUDA free-memory discovery is required for {device}')
            limit = min(limit, int(free*MEMORY_FRACTION)//estimate['gpu_per_worker_estimate'])
        elif device != 'cpu':
            raise ValueError(f'Unsupported admission device: {device}')
        per_device[device] = max(0, min(100, limit))
    return dict(workers_per_device=per_device,
        maximum_workers_estimate=min(shared_limit, sum(per_device.values())),
        host_available_bytes=int(host_available_bytes), gpu_available_bytes=dict(gpu_available_bytes),
        cpu_allowed=float(cpu_allowed), cpu_worker_limit=cpu_limit, host_worker_limit=host_limit,
        cpu_policy_note='One lane per selected GPU may share a smaller CPU quota; '
                        'additional workers must fit effective CPU availability.',
        shared_workers_per_device=shared_limit//len(devices), device_count=len(devices),
        **estimate)


def _finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _wave(run_wave, solver, device, count, seeds):
    try:
        rows, elapsed = run_wave(solver, device, count, list(seeds), calibration=True)
        return _normalize_wave(rows, elapsed, count, seeds)
    except Exception as exc:
        # Source/input identity violations and incomplete owned-worker shutdown
        # must stop the controller. They are not evidence that a safe baseline
        # can continue, even if their exception otherwise resembles a pilot OOM.
        if getattr(exc, 'fatal_calibration', False):
            raise
        return dict(workers=count, seeds=list(seeds), rows=[], elapsed_s=None,
                    trials_per_second=None, errors=[f'{type(exc).__name__}: {exc}'])


def _normalize_wave(rows, elapsed, count, seeds):
    errors, clean, seen = [], [], set()
    for row in rows:
        seed = row.get('seed')
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError('Calibration row seed must be an integer')
        if seed not in seeds or seed in seen:
            errors.append('duplicate_or_unexpected_seed')
        seen.add(seed)
        values = row.get('energies', row.get('energies9', []))
        if len(values) != len(CHECKPOINTS):
            errors.append('checkpoint_count_mismatch')
        if any(value is not None and _finite_or_none(value) is None and
               not (isinstance(value, float) and math.isnan(value)) for value in values):
            errors.append('invalid_energy')
        values = [_finite_or_none(value) for value in values]
        incumbent = None
        for value in values:
            if incumbent is not None and (value is None or value > incumbent):
                errors.append('invalid_incumbent_curve')
            if value is not None:
                incumbent = value
        status = row.get('status')
        if status != 'complete':
            errors.append('trial_failure')
        clean.append(dict(seed=seed, energies=values, status=status,
            **{key: _finite_or_none(row.get(key)) for key in (
                'time_to_optimum_s', 'actual_solve_wall_s',
                'gpu_reserved_peak_bytes', 'host_rss_peak_bytes')}))
    if seen != set(seeds):
        errors.append('missing_seed')
    elapsed = _finite_or_none(elapsed)
    if elapsed is None or elapsed <= 0:
        errors.append('invalid_wave_elapsed')
    return dict(workers=count, seeds=list(seeds), rows=clean, elapsed_s=elapsed,
        trials_per_second=len(seeds)/elapsed if elapsed and elapsed > 0 and not errors else None,
        errors=sorted(set(errors)))


def compare_quality(baseline, candidate, seeds, reference, *, reference_type='BKS',
                    hit_tolerance=0., gap_tolerance=0., strict_pairs=True):
    """Reject observed hit, mean signed gap, or paired-incumbent regressions."""
    reasons = list(baseline.get('errors', []))+list(candidate.get('errors', []))
    if reasons:
        return dict(accepted=False, reasons=sorted(set(reasons)), checkpoints=[])
    base = {row['seed']: row for row in baseline['rows']}
    proposed = {row['seed']: row for row in candidate['rows']}
    if set(base) != set(seeds) or set(proposed) != set(seeds):
        return dict(accepted=False, reasons=['unpaired_seeds'], checkpoints=[])
    if any(base[seed]['energies'][-1] is None or proposed[seed]['energies'][-1] is None for seed in seeds):
        reasons.append('missing_final_comparable')
    checkpoints = []
    for column, checkpoint in enumerate(CHECKPOINTS):
        before = [base[seed]['energies'][column] for seed in seeds]
        after = [proposed[seed]['energies'][column] for seed in seeds]
        observed_pairs = [(old, new) for old,new in zip(before,after) if old is not None and new is not None]
        if any(old is not None and new is None for old,new in zip(before,after)):
            reasons.append('missing_observed_comparable')
        available = all(value is not None for value in before+after)
        if reference_type == 'OPTIMUM' and any(value is not None and value < reference-hit_tolerance for value in before+after):
            reasons.append('below_proven_optimum')
        exact_base = sum(value is not None and value <= reference+hit_tolerance for value in before)
        exact_next = sum(value is not None and value <= reference+hit_tolerance for value in after)
        near_base = sum(value is not None and value <= reference+.01*abs(reference) for value in before)
        near_next = sum(value is not None and value <= reference+.01*abs(reference) for value in after)
        gap_base = (math.fsum(value-reference for value in before)/len(seeds)
                    if all(value is not None for value in before) else None)
        gap_next = (math.fsum(value-reference for value in after)/len(seeds)
                    if all(value is not None for value in after) else None)
        worse_pairs = sum(new > old+gap_tolerance for old,new in observed_pairs)
        if exact_next < exact_base:
            reasons.append('exact_hit_regression')
        if near_next < near_base:
            reasons.append('one_percent_hit_regression')
        if gap_base is not None and gap_next is not None and gap_next > gap_base+gap_tolerance:
            reasons.append('mean_signed_gap_regression')
        if strict_pairs and worse_pairs:
            reasons.append('paired_incumbent_regression')
        checkpoints.append(dict(seconds=checkpoint, comparable=available,
            observed_paired_seeds=len(observed_pairs),
            unobserved_both_seeds=sum(old is None and new is None for old,new in zip(before,after)),
            baseline_exact_hits=exact_base, candidate_exact_hits=exact_next,
            baseline_one_percent_hits=near_base, candidate_one_percent_hits=near_next,
            baseline_mean_signed_gap=gap_base, candidate_mean_signed_gap=gap_next,
            worse_paired_seeds=worse_pairs))
    return dict(accepted=not reasons, reasons=sorted(set(reasons)), checkpoints=checkpoints)


def _quality(baseline, candidate, seeds, problem, options):
    return compare_quality(baseline, candidate, seeds, problem['reference'],
        reference_type=problem.get('reference_type', 'BKS'),
        hit_tolerance=0. if problem.get('integer_objective', True) else 1e-9,
        gap_tolerance=options['gap_tolerance'], strict_pairs=options['strict_pairs'])


def _comparison(baseline, candidate, seeds, problem, options):
    result = _quality(baseline, candidate, seeds, problem, options)
    before, after = baseline['trials_per_second'], candidate['trials_per_second']
    ratio = after/before if before and after else None
    result['throughput_ratio'] = ratio
    if ratio is None or ratio < options['min_speedup']:
        result['reasons'] = sorted(set(result['reasons']+['insufficient_throughput_gain']))
        result['accepted'] = False
    return result


def _measured_limit(capacity, device, baseline):
    limit = capacity['workers_per_device'][device]
    # A real zero is a known measurement. None means that the worker could not
    # measure its peak; do not turn missing instrumentation into free capacity.
    if any(row.get(key) is None or row[key] < 0 for row in baseline['rows']
           for key in ('gpu_reserved_peak_bytes', 'host_rss_peak_bytes')):
        return 1, dict(gpu=None, host=None, available=False)
    peaks = dict(gpu=max((row.get('gpu_reserved_peak_bytes') or 0 for row in baseline['rows']), default=0),
                 host=max((row.get('host_rss_peak_bytes') or 0 for row in baseline['rows']), default=0),
                 available=True)
    if 'host_available_bytes' in capacity and peaks['host']:
        per_host = max(capacity.get('host_per_worker_estimate', 0), peaks['host'])
        host_slots = int(capacity['host_available_bytes']*MEMORY_FRACTION)//math.ceil(per_host)
        limit = min(limit, host_slots//capacity.get('device_count', 1))
    free = capacity.get('gpu_available_bytes', {}).get(device)
    if free is not None and peaks['gpu']:
        per_gpu = max(capacity.get('gpu_per_worker_estimate', 0), peaks['gpu']+GIB)
        limit = min(limit, int(free*MEMORY_FRACTION)//math.ceil(per_gpu))
    return max(0, limit), peaks


def _check_candidate_memory(comparison, capacity, device, candidate, count):
    limit, peaks = _measured_limit(capacity, device, candidate)
    comparison['memory_admitted_maximum'] = limit
    comparison['memory_peaks_available'] = peaks['available']
    if count > limit:
        comparison['accepted'] = False
        comparison['reasons'] = sorted(set(comparison['reasons']+['candidate_memory_not_admitted']))


def calibrate_solver(prepared, args, solver, devices, run_wave):
    """Return frozen per-device counts and independently reviewable evidence.

    All integer counts 2..admitted maximum receive the same screening seeds.
    Select the highest passing count, then compare it to count 1 on fresh
    holdout seeds. A held-out rejection returns count 1, without trying another
    candidate against the same holdout. No calibration row enters a campaign.
    """
    options = calibration_options(args)
    devices = list(devices)
    counts = {device: 1 for device in devices}
    screening, holdout = calibration_seeds(prepared.get('seeds', []),
        options['screening_seeds'], options['holdout_seeds'])
    report = dict(policy_version=POLICY_VERSION, solver=solver,
        parameter_sha256=_digest(prepared['settings'][solver]),
        cache_key=calibration_cache_key(prepared, solver, devices, options),
        options=options, screening_seeds=screening, holdout_seeds=holdout,
        selected=counts, devices={}, limitation=LIMITATION)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('Calibration requires unique selected devices')
    if any(device == 'cpu' for device in devices):
        report['skip_reason'] = 'CPU uses one worker and skips automatic calibration'
        return counts, report
    if getattr(args, 'no_autotune', False):
        report['skip_reason'] = 'Automatic calibration disabled'
        return counts, report
    capacity = prepared.get('capacity') or admission_capacity(prepared, solver, devices)
    report['admission'] = capacity
    shared_limit = capacity.get('maximum_workers_estimate', sum(capacity['workers_per_device'].values()))
    for device in devices:
        maximum = min(options['max_workers'], capacity['workers_per_device'][device],
                      shared_limit-sum(counts.values())+counts[device])
        if maximum < 1:
            raise ValueError(f'Even one unchanged worker is not memory/CPU admitted on {device}')
        result = dict(admitted_maximum=maximum, selected=1, screening=[], holdout=None)
        report['devices'][device] = result
        if maximum == 1:
            result['reason'] = 'admission_allows_one_worker'
            continue
        baseline = _wave(run_wave, solver, device, 1, screening)
        result['baseline'] = baseline
        baseline_quality = _quality(baseline, baseline, screening, prepared['manifest']['problems'][0], options)
        if not baseline_quality['accepted']:
            result['reason'] = 'baseline_quality_unavailable'
            result['baseline_quality'] = baseline_quality
            continue
        measured, peaks = _measured_limit(capacity, device, baseline)
        maximum = min(maximum, measured)
        result.update(admitted_maximum=maximum, baseline_memory_peaks=peaks)
        if maximum < 1:
            raise ValueError(f'Measured single-worker baseline exceeds available memory admission on {device}')
        if not peaks['available']:
            result['reason'] = 'baseline_memory_measurement_unavailable'
            continue
        selected = 1
        for count in range(2, maximum+1):
            candidate = _wave(run_wave, solver, device, count, screening)
            quality = _comparison(baseline, candidate, screening, prepared['manifest']['problems'][0], options)
            _check_candidate_memory(quality, capacity, device, candidate, count)
            result['screening'].append(dict(wave=candidate, comparison=quality))
            if quality['accepted']:
                selected = count
        if selected == 1:
            result['reason'] = 'no_screening_candidate_passed'
            continue
        held_base = _wave(run_wave, solver, device, 1, holdout)
        held_limit, held_peaks = _measured_limit(capacity, device, held_base)
        if held_limit < 1:
            raise ValueError(f'Measured held-out baseline exceeds available memory admission on {device}')
        if held_limit < selected:
            result['holdout'] = dict(candidate_workers=selected, baseline=held_base,
                candidate=None, comparison=dict(accepted=False,
                    reasons=['holdout_baseline_memory_not_admitted'], checkpoints=[],
                    baseline_memory_peaks=held_peaks))
            result['reason'] = 'holdout_rejected_fallback_one'
            continue
        held_candidate = _wave(run_wave, solver, device, selected, holdout)
        quality = _comparison(held_base, held_candidate, holdout, prepared['manifest']['problems'][0], options)
        _check_candidate_memory(quality, capacity, device, held_candidate, selected)
        result['holdout'] = dict(candidate_workers=selected, baseline=held_base,
                                 candidate=held_candidate, comparison=quality)
        if quality['accepted']:
            counts[device] = selected
            result.update(selected=selected, reason='screening_and_holdout_passed')
        else:
            result['reason'] = 'holdout_rejected_fallback_one'
    return counts, report
