"""Validate and plot an immutable 23-solver baseline plus two new solvers.

This creates a derived comparison, never a resumable CompactStore experiment.
No input campaign files or existing baseline figures are modified. Partial mode
reports progress, but publishes curves only for complete 25-solver groups.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CHECKPOINTS = (.05, .1, .2, .5, 1., 2., 5., 10., 20.)
NEW_SOLVERS = frozenset(('lib_su2_chiral_isospin', 'lib_gravity_tunneling'))
BASELINE_SOLVERS = frozenset(('lib_altermagnet', 'lib_angular_annealing',
    'lib_contact_annealing', 'lib_dynamical_geometry', 'lib_easy_axis_annealing',
    'lib_exchange_cascade', 'lib_geometric', 'lib_greedy_local_search',
    'lib_heat_bath_annealing', 'lib_mean_field_annealing', 'lib_phonon_exchange',
    'lib_random_search', 'lib_replica_annealing', 'lib_simulated_annealing',
    'lib_simulated_bifurcation', 'lib_spherical_annealing',
    'lib_spin_coherent_annealing', 'lib_spin_vector_langevin',
    'lib_supersymmetric', 'lib_tabu_search', 'lib_tap_annealing',
    'lib_transverse_route', 'lib_vector_amplitude_annealing'))
GROUPS = tuple((n, density) for n in (100, 200, 500, 1000, 5000, 10000)
               for density in ('sparse', 'dense') if (n, density) != (10000, 'dense'))
AGGREGATE_COLUMNS = ('checkpoint_seconds', 'gap_to_solution', 'hit_probability', 'TTS99')
SUCCESS = frozenset(('complete', 'completed'))
TERMINAL = SUCCESS | frozenset(('failed', 'unavailable', 'no_in_budget_candidate',
    'error', 'watchdog_timeout', 'invalid_output', 'validation_error', 'oom'))
CLOCK_POLICY = {
    'budget_seconds': 20., 'cpu_threads': 1, 'tf32': False,
    'checkpoint_policy': 'last completed independently scored incumbent at or before each boundary',
    'natural_return_policy': 'retain same result to20s unless verified optimum; do not restart or retune protected solvers',
    'stop_on_optimum': True, 'bks_policy': 'continue after matching or improving BKS',
    'optimum_time_policy': 'first observed completed independently scored optimum, seconds from solve origin',
    'scheduling_policy': 'homogeneous solver groups; all immutable inputs ready before first dispatch',
}
TIMING_DESCRIPTION = ('Original 23-solver wall clock: immutable input/operator and worker '
    'preparation excluded; RNG reset, fresh algorithm state, algorithm preprocessing, '
    'search, decoding, independent scoring and completed certification included. '
    'One uninterrupted 20-second window; no checkpoint restart or active-time variant.')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def checked(path):
    wrapper = read_json(path)
    if not isinstance(wrapper, dict) or wrapper.get('sha256') != digest(wrapper.get('payload')):
        raise ValueError(f'Corrupt checked metadata: {path}')
    return wrapper['payload']


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Unsafe identifier: {value!r}')
    return value


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes campaign root: {path}')
    return path


def group_name(key):
    return f'{key[0]}_{key[1]}'


def discover(root):
    """Inspect at most six directory levels/2,000 directories; ignore payload trees."""
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    found, visited = [], 0
    for folder, directories, files in os.walk(root, followlinks=False):
        visited += 1
        if visited > 2000:
            raise ValueError('Campaign discovery exceeded 2,000 directories')
        folder = Path(folder)
        if 'manifest.json' in files:
            path = contained(root, folder.relative_to(root) / 'manifest.json')
            found.append(path)
            directories[:] = []
        elif len(folder.relative_to(root).parts) >= 6:
            directories[:] = []
        else:
            directories[:] = sorted(d for d in directories if not d.startswith('.') and
                d.lower() not in {'raw', 'progress', 'aggregated', 'plots', 'postprocessed', '__pycache__'})
    return sorted(found)


def load_campaign(path, allowed_solvers, allow_partial=False):
    root = path.parent.resolve()
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema_version') != 2 or not manifest.get('keep_raw_results'):
        raise ValueError(f'Expected retained-raw compact schema 2: {path}')
    if manifest.get('seeds') != list(range(100)) or tuple(manifest.get('checkpoints_seconds', ())) != CHECKPOINTS:
        raise ValueError(f'Expected exact seed order 0..99 and nine checkpoints: {path}')
    solvers = set(manifest.get('solvers', {}))
    if not solvers or not solvers <= allowed_solvers:
        raise ValueError(f'Unexpected solver inventory: {path}')
    if len(manifest.get('problems', [])) != 1:
        raise ValueError(f'Expected one input per campaign directory: {path}')
    problem = manifest['problems'][0]
    key = (problem['n'], problem['density'])
    if key not in GROUPS:
        raise ValueError(f'Out-of-scope group, including skipped dense 10000: {key}')
    identifier(problem['id'])
    if problem.get('reference_type') not in ('OPTIMUM', 'BKS') or not problem.get('integer_objective', True):
        raise ValueError('Expected the frozen integer optimum/BKS references')
    execution = manifest.get('execution', {})
    if any(execution.get(k) != v for k, v in CLOCK_POLICY.items()):
        raise ValueError(f'Clock policy differs from original 23-solver campaign: {path}')
    if execution.get('execution_mode') not in ('isolated_latency', 'multi_gpu_isolated'):
        raise ValueError('Shared-device execution does not match the original worker policy')
    devices = execution.get('devices', [])
    if not devices or any(not d.startswith('cuda:') for d in devices):
        raise ValueError('Campaign must use explicitly recorded CUDA devices')
    plans = execution.get('worker_plan', {})
    if set(plans) != solvers or any(set(plans[s]) != set(devices) or
        any(count != 1 for count in plans[s].values()) for s in solvers):
        raise ValueError('Expected one worker per selected GPU and solver')
    identity = digest(manifest)
    state_path = contained(root, 'state.json')
    state = checked(state_path)
    if state.get('manifest_sha256') != identity or state.get('schema_version') != 2:
        raise ValueError('State identity differs from manifest')
    complete = bool(state.get('complete'))
    if not complete and not allow_partial:
        raise ValueError(f'Incomplete campaign: {root}')
    completion = checked(contained(root, 'completion.json')) if complete else None
    if completion and (completion.get('manifest_sha256') != identity or completion.get('schema_version') != 2):
        raise ValueError('Completion identity differs from manifest')
    snapshots = {'manifest.json': manifest_bytes, 'state.json': state_path.read_bytes()}
    if complete:
        snapshots['completion.json'] = (root / 'completion.json').read_bytes()
    curves, statuses, terminal, failures = {}, {}, 0, 0
    for solver in sorted(solvers):
        identifier(solver)
        name = problem['id']
        relative = f'progress/{solver}/{name}.json'
        progress_path = contained(root, relative)
        progress = checked(progress_path)
        statuses[solver] = progress['status']
        if (progress.get('manifest_sha256') != identity or progress.get('solver_id') != solver or
            progress.get('problem_id') != name or len(progress['status']) != 100):
            raise ValueError('Progress identity or planned seed count differs')
        terminal += sum(s is not None for s in progress['status'])
        failures += sum(s is not None and s not in SUCCESS for s in progress['status'])
        snapshots[relative] = progress_path.read_bytes()
        raw_relative = f'raw/{solver}/{name}.npy'
        raw_bytes = contained(root, raw_relative).read_bytes()
        energies = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
        if energies.dtype != np.float64 or energies.shape != (100, 9) or np.isinf(energies).any():
            raise ValueError('Raw energies have invalid shape, dtype or infinity')
        hashes = progress.get('row_sha256', [])
        times = progress.get('time_to_optimum_s', [])
        if len(hashes) != 100 or len(times) != 100:
            raise ValueError('Missing per-seed checksums or optimum times')
        for i, row in enumerate(energies):
            status = progress['status'][i]
            if status is None:
                if hashes[i] is not None:
                    raise ValueError('Pending seed has a committed checksum')
                continue  # Orphan/uncommitted rows never enter published curves.
            if status not in TERMINAL:
                raise ValueError(f'Unrecognized nonterminal seed status: {status!r}')
            stable = np.asarray(row, dtype='<f8').copy()
            stable[np.isnan(stable)] = np.nan
            if sha(stable.tobytes()) != hashes[i]:
                raise ValueError(f'Raw row checksum differs: {solver}, seed {i}')
            if np.any(np.diff(row[np.isfinite(row)]) > 0):
                raise ValueError('Incumbent energy increases')
            if status not in SUCCESS and not np.isnan(row).all():
                raise ValueError('Failed seed contains available energies')
            if status in SUCCESS and not np.isfinite(row[-1]):
                raise ValueError('Successful seed has no final incumbent')
            if problem['reference_type'] == 'OPTIMUM' and np.any(row < problem['reference']):
                raise ValueError('Energy below the verified optimum')
            timing = times[i]
            if timing is not None:
                exact = np.isfinite(row) & (row == problem['reference'])
                checkpoints = np.asarray(CHECKPOINTS)
                if (status not in SUCCESS or problem['reference_type'] != 'OPTIMUM' or
                    isinstance(timing, bool) or not isinstance(timing, (int, float)) or
                    not math.isfinite(timing) or not 0 <= timing < 20 or not exact.any() or
                    np.any(exact & (checkpoints < timing)) or not exact[checkpoints > timing].all()):
                    raise ValueError('Optimum time conflicts with checkpoint history')
        snapshots[raw_relative] = raw_bytes
        if not complete:
            continue
        csv_relative = f'aggregated/{solver}/{name}.csv'
        csv_bytes = contained(root, csv_relative).read_bytes()
        if (sha(csv_bytes) != progress.get('aggregate_sha256') or
            sha(csv_bytes) != completion['aggregates'][solver][name]):
            raise ValueError('Aggregate CSV checksum differs')
        reader = csv.DictReader(io.StringIO(csv_bytes.decode('utf-8')))
        if tuple(reader.fieldnames or ()) != AGGREGATE_COLUMNS:
            raise ValueError('Aggregate schema differs')
        rows = np.array([[float(row[k]) for k in AGGREGATE_COLUMNS] for row in reader])
        if rows.shape != (9, 4) or not np.array_equal(rows[:, 0], CHECKPOINTS):
            raise ValueError('Aggregate checkpoints differ')
        reference = problem['reference']
        exact = (np.isfinite(energies) & (energies <= reference)).sum(axis=0)
        near = (np.isfinite(energies) & (energies <= reference + .01 * abs(reference))).sum(axis=0)
        gap = np.array([np.mean(energies[:, j] - reference) if np.isfinite(energies[:, j]).all()
                        else np.nan for j in range(9)])
        probabilities = exact / 100
        tts = np.array([math.inf if p == 0 else 0. if p == 1 else t * math.log(.01) /
                        math.log1p(-float(p)) for t, p in zip(CHECKPOINTS, probabilities)])
        if not np.array_equal(rows[:, 1], gap, equal_nan=True) or not np.array_equal(rows[:, 2], probabilities):
            raise ValueError('Saved gap/hit probability differs from raw energies')
        # Linux/Windows libm may differ by a few ulps. Saved CSV bytes stay intact.
        if not np.allclose(rows[:, 3], tts, rtol=2e-15, atol=0., equal_nan=True):
            raise ValueError('Saved TTS99 differs from the defined formula')
        snapshots[csv_relative] = csv_bytes
        curves[solver] = dict(aggregate=rows, exact_hits=exact, near_hits=near,
            failure_count=sum(status not in SUCCESS for status in progress['status']))
    planned = 100 * len(solvers)
    if state.get('planned') != planned or state.get('terminal') != terminal or state.get('failures') != failures:
        raise ValueError('Global progress counts differ')
    if complete and terminal != planned:
        raise ValueError('Final comparison requires 100 terminal attempts for every solver')
    return dict(root=root, manifest=manifest, identity=identity, key=key, complete=complete,
                snapshots=snapshots, curves=curves, terminal=terminal, failures=failures)


def provenance(campaign):
    m = campaign['manifest']
    return dict(directory=str(campaign['root']), manifest_sha256=campaign['identity'],
        source=m['source'], configuration=m['configuration'], catalog=m['catalog'],
        inputs=m['inputs'], environment=m['environment'], execution=m['execution'],
        solvers=m['solvers'], files_sha256={p: sha(b) for p, b in campaign['snapshots'].items()},
        terminal=campaign['terminal'], failures=campaign['failures'], complete=campaign['complete'])


def input_identity(baseline, new_campaign, catalog, fingerprints, new_catalog=None):
    old = baseline['manifest']; new = new_campaign['manifest']; problem = old['problems'][0]
    name = problem['id']
    if new['problems'] != old['problems']:
        raise ValueError('New campaign problem/reference differs from the baseline')
    entries = [e for e in catalog['instances'] if e['instance_id'] == name]
    if len(entries) != 1:
        raise ValueError('Missing or ambiguous original catalog input')
    entry = entries[0]; frozen = fingerprints['instances'][name]
    if (entry['binary_variables'] != problem['n'] or entry['density_category'] != problem['density'] or
        entry['normalized_reference_objective_min'] != problem['reference'] or
        entry['normalized_sha256'] != old['inputs'][name] or
        entry['raw_sha256'] != frozen['raw_sha256'] or frozen['objective_min'] != problem['reference']):
        raise ValueError('Frozen source/input/reference proof differs')
    new_sha = new['inputs'].get(name)
    if new_sha not in (old['inputs'][name], frozen['normalized_sha256']):
        raise ValueError('New input bytes do not match an approved frozen representation')
    if new_sha != old['inputs'][name]:
        if new_catalog is None:
            raise ValueError('A frozen new catalog is required for reserialized NPZ input')
        entries = [e for e in new_catalog['instances'] if e['instance_id'] == name]
        if (len(entries) != 1 or entries[0]['normalized_sha256'] != new_sha or
            entries[0]['raw_sha256'] != frozen['raw_sha256'] or
            entries[0]['binary_variables'] != problem['n'] or
            entries[0]['density_category'] != problem['density'] or
            entries[0]['normalized_reference_objective_min'] != problem['reference']):
            raise ValueError('New catalog/source proof differs')
    pairs = entry['nonzero_offdiagonal_pairs']
    density = 100 * pairs / (problem['n'] * (problem['n'] - 1) / 2)
    if not math.isclose(density, entry['actual_density_percent'], rel_tol=0., abs_tol=1e-12):
        raise ValueError('Catalog pair density disagrees')
    return dict(numeric_coo_sha256=frozen['numeric_coo_sha256'],
        digest_method='continuous-numeric-coo-v1', raw_source_sha256=frozen['raw_sha256'],
        baseline_npz_sha256=old['inputs'][name], new_npz_sha256=new_sha,
        representation_bytes_equal=old['inputs'][name] == new_sha, density_percent=density)


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def plot_group(output, problem, curves, density, hardware_equal):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    names = sorted(BASELINE_SOLVERS) + sorted(NEW_SOLVERS)
    colors = ('#0072b2', '#d55e00', '#009e73', '#cc79a7', '#6f5aa5')
    reference = problem['reference']; optimum = problem['reference_type'] == 'OPTIMUM'
    target = 'optimum' if optimum else 'best-known value'
    specs = [('exact_hits', f'{"Exact optimum" if optimum else "BKS or better"} hits vs time',
              'Reference hits out of 100 planned seeds'),
             ('hits_within_1pct', f'Seeds within 1% of {target} vs time', 'Qualifying seeds out of 100'),
             ('gap_to_reference_percent', f'Mean percentage gap to {target} vs time', 'Signed gap (%)')]
    paths = []
    for metric, title, ylabel in specs:
        figure = Figure(figsize=(22.5, 14.8), dpi=100, facecolor='white')
        FigureCanvasAgg(figure)
        axes = figure.subplots(2, 3).ravel()
        # Keep the original six-panel layout; the last panel contains five methods.
        panels = [names[i:i + 4] for i in range(0, 20, 4)] + [names[20:]]
        for ax, selected in zip(axes, panels):
            for index, solver in enumerate(selected):
                curve = curves[solver]
                if metric == 'exact_hits': y = curve['exact_hits'].astype(float)
                elif metric == 'hits_within_1pct': y = curve['near_hits'].astype(float)
                else: y = 100 * curve['aggregate'][:, 1] / abs(reference)
                label = solver.removeprefix('lib_')
                if curve['failure_count']:
                    label += f' ({curve["failure_count"]} failed)'
                ax.plot(CHECKPOINTS, y, color=colors[index % len(colors)], marker='o', markersize=4,
                    linewidth=2, linestyle='--' if solver in NEW_SOLVERS else '-',
                    label=label)
            ax.set_xscale('log'); ax.set_xticks(CHECKPOINTS)
            ax.set_xticklabels([f'{t:g}' for t in CHECKPOINTS]); ax.grid(alpha=.3)
            ax.set_title(f'Solvers {names.index(selected[0]) + 1}–{names.index(selected[-1]) + 1}', loc='left')
            ax.set_ylabel(ylabel); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.11), fontsize=9)
            if metric != 'gap_to_reference_percent': ax.set_ylim(-2, 102)
            else:
                ax.axhline(1., color='#545b65', linestyle='--', linewidth=1)
                lo, hi = ax.get_ylim(); ax.set_ylim(min(0., lo), max(2., hi))
        figure.suptitle(title, x=.04, ha='left', fontsize=21, fontweight='bold')
        hardware = 'Recorded hardware identities differ between campaigns.' if not hardware_equal else 'Recorded hardware identities match.'
        figure.text(.04, .94, f'{problem["id"]} | {problem["n"]} variables | {density:.2f}% QUBO pair density '
            f'({problem["density"]}) | {target} {reference:g} | 100 planned attempts per solver', fontsize=11)
        figure.text(.04, .02, 'Checkpoint seconds (logarithmic x-axis). Dashed solver curves: two new methods. ' + hardware, fontsize=10)
        figure.subplots_adjust(top=.88, bottom=.16, hspace=.62, wspace=.23)
        destination = output / f'{problem["n"]}_{problem["density"]}_{metric}.png'
        destination.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO(); figure.savefig(buffer, format='png', dpi=100); figure.clear()
        atomic_bytes(destination, buffer.getvalue()); paths.append(str(destination))
    return paths


def compare(baseline_root, new_root, output, *, baseline_catalog, input_fingerprints,
            new_catalog=None, allow_partial=False, plots=True):
    baseline_root, new_root, output = map(lambda p: Path(p).resolve(), (baseline_root, new_root, output))
    for source in (baseline_root, new_root):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Comparison output must be separate from both source roots')
    baseline_catalog = Path(baseline_catalog).resolve(); input_fingerprints = Path(input_fingerprints).resolve()
    catalog = read_json(baseline_catalog); fingerprints = read_json(input_fingerprints)
    if fingerprints.get('digest_method') != 'continuous-numeric-coo-v1':
        raise ValueError('Unsupported canonical input fingerprint proof')
    new_catalog_path = Path(new_catalog).resolve() if new_catalog else None
    new_catalog_data = read_json(new_catalog_path) if new_catalog_path else None
    if new_catalog_path and sha(new_catalog_path.read_bytes()) != fingerprints.get('frozen_catalog_sha256'):
        raise ValueError('New catalog is not bound to the frozen canonical fingerprints')
    baseline, additions = {}, {}
    for path in discover(baseline_root):
        campaign = load_campaign(path, BASELINE_SOLVERS)
        if set(campaign['manifest']['solvers']) != BASELINE_SOLVERS:
            raise ValueError('Expected all original 23 canonical solvers')
        if campaign['failures']:
            raise ValueError('The frozen original baseline has zero recorded failures')
        if campaign['manifest']['catalog'] != sha(baseline_catalog.read_bytes()):
            raise ValueError('Original catalog hash differs from baseline manifest')
        if campaign['key'] in baseline:
            raise ValueError('Duplicate original group is ambiguous')
        baseline[campaign['key']] = campaign
    if set(baseline) != set(GROUPS):
        raise ValueError('Baseline must contain exactly the 11 approved groups')
    for path in discover(new_root):
        campaign = load_campaign(path, NEW_SOLVERS, allow_partial)
        key = campaign['key']; additions.setdefault(key, []).append(campaign)
    report = dict(schema_version=1, artifact_type='derived_comparison_not_resumable_campaign',
        skipped_groups=['10000_dense'], planned_new_trials=2200, baseline_trials=25300,
        new_terminal_attempts=sum(c['terminal'] for campaigns in additions.values() for c in campaigns),
        new_recorded_failures=sum(c['failures'] for campaigns in additions.values() for c in campaigns),
        failure_policy='Preserve terminal failures as all-NaN rows; denominator remains 100 planned attempts; no hidden retry or fallback.',
        seed_order=list(range(100)), checkpoints_seconds=list(CHECKPOINTS), clock_policy=CLOCK_POLICY,
        timing_description=TIMING_DESCRIPTION, input_fingerprints_sha256=sha(input_fingerprints.read_bytes()),
        input_fingerprints_path=str(input_fingerprints), baseline_catalog_sha256=sha(baseline_catalog.read_bytes()),
        new_catalog_sha256=sha(new_catalog_path.read_bytes()) if new_catalog_path else None,
        groups={}, complete=True, plots=[])
    report['new_successful_attempts'] = report['new_terminal_attempts'] - report['new_recorded_failures']
    copies = []
    for key in GROUPS:
        original = baseline[key]; new_campaigns = additions.get(key, [])
        selected = set(); curves = dict(original['curves']); identities = []
        for campaign in new_campaigns:
            current = set(campaign['manifest']['solvers'])
            if selected & current: raise ValueError('Duplicate new solver results are ambiguous')
            selected |= current
            identities.append(input_identity(original, campaign, catalog, fingerprints, new_catalog_data))
            curves.update(campaign['curves'])
        ready = selected == NEW_SOLVERS and all(c['complete'] for c in new_campaigns)
        if not ready and not allow_partial:
            raise ValueError(f'Missing or incomplete two-solver group: {group_name(key)}')
        report['complete'] &= ready
        old_hardware = original['manifest']['environment'].get('hardware_id')
        same_hardware = bool(new_campaigns) and all(old_hardware == c['manifest']['environment'].get('hardware_id') for c in new_campaigns)
        group = dict(complete=ready, problem=original['manifest']['problems'][0],
            baseline=provenance(original), new=[provenance(c) for c in new_campaigns], input_identity=identities,
            hardware_identities_equal=same_hardware,
            hardware_note='Hardware identity equality is not inferred; timing/quality curves retain their separate measured hardware provenance.',
            solver_count=len(BASELINE_SOLVERS | selected), expected_new_solvers=sorted(NEW_SOLVERS))
        report['groups'][group_name(key)] = group
        for label, campaign in [('baseline', original)] + [(f'new_{i}', c) for i, c in enumerate(new_campaigns)]:
            for relative, data in campaign['snapshots'].items():
                destination = contained(output, Path('source_snapshots') / group_name(key) / label / relative)
                copies.append((destination, data))
    # All validation precedes writes. Source bytes are checked again after export.
    for destination, data in copies: atomic_bytes(destination, data)
    for key in GROUPS:
        group = report['groups'][group_name(key)]
        if group['complete'] and plots:
            curves = dict(baseline[key]['curves'])
            for campaign in additions[key]: curves.update(campaign['curves'])
            report['plots'].extend(plot_group(output / 'plots', group['problem'], curves,
                group['input_identity'][0]['density_percent'], group['hardware_identities_equal']))
    for campaign in list(baseline.values()) + [c for groups in additions.values() for c in groups]:
        for relative, data in campaign['snapshots'].items():
            if contained(campaign['root'], relative).read_bytes() != data:
                raise ValueError('Source campaign changed while comparison was being generated')
    report['source_bytes_preserved'] = True
    atomic_bytes(contained(output, 'comparison.json'), json.dumps(report, indent=2, allow_nan=False).encode() + b'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--new-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-catalog', type=Path, default=WORKSPACE / 'vast_git_publish_solver_testing4/configs/continuous_catalog.json')
    parser.add_argument('--input-fingerprints', type=Path, default=WORKSPACE / 'margin_testing_publish5/benchmark_data/continuous12/canonical_fingerprints.json')
    parser.add_argument('--new-catalog', type=Path, help='Frozen catalog bound to canonical fingerprints, when NPZ byte representations differ')
    parser.add_argument('--allow-partial', action='store_true', help='Report pending groups; publish figures only for complete groups')
    parser.add_argument('--no-plots', action='store_true', help='Validate and export comparison metadata/source snapshots without matplotlib')
    args = parser.parse_args(argv)
    try:
        report = compare(args.baseline_root, args.new_root, args.output,
            baseline_catalog=args.baseline_catalog, input_fingerprints=args.input_fingerprints,
            new_catalog=args.new_catalog, allow_partial=args.allow_partial, plots=not args.no_plots)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(dict(complete=report['complete'], groups=11,
        completed_groups=sum(g['complete'] for g in report['groups'].values()),
        new_terminal_attempts=report['new_terminal_attempts'],
        new_recorded_failures=report['new_recorded_failures'],
        new_successful_attempts=report['new_successful_attempts'],
        source_bytes_preserved=True, plots=len(report['plots']), metadata=str(args.output / 'comparison.json'))))


if __name__ == '__main__':
    main()
