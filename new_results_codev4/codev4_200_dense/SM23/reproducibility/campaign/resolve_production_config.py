"""Resolve backend defaults without changing the selected numerical settings."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys


def resolve(repo, density, root):
    sys.path.insert(0, str(Path(repo).resolve() / 'src'))
    from qubo_solvers import create_bqm_solver
    from qubo_solvers._hamiltonian_catalog import HAMILTONIAN_SOLVERS
    from qubo_benchmark.runtime.common import atomic, read, filehash, source_snapshot_hash, digest
    root = Path(root).resolve()
    folder = root / density
    selected = folder / 'selected_config.json'
    report_path = folder / 'tuning' / 'report.json'
    report = read(report_path)
    if not report.get('complete') or set(report['selections']) != set(HAMILTONIAN_SOLVERS):
        raise ValueError('Complete tuning evidence for all23 solvers is required')
    if report['identity']['source_snapshot'] != source_snapshot_hash():
        raise ValueError('Numerical source differs from tuning')
    candidates_path = root / 'tuning_candidates.json'
    if filehash(candidates_path) != report['identity']['candidates_sha256']:
        raise ValueError('Tuning candidate definition changed')
    candidates = read(candidates_path)
    steps = read(folder / 'tuning' / 'timing_steps.json')
    source_config = read(selected)
    review_path = folder / 'schedule_review.json'
    review = read(review_path) if review_path.exists() else None
    if review:
        if not review.get('complete') or any(review['identity'][key] != report['identity'][key]
                                             for key in ('source_snapshot', 'input_sha256')):
            raise ValueError('Schedule review is incomplete or uses different source/input')
        original = folder / 'schedule_review' / 'original_selected_config.json'
        if digest(read(original)) != review['original_config_sha256']:
            raise ValueError('Original selected configuration differs from schedule review')
        if digest(read(selected)) != review['selected_config_sha256']:
            raise ValueError('Selected configuration differs from completed schedule review')
        original_config = read(original)
    else:
        original_config = source_config
    result = deepcopy(source_config)

    def validated(trial, parameters):
        final = trial['result']['energies'][-1]
        return (trial['job']['parameters'] == parameters
                and trial['result']['status'] == 'complete'
                and final is not None and math.isfinite(final)
                and trial['job']['input_sha256'] == report['identity']['input_sha256'])

    for solver in HAMILTONIAN_SOLVERS:
        name = report['selections'][solver]['selected']
        option = next(item for item in candidates['solvers'][solver] if item['id'] == name)
        expected = dict(candidates['common'], **option['parameters'], max_steps=steps[solver])
        if original_config['solvers'][solver] != expected:
            raise ValueError(f'Configuration differs from original tuning selection: {solver}')
        for seed in report['identity']['holdout_seeds']:
            trial = read(folder / 'tuning' / 'trials' / f'holdout__{solver}__{name}__{seed}.json')
            if not validated(trial, expected):
                raise ValueError(f'Selected configuration lacks successful validation: {solver}')
        if review and solver in review['solvers']:
            revised = review['solvers'][solver]
            if revised['accepted']:
                if revised['old_parameters'] != expected:
                    raise ValueError(f'Schedule review baseline differs: {solver}')
                if dict(revised['new_parameters'], max_steps=expected['max_steps']) != expected:
                    raise ValueError(f'Schedule review changed physical parameters: {solver}')
                expected = revised['new_parameters']
                support = (folder / revised['supporting_trial']).resolve()
                if not support.is_relative_to(folder.resolve()):
                    raise ValueError('Schedule review trial path leaves the density directory')
                trial = read(support)
                if not validated(trial, expected):
                    raise ValueError(f'Revised schedule lacks successful validation: {solver}')
        if source_config['solvers'][solver] != expected:
            raise ValueError(f'Selected numerical parameters differ from evidence: {solver}')
        resolved = create_bqm_solver(solver, {'device': 'cpu'})._getParameters(expected)
        resolved.pop('seed', None)
        result['solvers'][solver] = resolved
    destination = folder / 'production_config.json'
    atomic(destination, result)
    identity = dict(schema_version=1, density=density,
        source_snapshot=source_snapshot_hash(), selected_config_sha256=filehash(selected),
        production_config_sha256=filehash(destination), tuning_report_sha256=filehash(report_path),
        candidate_definition_sha256=filehash(candidates_path),
        schedule_review_sha256=filehash(review_path) if review else None,
        tuning_helper_sha256=report['identity']['helper_sha256'],
        input_sha256=report['identity']['input_sha256'],
        validated_solvers=list(HAMILTONIAN_SOLVERS),
        resolution_policy='Expand implicit constructor/backend defaults; preserve all supplied numerical settings')
    atomic(folder / 'production_config_identity.json', identity)
    print(f'Resolved and validated23 solvers: {destination}', flush=True)
    return destination


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=str(root / 'repo'))
    parser.add_argument('--root', default=str(root))
    parser.add_argument('--density', choices=('sparse', 'dense'), required=True)
    args = parser.parse_args(argv)
    resolve(args.repo, args.density, args.root)


if __name__ == '__main__':
    main()
