"""Validate the frozen production configuration against bounded tuning evidence."""
from pathlib import Path


def production_evidence(args, config):
    from qubo_benchmark.runtime.common import read, filehash, source_snapshot_hash
    from qubo_solvers._hamiltonian_catalog import HAMILTONIAN_SOLVERS
    config = Path(config).resolve()
    folder = config.parent
    identity = read(folder / 'production_config_identity.json')
    files = {
        'production_config_sha256': config,
        'selected_config_sha256': folder / 'selected_config.json',
        'tuning_report_sha256': folder / 'tuning' / 'report.json',
        'candidate_definition_sha256': Path(args.candidates),
    }
    review = folder / 'schedule_review.json'
    if identity.get('schedule_review_sha256') is not None:
        files['schedule_review_sha256'] = review
    elif review.exists():
        raise ValueError('Schedule review was added after configuration resolution; resolve again')
    for key, path in files.items():
        if filehash(path) != identity[key]:
            raise ValueError(f'Frozen production evidence changed: {path}')
    if identity['density'] != args.density:
        raise ValueError('Production configuration belongs to a different density')
    if identity['source_snapshot'] != source_snapshot_hash():
        raise ValueError('Numerical source changed after tuning/configuration resolution')
    if set(identity['validated_solvers']) != set(HAMILTONIAN_SOLVERS):
        raise ValueError('Production configuration does not validate all23 new solvers')
    return dict(identity, wrapper_sha256=filehash(Path(__file__).with_name('campaign.py')),
                guard_sha256=filehash(__file__),
                resolution_identity_sha256=filehash(folder / 'production_config_identity.json'))
