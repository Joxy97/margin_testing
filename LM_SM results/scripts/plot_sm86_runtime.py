"""Render the existing compact-runtime metrics for the SM86 result collection.

The runtime's Matplotlib lines, markers, grid, logarithmic checkpoint axis,
probability range and symlog TTS axis are retained. The 78 compatible solvers
are split into two six-panel pages: 42 models (seven per panel), then 36 (six).
No earlier collection or plot is modified. This reads a derived collection;
it does not pretend that four independent runs share one resumable manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parents[1]
CHECKPOINTS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
COLUMNS = ('checkpoint_seconds', 'gap_to_solution', 'hit_probability', 'TTS99')
LABELS = dict(gap_to_solution='Mean signed gap (raw objective units)',
              hit_probability='Hit probability', TTS99='TTS99 (seconds)')
UNSUPPORTED = frozenset('lib_c3_' + name for name in (
    'xyzz_hexagonal_code', 'haah_cubic_code', 'quantum_double', 'color_code',
    'toric_code', 'surface_code', 'xzzx_surface_code', 'bacon_shor',
))


def default_root():
    choices = [BASE / name / 'SM86' for name in ('codev4_200_dense', 'dense_200')]
    candidates = [path for path in choices if (path / 'manifest.json').is_file()]
    if len(candidates) != 1:
        raise ValueError('Expected exactly one local SM86 dense_200 collection')
    return candidates[0]


def same_float(actual, expected):
    if math.isnan(expected):
        return math.isnan(actual)
    if math.isinf(expected):
        return actual == expected
    return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-7)


def load_curves(root):
    """Check exported metrics against raw energies before presenting them."""
    manifest = json.loads((root / 'manifest.json').read_bytes())
    if manifest.get('schema_version') != 'sm86_collection_v1':
        raise ValueError('Expected the explicit derived SM86 collection manifest')
    if manifest['seeds'] != list(range(100)) or tuple(manifest['checkpoints_seconds']) != CHECKPOINTS:
        raise ValueError('Expected seeds 0..99 and the existing nine checkpoints')
    solvers = sorted(manifest['solvers'])
    if (len(solvers) != 78 or any(not name.startswith('lib_c3_') for name in solvers)
            or UNSUPPORTED.intersection(solvers)):
        raise ValueError('Expected 78 compatible Compendium III solvers, excluding structured models')
    if len(manifest['problems']) != 1:
        raise ValueError('Expected the single existing n200 dense problem')
    problem = manifest['problems'][0]
    if (problem['id'], problem['n'], problem['density'], problem['reference'],
            problem['reference_type']) != ('gka5e', 200, 'dense', -35154, 'OPTIMUM'):
        raise ValueError('Unexpected dense problem or reference')
    present = sorted(path.name for path in (root / 'aggregated').iterdir() if path.is_dir())
    if present != solvers:
        raise ValueError('Aggregate solver directories disagree with the collection manifest')
    curves = {}
    for solver in solvers:
        path = root / 'aggregated' / solver / 'gka5e.csv'
        with path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != COLUMNS:
                raise ValueError(f'Unexpected compact aggregate schema: {path}')
            rows = [{name: float(value) for name, value in row.items()} for row in reader]
        if tuple(row['checkpoint_seconds'] for row in rows) != CHECKPOINTS:
            raise ValueError(f'Unexpected checkpoint schedule: {path}')
        raw = np.load(root / 'raw' / solver / 'gka5e.npy', allow_pickle=False)
        if raw.shape != (100, 9) or raw.dtype != np.dtype('<f8'):
            raise ValueError(f'Unexpected raw energy shape/dtype for {solver}')
        if np.isinf(raw).any() or np.any(raw[np.isfinite(raw)] < problem['reference']):
            raise ValueError(f'Invalid energy or a value below the optimum for {solver}')
        for index, row in enumerate(rows):
            values = raw[:, index]
            probability = float(np.count_nonzero(values <= problem['reference']) / 100)
            gap = float(values.mean() - problem['reference']) if np.isfinite(values).all() else math.nan
            tts = (math.inf if probability == 0 else 0.0 if probability == 1 else
                   CHECKPOINTS[index] * math.log(.01) / math.log1p(-probability))
            for metric, expected in (('gap_to_solution', gap), ('hit_probability', probability), ('TTS99', tts)):
                if not same_float(row[metric], expected):
                    raise ValueError(f'Aggregate {metric} differs from raw energies for {solver}, checkpoint {index}')
        curves[solver] = rows
    return manifest, problem, curves


def plot(root, manifest, problem, curves):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    destination = root / 'plots' / problem['id']
    destination.mkdir(parents=True, exist_ok=True)
    solvers = sorted(curves)
    paths = []
    for part, page_start, per_panel in ((1, 0, 7), (2, 42, 6)):
        for metric, label in LABELS.items():
            # Each panel keeps the runtime's 7.2 x 4.4 inch plotting allocation.
            figure = Figure(figsize=(21.6, 9.4), dpi=110)
            FigureCanvasAgg(figure)
            axes_list = figure.subplots(2, 3).ravel()
            figure.suptitle(f"{problem['id']}: n=200, dense, OPTIMUM, seeds=100 | "
                           f"SM86, part {part} of 2", fontsize=14)
            figure.text(.5, .017, '78 compatible Compendium III models; eight structured decoders unsupported. '
                        'Hardware differs across original host partitions.',
                        ha='center', fontsize=9)
            try:
                for panel, axes in enumerate(axes_list):
                    start = page_start + panel * per_panel
                    selected = solvers[start:start + per_panel]
                    for solver in selected:
                        rows = curves[solver]
                        x = [row['checkpoint_seconds'] for row in rows]
                        y = np.asarray([row[metric] for row in rows])
                        y[~np.isfinite(y)] = np.nan
                        axes.plot(x, y, marker='.', linewidth=1.1, label=solver)
                    axes.set_xscale('log')
                    axes.set_xlim(CHECKPOINTS[0], CHECKPOINTS[-1])
                    axes.set_xlabel('Checkpoint (seconds)')
                    axes.set_ylabel(label)
                    axes.set_title(f'Solvers {start + 1}-{start + len(selected)}', fontsize=9)
                    if metric == 'TTS99' and not any(
                            np.isfinite(row[metric]) for solver in selected for row in curves[solver]):
                        axes.text(.5, .5, 'No finite TTS99: zero observed hits', transform=axes.transAxes,
                                  ha='center', va='center', fontsize=9)
                    if metric == 'hit_probability':
                        axes.set_ylim(-0.025, 1.025)
                    elif metric == 'TTS99':
                        axes.set_yscale('symlog', linthresh=CHECKPOINTS[0])
                        # TTS is nonnegative. Avoid a large, meaningless negative
                        # autoscale region when one curve reaches the p=1 limit.
                        axes.set_ylim(bottom=0)
                    axes.grid(True, alpha=0.25)
                    axes.legend(fontsize=6, ncol=1, loc='best')
                figure.tight_layout(rect=(0, .04, 1, .96))
                path = destination / f'{metric}_SM86_part{part}.png'
                figure.savefig(path, dpi=110)
                paths.append(path)
            finally:
                figure.clear()
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='SM86 collection; otherwise discover under this results tree')
    parser.add_argument('--validate-only', action='store_true', help='Check inputs without rendering images')
    args = parser.parse_args()
    root = args.root.resolve() if args.root else default_root()
    manifest, problem, curves = load_curves(root)
    if args.validate_only:
        print(json.dumps(dict(status='passed', solvers=len(curves), trials=7800, images_written=0)))
        return
    paths = plot(root, manifest, problem, curves)
    print(json.dumps(dict(status='passed', solvers=len(curves), images_written=len(paths),
                          paths=[str(path) for path in paths]), indent=2))


if __name__ == '__main__':
    main()
