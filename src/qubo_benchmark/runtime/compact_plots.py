"""Exactly three modest PNG curves per problem, derived only from aggregate CSVs."""
import csv
from pathlib import Path

import numpy as np

from .common import read
from .compact_metrics import AGGREGATE_COLUMNS


def plot_results(root, problems=None):
    """Plot mean signed gap in raw objective units, reference hits and TTS99.

    No matrix or solution-vector input is required. Infinite TTS99 is omitted
    from the plotted curve, while the literal p=1 limit of zero is retained.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    root = Path(root)
    manifest = read(root / 'manifest.json')
    selected = set(problems) if problems is not None else None
    paths = []
    labels = dict(gap_to_solution='Mean signed gap (raw objective units)', hit_probability='Hit probability',
                  TTS99='TTS99 (seconds)')
    for problem in manifest['problems']:
        key = problem['id']
        if selected is not None and key not in selected:
            continue
        curves = {}
        for solver in manifest['solvers']:
            path = root / 'aggregated' / solver / (key + '.csv')
            if not path.exists():
                continue
            with path.open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != AGGREGATE_COLUMNS:
                    raise ValueError(f'Compact plot aggregate schema mismatch: {path}')
                rows = [{name: float(value) for name, value in row.items()} for row in reader]
            if len(rows) != len(manifest['checkpoints_seconds']):
                raise ValueError(f'Partial compact plot aggregate: {path}')
            if [row['checkpoint_seconds'] for row in rows] != manifest['checkpoints_seconds']:
                raise ValueError(f'Stale compact plot checkpoint times: {path}')
            curves[solver] = rows
        if not curves:
            continue
        destination = root / 'plots' / key
        destination.mkdir(parents=True, exist_ok=True)
        for metric, label in labels.items():
            figure = Figure(figsize=(7.2, 4.4), dpi=110)
            FigureCanvasAgg(figure)
            axes = figure.add_subplot(111)
            for solver, rows in curves.items():
                x = [row['checkpoint_seconds'] for row in rows]
                y = np.asarray([row[metric] for row in rows])
                y[~np.isfinite(y)] = np.nan
                axes.plot(x, y, marker='.', linewidth=1.1, label=solver)
            axes.set_xscale('log')
            axes.set_xlabel('Checkpoint (seconds)')
            axes.set_ylabel(label)
            axes.set_title(f"{key}: n={problem['n']}, {problem['density']}, {problem['reference_type']}, "
                           f"seeds={len(manifest['seeds'])}", fontsize=9)
            if metric == 'TTS99' and not any(np.isfinite(row[metric]) for rows in curves.values() for row in rows):
                axes.text(.5, .5, 'No finite TTS99: zero observed hits', transform=axes.transAxes,
                          ha='center', va='center', fontsize=9)
            if metric == 'hit_probability':
                axes.set_ylim(-0.025, 1.025)
            elif metric == 'TTS99':
                axes.set_yscale('symlog', linthresh=manifest['checkpoints_seconds'][0])
            axes.grid(True, alpha=0.25)
            axes.legend(fontsize=6, ncol=2 if len(curves) > 8 else 1, loc='best')
            figure.tight_layout()
            path = destination / (metric + '.png')
            figure.savefig(path, dpi=110)
            figure.clear()
            paths.append(path)
    return paths
