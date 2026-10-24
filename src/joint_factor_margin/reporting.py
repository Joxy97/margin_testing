"""Explicit artifact writers; importing the application never imports plotting."""

from pathlib import Path
import json
import os
import tempfile

from .values import primitive


def writeJson(value, path: str | Path):
    """Publish strict JSON atomically, retaining explicit diagnostic statuses."""
    path = Path(path)
    rendered = json.dumps(primitive(value), indent=2, sort_keys=True, allow_nan=False)+"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(rendered)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def writeBacktest(results, directory: str | Path, *, plots=False):
    from backtesting import BacktestCSVReporter
    directory = Path(directory)
    files = BacktestCSVReporter().write(results, directory)
    writeJson(results, directory/"results.json")
    if plots:
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
        except ImportError as error:
            raise ImportError("Backtest plots require joint-factor-margin[reports]") from error
        for name, paths in files.items():
            daily = results.results[name].dailyResults
            figure = Figure(figsize=(10, 4), layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.subplots()
            axes.plot([day.date for day in daily], [day.margin for day in daily], label="Margin")
            axes.plot([day.date for day in daily], [max(0., day.realizedLoss) for day in daily], label="Realized loss")
            axes.set(xlabel="Evaluation date", ylabel="Portfolio currency amount", title=name)
            axes.legend()
            figure.savefig(paths.breaches.parent/"coverage.png", dpi=150)
    return files
