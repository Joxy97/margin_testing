"""Optional per-call observation; no benchmark or reference data enter algorithms.

Hooks are inert outside a scoped observer. A cooperative interruption unwinds the
current solve; the observer owns already completed, immutable candidates.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_observer = ContextVar('qubo_observer', default=None)


class SolveInterrupted(Exception):
    pass


def current():
    return _observer.get()


def poll():
    observer = current()
    if observer is not None:
        observer.poll()


def capture(samples, raw=None, *, phase='checkpoint', iteration=None):
    observer = current()
    if observer is not None:
        observer.capture(samples, raw, phase=phase, iteration=iteration)


def prepared(matrix, **details):
    observer = current()
    if observer is not None and hasattr(observer, 'prepared'):
        observer.prepared(matrix, **details)


@contextmanager
def observing(observer):
    token = _observer.set(observer)
    try:
        yield observer
    finally:
        _observer.reset(token)
