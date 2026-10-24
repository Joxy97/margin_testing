"""Compendium III classical QUBO proposals and metadata-aware decoders.

These are the document's classical adaptations, not quantum simulations or a
claim of optimality. Eight decoding models require explicit structural inputs.
"""
from ._compendium_iii_catalog import (
    COMPENDIUM_III_SOLVERS, COMPENDIUM_III_QUBO_SOLVERS,
    COMPENDIUM_III_STRUCTURED_SOLVERS,
)
from .compendium_iii_a import COMPENDIUM_III_A_CLASSES
from .compendium_iii_b import COMPENDIUM_III_B_CLASSES, compile_quantum_double, haah_metadata
from .compendium_iii_c import COMPENDIUM_III_C_CLASSES, compile_parity_decoder

_classes = {
    **COMPENDIUM_III_A_CLASSES, **COMPENDIUM_III_B_CLASSES, **COMPENDIUM_III_C_CLASSES,
}
assert len(_classes) == 86 and set(_classes) == set(COMPENDIUM_III_SOLVERS)
COMPENDIUM_III_CLASSES = {name: _classes[name] for name in COMPENDIUM_III_SOLVERS}
__all__ = [
    'COMPENDIUM_III_SOLVERS', 'COMPENDIUM_III_QUBO_SOLVERS',
    'COMPENDIUM_III_STRUCTURED_SOLVERS', 'COMPENDIUM_III_CLASSES',
    'compile_quantum_double', 'compile_parity_decoder', 'haah_metadata',
]
for _cls in COMPENDIUM_III_CLASSES.values():
    globals()[_cls.__name__] = _cls
    __all__.append(_cls.__name__)
del _cls, _classes
