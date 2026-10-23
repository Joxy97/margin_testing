"""Twenty separately specified classical proposal engines from compendium II."""
from ._open_system_catalog import OPEN_SYSTEM_SOLVERS
from .open_system_continuous import OPEN_CONTINUOUS_CLASSES
from .open_system_discrete import OPEN_DISCRETE_CLASSES
from .open_system_spin import OPEN_SPIN_CLASSES

_classes = {**OPEN_CONTINUOUS_CLASSES, **OPEN_DISCRETE_CLASSES, **OPEN_SPIN_CLASSES}
assert set(_classes) == set(OPEN_SYSTEM_SOLVERS)
OPEN_SYSTEM_CLASSES = {name: _classes[name] for name in OPEN_SYSTEM_SOLVERS}
__all__ = ['OPEN_SYSTEM_SOLVERS', 'OPEN_SYSTEM_CLASSES']

from .open_system_continuous import (DopplerCooling, CavityAtomCooling, CavityOptomechanics, OptomechanicalCooling, SteadyStateSqueezing, TomonagaLuttinger, QuadratureSqueezing)

from .open_system_spin import (NoisyLaserAtom, KickedTop, KaneMele, BCS, OneAxisTwisting, TwoAxisCounterTwisting)

from .open_system_discrete import (NoisyExcitationTransport, Kondo, MooreRead, Emery, AKLT, ANNNI, LiebMattis)

__all__ += ['DopplerCooling', 'CavityAtomCooling', 'NoisyLaserAtom', 'KickedTop', 'CavityOptomechanics', 'OptomechanicalCooling', 'NoisyExcitationTransport', 'SteadyStateSqueezing', 'TomonagaLuttinger', 'KaneMele', 'Kondo', 'MooreRead', 'Emery', 'BCS', 'AKLT', 'ANNNI', 'LiebMattis', 'QuadratureSqueezing', 'OneAxisTwisting', 'TwoAxisCounterTwisting']
