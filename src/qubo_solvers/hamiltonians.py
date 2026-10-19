"""Public inventory of the 23 separately implemented Hamiltonian candidates."""

from .hamiltonian_continuous import (
    SSHDimerOscillator, BoseHubbard, Inozemtsev, Dicke, HaldaneShastry,
    JaynesCummings, JaynesCummingsHubbard, NVector,
)
from .hamiltonian_spin import (
    QuantumClock, MajumdarGhosh, J1J2, Kuramoto, QuantumRotor, SpinSqueezing,
)
from .hamiltonian_gaussian import SparseSYK, FermiHubbard, KitaevChain, BogoliubovDeGennes
from .hamiltonian_discrete import TJVacancy, PottsHeatBath, ZNChiral, XYCoordinate, QuantumDimerWorldline
from ._hamiltonian_catalog import HAMILTONIAN_SOLVERS


HAMILTONIAN_CLASSES = {
    'lib_ssh': SSHDimerOscillator,
    'lib_syk': SparseSYK,
    'lib_bose_hubbard': BoseHubbard,
    'lib_fermi_hubbard': FermiHubbard,
    'lib_tj_vacancy': TJVacancy,
    'lib_inozemtsev': Inozemtsev,
    'lib_dicke': Dicke,
    'lib_haldane_shastry': HaldaneShastry,
    'lib_quantum_clock': QuantumClock,
    'lib_jaynes_cummings': JaynesCummings,
    'lib_jaynes_cummings_hubbard': JaynesCummingsHubbard,
    'lib_majumdar_ghosh': MajumdarGhosh,
    'lib_j1_j2': J1J2,
    'lib_kuramoto': Kuramoto,
    'lib_potts': PottsHeatBath,
    'lib_zn_chiral': ZNChiral,
    'lib_n_vector': NVector,
    'lib_quantum_rotor': QuantumRotor,
    'lib_xy_coordinate': XYCoordinate,
    'lib_quantum_dimer': QuantumDimerWorldline,
    'lib_kitaev_chain': KitaevChain,
    'lib_bdg': BogoliubovDeGennes,
    'lib_spin_squeezing': SpinSqueezing,
}
assert tuple(HAMILTONIAN_CLASSES) == HAMILTONIAN_SOLVERS

__all__ = ['HAMILTONIAN_SOLVERS'] + [cls.__name__ for cls in HAMILTONIAN_CLASSES.values()]
