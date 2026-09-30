"""Compatibility type names; canonical solver options belong to the library."""
from qubo_solvers.backends.classical import PlanarGraphBQMSolver
from qubo_solvers.backends.tree_decomposition import TreeDecompositionBQMSolver, TreeDecompositionSamplerBQMSolver
from qubo_solvers.backends.library_solver import (
    LibraryRandomSearchBQMSolver as RandomBQMSolver,
    LibrarySimulatedAnnealingBQMSolver as SimulatedAnnealingBQMSolver,
    LibraryGreedyLocalSearchBQMSolver as SteepestDescentBQMSolver,
    LibraryTabuSearchBQMSolver as TabuBQMSolver,
)
