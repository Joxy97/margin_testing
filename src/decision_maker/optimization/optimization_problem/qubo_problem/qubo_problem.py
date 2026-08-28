"""Application-level representation of a QUBO problem."""

from dimod import BinaryQuadraticModel

from ..optimization_problem import OptimizationProblem


class QUBOProblem(BinaryQuadraticModel, OptimizationProblem):
    """A binary quadratic model owned by the application boundary."""

    pass
