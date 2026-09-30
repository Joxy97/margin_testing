"""Lazy library-owned execution backends and compact application interfaces."""

from importlib import import_module


_BACKENDS = {
    "lib_simulated_bifurcation": ("simulated_bifurcation", "SimulatedBifurcationBQMSolver"),
    "lib_altermagnet": ("altermagnet", "TorchAltermagnetBQMSolver"),
    "lib_dynamical_geometry": ("dynamical_geometry", "TorchDynamicalGeometryBQMSolver"),
    "lib_geometric": ("geometric", "TorchGeometricBQMSolver"),
    "lib_phonon_exchange": ("phonon_exchange", "TorchPhononExchangeBQMSolver"),
    "lib_supersymmetric": ("supersymmetric", "TorchSupersymmetricBQMSolver"),
    "lib_categorical": ("categorical", "TorchCategoricalBQMSolver"),
    "lib_categorical_trf": ("categorical_trf", "TorchCategoricalTRFBQMSolver"),
    "lib_planar_graph": ("classical", "PlanarGraphBQMSolver"),
    "lib_tree_decomposition_solver": ("tree_decomposition", "TreeDecompositionBQMSolver"),
    "lib_tree_decomposition_sampler": ("tree_decomposition", "TreeDecompositionSamplerBQMSolver"),
}
BACKEND_IDS = tuple(_BACKENDS)


def backend_class(name):
    """Resolve a canonical implementation without importing application code."""
    try:
        module, class_name = _BACKENDS[name]
    except KeyError as error:
        raise ValueError(f"Unknown library backend: {name!r}") from error
    return getattr(import_module(f"{__name__}.{module}"), class_name)


def create_backend(name, constructor_parameters=None):
    return backend_class(name)(**dict(constructor_parameters or {}))
