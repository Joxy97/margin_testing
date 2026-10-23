"""Small independent objective/adapter checks for all twenty candidates."""
import itertools

import numpy as np
import pytest
import torch

from qubo_solvers import QUBO, create_solver, create_bqm_solver
from qubo_solvers._open_system_catalog import OPEN_SYSTEM_SOLVERS
from qubo_solvers.backends.problem import QUBOProblem


@pytest.mark.parametrize('name', OPEN_SYSTEM_SOLVERS)
def test_exact_integer_oracle_and_ising_conversion(name, device):
    matrix=torch.tensor([[-3,1,0,2],[1,2,-2,0],[0,-2,-1,1],[2,0,1,-4]],
                        dtype=torch.float64,device=device)
    problem=QUBO(matrix,7.)
    bits=torch.tensor(list(itertools.product((0,1),repeat=4)),device=device,dtype=torch.int8)
    oracle=[7+sum(int(matrix[i,j])*int(x[i])*int(x[j]) for i in range(4) for j in range(4)) for x in bits]
    for spin in [False,True]:
        p=problem.to_ising() if spin else problem
        result=create_solver(name,max_steps=2).solve(p,restarts=16,batch_size=8,seed=17,
            initial_assignments=2*bits-1 if spin else bits,best_only=True)
        assert float(result.best_energy)==min(oracle)
        torch.testing.assert_close(result.best_energy,p.energy(result.best_assignment))


@pytest.mark.parametrize('name', OPEN_SYSTEM_SOLVERS)
def test_compact_sparse_precision_and_memory_guard(name, device):
    problem=QUBOProblem(np.array([-2.,1.,-1.,-3.,4.]),np.array([0,1],dtype=np.uint32),
                       np.array([2,3],dtype=np.uint32),np.array([1.,-2.]),offset=2.5)
    factory=create_bqm_solver(name,{'device':str(device)})
    parameters=dict(max_steps=2,runs=2,run_batch_size=2,best_only=True,sparse_threshold=1.)
    result=factory.solve(problem,parameters)
    assert result.energy==pytest.approx(problem.energy(result.sample),abs=1e-12)
    with pytest.raises(MemoryError):
        factory.solve(problem,dict(parameters,memory_limit_bytes=1))


def test_explicit_configuration_and_source_precision_admission():
    from qubo_solvers import SOLVERS
    from qubo_benchmark.runtime.selection import inventory
    from qubo_benchmark.runtime.worker import native_input_dtype
    from qubo_benchmark.runtime.continuous_calibration import estimate_worker_memory
    assert len(SOLVERS)==73 and len(set(SOLVERS))==73
    rows={r['solver_id']:r for r in inventory({})}
    for name in OPEN_SYSTEM_SOLVERS:
        assert rows[name]['status']=='excluded'
        assert 'explicit' in rows[name]['reason']
        assert native_input_dtype(dict(solver=name,parameters={'dtype':'float32'}))=='float64'
        p=create_bqm_solver(name,{'device':'cpu'})._getParameters({'runs':2,'run_batch_size':2})
        estimate=estimate_worker_memory(8,16,p,True,name)
        assert estimate['workspace_bytes']>0


@pytest.mark.parametrize('name', OPEN_SYSTEM_SOLVERS)
def test_public_class_and_resident_source_precision(name, device, monkeypatch):
    import qubo_solvers as library
    from qubo_solvers.backends.resource_plan import BQMResourcePlan
    from qubo_solvers.backends.torch_qubo import TorchQUBO
    cls=library.library_solver_class(name)
    assert getattr(library,cls.__name__) is cls and cls.__name__ in library.__all__
    source=QUBOProblem(np.array([16777217.,-3.,-.25]),[0,1,0,2],[1,0,0,2],
                      [.5,-.125,.25,-.75],offset=2.75)
    resident=TorchQUBO(source,*(torch.tensor(a,dtype=t,device=device) for a,t in (
        (source.linear,torch.float64),(source.quadraticHeads,torch.int64),
        (source.quadraticTails,torch.int64),(source.quadraticBiases,torch.float64))))
    adapter=create_bqm_solver(name,{'device':str(device)})
    def forbid(*args,**kwargs):
        pytest.fail('The new solver must use source-precision resident execution')
    monkeypatch.setattr(adapter,'_toLibraryResident',forbid)
    params=dict(max_steps=2,runs=3,run_batch_size=2,dtype='float32',seed=7)
    result,=adapter.solveResidentPlanned([resident],BQMResourcePlan.create([1],1),params)
    assert result.energy==source.energy(result.sample)
