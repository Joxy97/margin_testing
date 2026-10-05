"""Independent scorer cache preserves exact signed/unsigned/overflow objectives."""
from itertools import product
import numpy as np
import pytest

from qubo_benchmark.model import Problem


@pytest.mark.parametrize('values,dtype,offset',[
    ([-128,127,-1],np.int8,7),
    ([-2**63,2**62,2**63-1],np.int64,-9),
    ([2**63-1,0,0],np.int64,10),
    ([0,-2**63,0],np.int64,0),
    ([1,127,255],np.uint8,-11),
    ([2**63-1,0,0],np.uint64,0),
    ([2**64-1,2**64-1,2**64-1],np.uint64,0),
    ([-2**62,0,2**62],np.int64,0),
])
def test_exact_integer_scoring_matches_python_bigint_matrix(values,dtype,offset):
    rows=np.array([0,0,1]);cols=np.array([0,1,1])
    problem=Problem(2,rows,cols,np.array(values,dtype=dtype),offset)
    # A dense object matrix computes x.T @ Q @ x using independent Python-int
    # arithmetic, including values beyond signed64 and cancellation cases.
    dense=np.array([[int(values[0]),int(values[1])],[int(values[1]),int(values[2])]],dtype=object)
    for bits in product((0,1),repeat=2):
        x=np.array(bits,dtype=object)
        expected=offset+x@dense@x
        for _ in range(2):
            actual=problem.score(bits)
            assert isinstance(actual,int)
            assert actual==expected


@pytest.mark.parametrize('values,offset',[
    (np.array([1e16,-1e16,.125],dtype=np.float64),.5),
    (np.array([.125,-.25,.75],dtype=np.float32),-3.),
    (np.array([2**62,-2**61,3],dtype=np.int64),.5),
])
def test_floating_objective_keeps_existing_reduction_order(values,offset):
    rows=np.array([0,0,1]);cols=np.array([0,1,1]);problem=Problem(2,rows,cols,values,offset)
    for bits in product((0,1),repeat=2):
        x=np.array(bits,dtype=np.int64)
        expected=float(offset+np.sum(values.astype(np.float64)*np.array([1,2,1])*x[rows]*x[cols]))
        assert problem.score(bits)==expected


def test_cached_factors_are_detached_readonly_and_survive_save_load(tmp_path):
    rows=np.array([0,0,1]);cols=np.array([0,1,1]);values=np.array([-1,2,3])
    problem=Problem(2,rows,cols,values,7)
    rows[:]=1;cols[:]=0;values[:]=999
    assert problem.score([1,1])==13
    assert problem._score_absolute_bound==15
    with pytest.raises(ValueError):problem.values[0]=999
    with pytest.raises(ValueError):problem._score_coefficients[0]=999
    path=tmp_path/'problem.npz';problem.save(path);loaded=Problem.load(path)
    for bits in product((0,1),repeat=2):assert loaded.score(bits)==problem.score(bits)
    assert loaded._score_absolute_bound==problem._score_absolute_bound


def test_empty_objective_and_invalid_binary_vectors():
    problem=Problem(3,np.array([],dtype=np.int64),np.array([],dtype=np.int64),np.array([],dtype=np.uint64),-7)
    assert problem.score([0,1,0])==-7
    for bits in ([0,1],[[0,1,0]],[0,1,.5],[0,1,float('nan')],[0,1,float('inf')]):
        with pytest.raises(ValueError):problem.score(bits)
