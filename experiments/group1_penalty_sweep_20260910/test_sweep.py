"""Small exact checks for the experiment's objective and policy boundaries."""

import itertools
import json
import unittest

import numpy as np
import torch

from benchmark_biqmac_sac import network
from margin_calculator.optimization.portfolio_risk_state_bqm_visitor import PortfolioRiskStateBQMVisitor
from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolverFactory
from run_sweep import HERE, SOLVERS, make_problem, parameters


class SweepTest(unittest.TestCase):
    def setUp(self):
        self.arrays=dict(linear=np.array([-.1,.2,-.3,.4]),offsets=np.array([0,2,4]),
            heads=np.array([0,1]),tails=np.array([2,3]),raw=np.array([100.,.1]),
            normalized=np.array([1.,0.]),seed=np.array(1))
        self.visitor=PortfolioRiskStateBQMVisitor()

    def test_bound_gives_feasible_global_minima_for_both_objectives(self):
        for variant in ('legacy','pruned'):
            problem,penalty,bound=make_problem(self.arrays,variant,'bound',self.visitor)
            self.assertGreater(penalty,bound)
            states=list(itertools.product((0,1),repeat=4))
            minimum=min(map(problem.energy,states))
            for bits in states:
                if abs(problem.energy(bits)-minimum)<1e-10:
                    self.assertEqual(sum(bits[:2]),1)
                    self.assertEqual(sum(bits[2:]),1)

    def test_penalty_preserves_feasible_energy_and_seed(self):
        for variant in ('legacy','pruned'):
            first,_,_=make_problem(self.arrays,variant,'1',self.visitor)
            for strength in ('10','bound'):
                second,_,_=make_problem(self.arrays,variant,strength,self.visitor)
                self.assertEqual(first.seedOffset,second.seedOffset)
                for a,b in itertools.product((0,1),repeat=2):
                    sample=[int(a==0),int(a==1),int(b==0),int(b==1)]
                    self.assertAlmostEqual(first.energy(sample),second.energy(sample))

    def test_pruning_removes_edges_and_keeps_return_scale(self):
        problem,penalty,_=make_problem(self.arrays,'pruned','1',self.visitor)
        np.testing.assert_allclose(problem.linear+penalty,self.arrays['linear'],rtol=0.,atol=1e-15)
        self.assertEqual(problem.interactionCount,3)
        self.assertAlmostEqual(problem.energy([1,0,1,0]),-.1-.3+.1)

    def test_saved_actors_produce_valid_bounded_solver_parameters(self):
        bank=json.loads((HERE/'sac_action_bank.json').read_text())
        saved=torch.load(HERE/'sac_selected_actors.pt',map_location='cpu',weights_only=True)
        problem,_,_=make_problem(self.arrays,'pruned','1',self.visitor)
        reference=np.array([1.3,.0001,0.,1.,0.,0.,0.,0.],dtype=np.float32)
        for name,kind in SOLVERS.items():
            actor=network(len(bank[name]));actor.load_state_dict(saved[name])
            supplied,arm,features=parameters(name,'sac',bank,actor,problem,16,1,reference)
            self.assertEqual(features,reference.tolist())
            self.assertEqual(supplied['runs'],1)
            self.assertEqual(supplied['dtype'],'float32')
            result=BQMSolverFactory.create(kind,{'device':'cpu'}).solve(problem,supplied)
            self.assertAlmostEqual(result.energy,problem.energy(result.sample))
            self.assertEqual(sum(result.sample[:2]),1)
            self.assertEqual(sum(result.sample[2:]),1)


if __name__=='__main__':unittest.main()
