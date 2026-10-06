"""Supersymmetric_QUBO_Solver.tex: classical curvature-assisted rotor search.

Implements the practical commuting tangent field, not a fermionic simulation.
Hessian products are matrix-free. Exact block/proof phases use source dyadics.
"""

import math

from .torch_physics import TorchPhysicsSolver, smoothSchedule, guardedStep
from .geometric import validateCertificateParameters


class TorchSupersymmetricBQMSolver(TorchPhysicsSolver):
    defaults = dict(time_step=.25, confinement_start=.01, confinement_end=.1,
                    curvature_interval=4, curvature_step=.1, metric_strength=1.,
                    armijo=1e-4, block_size=4, block_sweeps=1, proof_nodes=0,
                    proof_max_variables=128, bound_iterations=8)

    @classmethod
    def _validateParameters(cls, p):
        cls._requirePositive(p, 'confinement_start', 'confinement_end', 'curvature_step')
        cls._requireNonnegative(p, 'metric_strength')
        validateCertificateParameters(p)

    @staticmethod
    def _hessianProduct(model, x, y, field, tangent, confinement):
        return (y*model.product(y*tangent)
                + (-x*field + 2*confinement*(x*x-y*y))*tangent)

    def _run(self, model, problem, p, start, width, collect):
        torch = model.torch
        initial = self._initial(model, problem, p, start, width, 2)
        theta = math.pi*initial[:, :, 0]
        tangent = initial[:, :, 1].clone()
        tangent /= tangent.norm(dim=0, keepdim=True).clamp_min(torch.finfo(theta.dtype).tiny)
        x, y = theta.cos(), theta.sin()
        product = model.product(x)
        bestX, bestEnergy = x.clone(), model.energy(x, product)
        stepSize = torch.full((width,), p['time_step'], device=x.device, dtype=x.dtype)
        acceptedCount = torch.zeros(width, device=x.device, dtype=torch.int64)
        collect(x)
        for step in range(p['steps']):
            from ..observation import poll
            poll()
            weight = smoothSchedule(step, p['steps'])
            confinement = p['confinement_start'] + weight*(p['confinement_end']-p['confinement_start'])
            field = product + model.h
            if step % p['curvature_interval'] == 0:
                response = self._hessianProduct(model, x, y, field, tangent, confinement)
                curvature = (tangent*response).sum(0, keepdim=True)
                tangent = tangent-p['curvature_step']*(response-tangent*curvature)
                tangent /= tangent.norm(dim=0, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
            gradient = -y*field + 2*confinement*y*x
            direction = gradient + p['metric_strength']*tangent*(tangent*gradient).sum(0, keepdim=True)
            trialTheta = torch.remainder(theta-stepSize*direction+math.pi, 2*math.pi)-math.pi
            trialX, trialY = trialTheta.cos(), trialTheta.sin()
            trialProduct = model.product(trialX)
            oldEnergy = model.energy(x, product)+confinement*y.double().square().sum(0)
            trialEnergy = model.energy(trialX, trialProduct)+confinement*trialY.double().square().sum(0)
            decrease = p['armijo']*stepSize*(gradient.double()*direction.double()).sum(0)
            accepted, stepSize = guardedStep(torch, x, trialX, oldEnergy, trialEnergy,
                                              stepSize, p['time_step'], decrease)
            theta = torch.where(accepted, trialTheta, theta)
            x, y = torch.where(accepted, trialX, x), torch.where(accepted, trialY, y)
            product = torch.where(accepted, trialProduct, product)
            acceptedCount += accepted
            energy = model.energy(x, product)
            improved = energy < bestEnergy
            bestX = torch.where(improved, x, bestX)
            bestEnergy = torch.minimum(bestEnergy, energy)
            if (step+1) % p['candidate_interval'] == 0 or step+1 == p['steps']:
                if not bool(torch.isfinite(tangent).all()):
                    raise FloatingPointError('Supersymmetric tangent field became nonfinite')
                collect(x)
        collect(bestX)
        return bestX, dict(runs=width, accepted=acceptedCount.cpu().tolist(),
            stalled=(stepSize <= 1e-8).cpu().tolist(),
            tangentNormError=float((tangent.norm(dim=0)-1).abs().max().cpu()))


