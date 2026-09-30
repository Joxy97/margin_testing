"""Altermagnet_QUBO_Solver.tex: paired spins with opposite lattice stiffnesses."""

import math

from .torch_physics import TorchPhysicsSolver, smoothSchedule, guardedStep


class TorchAltermagnetBQMSolver(TorchPhysicsSolver):
    defaults = dict(time_step=.4, pairing=.1, anisotropy=.02, canting=0.,
                    stiffness=.01, splitting=.006)

    @staticmethod
    def _size(n):
        return (math.isqrt(n-1)+1)**2

    @classmethod
    def _validateParameters(cls, p):
        cls._requirePositive(p, 'pairing', 'anisotropy')
        cls._requireNonnegative(p, 'canting', 'stiffness')
        if abs(p['splitting']) > p['stiffness']:
            raise ValueError('abs(splitting) must not exceed stiffness')
        if p['time_step'] > .5:
            raise ValueError('Altermagnet time_step must not exceed .5')

    @staticmethod
    def _laplacian(torch, spin, length, horizontal, vertical):
        transverse = spin.clone()
        transverse[:, :, 2] = 0
        grid = transverse.reshape(length, length, spin.shape[1], 3)
        lx = 2*grid - torch.roll(grid, 1, 1) - torch.roll(grid, -1, 1)
        ly = 2*grid - torch.roll(grid, 1, 0) - torch.roll(grid, -1, 0)
        return (horizontal*lx + vertical*ly).reshape_as(spin)

    @staticmethod
    def _rational(torch, spin, gradient, tau, damping):
        """Cayley precession, rational damping, and one radial Newton correction."""
        p, d = tau/2, damping*tau/2
        norm2 = gradient.square().sum(-1, keepdim=True)
        w = (spin*gradient).sum(-1, keepdim=True)
        cross = torch.linalg.cross(gradient, spin)
        rotated = spin + (-2*p*cross + 2*p*p*torch.linalg.cross(gradient, cross))/(1+p*p*norm2)
        trial = ((1-d*d*norm2)*rotated + 2*d*(d*w-1)*gradient)/(1+d*d*norm2-2*d*w)
        return (1.5-.5*trial.square().sum(-1, keepdim=True))*trial

    def _run(self, model, problem, p, start, width, collect):
        torch = model.torch
        initial = self._initial(model, problem, p, start, width, 6)
        a = initial[:, :, :3].clone()
        a /= a.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(a.dtype).tiny)
        b = -a + .15*initial[:, :, 3:]
        b /= b.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(b.dtype).tiny)
        mobility = 1/(.5*model.degree + 2*p['pairing'] + 8*p['canting']
                      + p['anisotropy'] + 8*p['stiffness'] + 1e-12)
        stepSize = torch.full((width,), p['time_step'], dtype=a.dtype, device=a.device)
        length = math.isqrt(model.size)
        x = (a[:, :, 2]-b[:, :, 2])/2
        product = model.product(x)
        bestX, bestEnergy = x.clone(), model.energy(x, product)
        acceptedCount = torch.zeros(width, dtype=torch.int64, device=a.device)
        collect(x)

        def local(first, second, anisotropy):
            la = self._laplacian(torch, first, length, p['stiffness']+p['splitting'],
                                 p['stiffness']-p['splitting'])
            lb = self._laplacian(torch, second, length, p['stiffness']-p['splitting'],
                                 p['stiffness']+p['splitting'])
            combined = first+second
            radius2 = combined.square().sum(-1, keepdim=True)
            pairing = (p['pairing']+p['canting']*radius2)*combined
            ga, gb = pairing+la, pairing+lb
            ga[:, :, :2] += anisotropy*first[:, :, :2]
            gb[:, :, :2] += anisotropy*second[:, :, :2]
            energy = (.5*p['pairing']*radius2.squeeze(-1).double()
                      + .25*p['canting']*radius2.squeeze(-1).double().square()).sum(0)
            energy += .5*anisotropy*(first[:, :, :2].double().square()
                                     +second[:, :, :2].double().square()).sum((0, 2))
            energy += .5*(first.double()*la.double()+second.double()*lb.double()).sum((0, 2))
            return ga, gb, energy

        for step in range(p['steps']):
            from ..observation import poll
            poll()
            weight = smoothSchedule(step, p['steps'])
            anisotropy, damping = p['anisotropy']*(.1+.9*weight), .15+.65*weight
            ga, gb, auxiliary = local(a, b, anisotropy)
            ga[:, :, 2] += .5*(product+model.h)
            gb[:, :, 2] -= .5*(product+model.h)
            tau = (mobility*stepSize).unsqueeze(-1)
            trialA = self._rational(torch, a, ga, tau, damping)
            trialB = self._rational(torch, b, gb, tau, damping)
            trialX = (trialA[:, :, 2]-trialB[:, :, 2])/2
            trialProduct = model.product(trialX)
            _, _, trialAuxiliary = local(trialA, trialB, anisotropy)
            oldEnergy = model.energy(x, product)+auxiliary
            trialEnergy = model.energy(trialX, trialProduct)+trialAuxiliary
            accepted, stepSize = guardedStep(torch, x, trialX, oldEnergy, trialEnergy,
                                              stepSize, p['time_step'])
            accepted &= torch.isfinite(trialA).all((0, 2)) & torch.isfinite(trialB).all((0, 2))
            a = torch.where(accepted[None, :, None], trialA, a)
            b = torch.where(accepted[None, :, None], trialB, b)
            x = torch.where(accepted, trialX, x)
            product = torch.where(accepted, trialProduct, product)
            acceptedCount += accepted
            energy = model.energy(x, product)
            improved = energy < bestEnergy
            bestX = torch.where(improved, x, bestX)
            bestEnergy = torch.minimum(bestEnergy, energy)
            if (step+1) % p['candidate_interval'] == 0 or step+1 == p['steps']:
                collect(x)
        collect(bestX)
        return bestX, dict(runs=width, accepted=acceptedCount.cpu().tolist(),
            stalled=(stepSize <= 1e-8).cpu().tolist(),
            spinNormError=float(torch.maximum((a.norm(dim=-1)-1).abs().max(),
                                              (b.norm(dim=-1)-1).abs().max()).cpu()))


