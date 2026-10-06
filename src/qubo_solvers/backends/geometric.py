"""Geometric_QUBO_Solver.tex: guarded variable-metric search and exact proof mode.

The default splits starts between flat and parabolic metrics. Elliptic search
is selectable. Optional block search and proof_nodes use exact source dyadics;
without a proof budget the returned candidate is explicitly heuristic.
"""

from .torch_physics import TorchPhysicsSolver, smoothSchedule, guardedStep


def validateCertificateParameters(p):
    if not 1 <= p['block_size'] <= 12:
        raise ValueError('block_size must be between 1 and 12')
    if not 0 < p['armijo'] < 1:
        raise ValueError('armijo must be strictly between zero and one')


class TorchGeometricBQMSolver(TorchPhysicsSolver):
    defaults = dict(time_step=.5, metric='mixed', axis=1., radius=.5, ellipse_axis=1.,
                    confinement_start=.01, confinement_end=.1, armijo=1e-4,
                    block_size=4, block_sweeps=1, proof_nodes=0,
                    proof_max_variables=128, bound_iterations=8)

    @classmethod
    def _validateParameters(cls, p):
        cls._requirePositive(p, 'axis', 'ellipse_axis', 'confinement_start', 'confinement_end')
        cls._requireNonnegative(p, 'radius')
        if p['metric'] not in ('flat', 'parabolic', 'mixed', 'ellipse'):
            raise ValueError('metric must be flat, parabolic, mixed or ellipse')
        validateCertificateParameters(p)

    def _run(self, model, problem, p, start, width, collect):
        torch = model.torch
        initial = self._initial(model, problem, p, start, width)[:, :, 0]
        x = .95*initial
        y = (1-x*x).sqrt()
        stepSize = torch.full((width,), p['time_step'], dtype=x.dtype, device=x.device)
        radius = torch.full((width,), p['radius'], dtype=x.dtype, device=x.device)
        if p['metric'] == 'flat':
            radius.zero_()
        elif p['metric'] == 'mixed':
            radius *= (torch.arange(start, start+width, device=x.device) % 2).to(x.dtype)
        product = model.product(x)
        bestX, bestEnergy = x.clone(), model.energy(x, product)
        acceptedCount = torch.zeros(width, dtype=torch.int64, device=x.device)
        collect(x)
        for step in range(p['steps']):
            from ..observation import poll
            poll()
            weight = smoothSchedule(step, p['steps'])
            confinement = p['confinement_start'] + weight*(p['confinement_end']-p['confinement_start'])
            gradient = product + model.h - 2*confinement*x
            if p['metric'] == 'ellipse':
                omega = y*gradient/(p['axis']**2*y*y+p['ellipse_axis']**2*x*x)
                angle = stepSize*omega
                norm = (1+angle*angle).sqrt()
                trialX = (x-angle*y)/norm
                trialY = (y+angle*x)/norm
                decrease = p['armijo']*stepSize*(y.double()*gradient.double()*omega.double()).sum(0)
            else:
                mass = p['axis']**2+4*radius.square()*x*x
                trialX = (x-stepSize*gradient/mass).clamp(-1, 1)
                decrease = -p['armijo']*(gradient.double()*(trialX-x).double()).sum(0)
            trialProduct = model.product(trialX)
            oldEnergy = model.energy(x, product)+confinement*(1-x.double().square()).sum(0)
            trialEnergy = model.energy(trialX, trialProduct)+confinement*(1-trialX.double().square()).sum(0)
            accepted, stepSize = guardedStep(torch, x, trialX, oldEnergy, trialEnergy,
                                              stepSize, p['time_step'], decrease)
            x = torch.where(accepted, trialX, x)
            product = torch.where(accepted, trialProduct, product)
            if p['metric'] == 'ellipse':
                y = torch.where(accepted, trialY, y)
            acceptedCount += accepted
            energy = model.energy(x, product)
            improved = energy < bestEnergy
            bestX = torch.where(improved, x, bestX)
            bestEnergy = torch.minimum(bestEnergy, energy)
            if (step+1) % p['candidate_interval'] == 0 or step+1 == p['steps']:
                collect(x)
        collect(bestX)
        return bestX, dict(runs=width, metric=p['metric'], accepted=acceptedCount.cpu().tolist(),
                          stalled=(stepSize <= 1e-8).cpu().tolist())


