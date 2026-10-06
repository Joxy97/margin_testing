"""Dynamical_Geometry_QUBO.tex: curved paired rotors and live SO(3) charge.

Uses the symmetric W/Ta/T1/T2 split with exact charge rotations and damping.
Controls are evaluated at the step midpoint; readout uses cos(theta).
"""

import math
from .. import observation

from .torch_physics import TorchPhysicsSolver, smoothSchedule


class TorchDynamicalGeometryBQMSolver(TorchPhysicsSolver):
    defaults = dict(time_step=.05, eta=1., charge_coupling=.5, charge_norm=1.,
                    field_inertia=1., field_strength=1., field_seed=.1,
                    velocity_scale=.1, damping=.2, terminal_damping=1.,
                    field_damping=.5, confinement_start=.01, confinement_end=.2,
                    quartic=0., rho_start=-.5, rho_peak=1., rho_end=-1.)

    @staticmethod
    def _size(n):
        return n + n % 2

    @classmethod
    def _validateParameters(cls, p):
        cls._requirePositive(p, 'charge_norm', 'field_inertia', 'field_strength', 'field_seed',
                             'confinement_start', 'confinement_end')
        cls._requireNonnegative(p, 'eta', 'charge_coupling', 'velocity_scale', 'damping',
                                'terminal_damping', 'field_damping', 'quartic')

    def _run(self, model, problem, p, start, width, collect):
        torch = model.torch
        initial = self._initial(model, problem, p, start, width, 6)
        theta = math.pi * initial[:, :, 0]
        pairs = model.size // 2
        charge = initial[::2, :, 1:4].clone()
        charge /= charge.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(theta.dtype).tiny)
        charge *= p['charge_norm']
        a = p['field_seed'] * initial[::2, :, 4]
        fieldMomentum = torch.zeros_like(a)
        g, eta = p['charge_coupling'], p['eta']
        momentum = p['velocity_scale'] * initial[:, :, 5]
        momentum[1::2] *= 1 + eta*a*a*theta[::2].sin().square()
        momentum[::2] += g*a*charge[:, :, 0]
        momentum[1::2] += g*a*charge[:, :, 1]
        x = theta.cos()
        product = model.product(x)
        collect(x)

        def damp(tau, gamma):
            for side in (0, 1):
                shift = g*a*charge[:, :, side]
                momentum[side::2].copy_(shift + math.exp(-gamma*tau)*(momentum[side::2]-shift))
            fieldMomentum.mul_(math.exp(-p['field_damping']*tau))

        def kick(tau, confinement, rho):
            y = theta.sin()
            gradient = -y*(product+model.h) + 2*confinement*x*y + p['quartic']*x*y**3
            momentum.sub_(tau*gradient)
            fieldMomentum.sub_(tau*p['field_strength']*a*(a*a-rho))

        def rotate(axis, angle):
            # Right-handed exact rotations of the coadjoint charge, not an
            # unconstrained Euclidean update followed by projection.
            c, s = angle.cos(), angle.sin()
            j, k = ((1, 2) if axis == 0 else (2, 0))
            oldJ, oldK = charge[:, :, j].clone(), charge[:, :, k].clone()
            charge[:, :, j] = c*oldJ-s*oldK
            charge[:, :, k] = s*oldJ+c*oldK

        def rotor1(tau):
            velocity = momentum[::2] - g*a*charge[:, :, 0]
            theta[::2].add_(tau*velocity)
            fieldMomentum.add_(tau*g*charge[:, :, 0]*velocity)
            rotate(0, -tau*g*a*velocity)

        def rotor2(tau):
            y, x1 = theta[::2].sin(), theta[::2].cos()
            mass = 1 + eta*a*a*y*y
            velocity = (momentum[1::2] - g*a*charge[:, :, 1]) / mass
            theta[1::2].add_(tau*velocity)
            momentum[::2].add_(tau*eta*a*a*y*x1*velocity.square())
            fieldMomentum.add_(tau*(g*charge[:, :, 1]*velocity + eta*a*y*y*velocity.square()))
            rotate(1, -tau*g*a*velocity)

        dt = p['time_step']
        for step in range(p['steps']):
            from ..observation import poll
            poll()
            progress = observation.schedule_fraction((step + .5) / p['steps'])
            weight = smoothSchedule(step+.5, p['steps'])
            confinement = p['confinement_start'] + weight*(p['confinement_end']-p['confinement_start'])
            gamma = p['damping'] + weight*(p['terminal_damping']-p['damping'])
            rho = (p['rho_start'] + 2*progress*(p['rho_peak']-p['rho_start']) if progress < .5
                   else p['rho_peak'] + (2*progress-1)*(p['rho_end']-p['rho_peak']))
            damp(dt/2, gamma)
            kick(dt/2, confinement, rho)
            a.add_(dt/2 * fieldMomentum / p['field_inertia'])
            rotor1(dt/2)
            rotor2(dt)
            rotor1(dt/2)
            a.add_(dt/2 * fieldMomentum / p['field_inertia'])
            theta.copy_(torch.remainder(theta + math.pi, 2*math.pi) - math.pi)
            x = theta.cos()
            product = model.product(x)
            kick(dt/2, confinement, rho)
            damp(dt/2, gamma)
            if (step+1) % p['candidate_interval'] == 0 or step+1 == p['steps']:
                if not bool(torch.isfinite(momentum).all() & torch.isfinite(a).all()
                            & torch.isfinite(fieldMomentum).all() & torch.isfinite(charge).all()):
                    raise FloatingPointError('Dynamical geometry became nonfinite; reduce time_step')
                collect(x)
        return x, dict(runs=width, pairs=pairs,
            chargeNormError=float((charge.norm(dim=-1)-p['charge_norm']).abs().max().cpu()))


