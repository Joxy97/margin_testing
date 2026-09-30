"""Phonon_Exchange_QUBO_Model.tex: calibrated pair-product imaginary-time flow.

Consecutive pairs retain a four-sector oscillator wavefunction. Outside edges
use mean fields. The finite symmetric grid is calibrated once per problem;
this is a variational approximation, not a full many-body quantum simulation.
"""

import numpy

from .torch_physics import TorchPhysicsSolver, smoothSchedule, guardedStep


class TorchPhononExchangeBQMSolver(TorchPhysicsSolver):
    defaults = dict(time_step=.01, oscillator_points=32, oscillator_extent=3.,
                    kinetic=.2, well_strength=2., well_position=1., mediator_coupling=.5,
                    transverse_start=1., conditional_rounding=False)

    @staticmethod
    def _size(n):
        return n+n % 2

    @classmethod
    def _validateParameters(cls, p):
        cls._requirePositive(p, 'oscillator_extent', 'kinetic', 'well_strength',
                             'well_position', 'mediator_coupling', 'transverse_start')
        if not 4 <= p['oscillator_points'] <= 512:
            raise ValueError('oscillator_points must be between 4 and 512')
        if p['oscillator_extent'] <= p['well_position']:
            raise ValueError('oscillator_extent must exceed well_position')

    @staticmethod
    def _calibrate(p):
        """Dirichlet tridiagonal oscillator; exact calibration on this grid."""
        from scipy.linalg import eigh_tridiagonal
        count, extent = p['oscillator_points'], p['oscillator_extent']
        spacing = 2*extent/(count+1)
        grid = numpy.linspace(-extent+spacing, extent-spacing, count)
        # Enforce bitwise parity before diagonalizing opposite tilted sectors.
        grid = (grid-grid[::-1])/2
        off = -p['kinetic']/spacing**2
        diagonal = -2*off + p['well_strength']*(grid*grid-p['well_position']**2)**2
        offDiagonal = numpy.full(count-1, off)
        e0, vectors = eigh_tridiagonal(diagonal, offDiagonal, select='i', select_range=(0, 0))
        e1 = eigh_tridiagonal(diagonal-2*p['mediator_coupling']*grid, offDiagonal,
                             select='i', select_range=(0, 0), eigvals_only=True)[0]
        opposite = eigh_tridiagonal(diagonal+2*p['mediator_coupling']*grid, offDiagonal,
                                   select='i', select_range=(0, 0), eigvals_only=True)[0]
        delta, center = (float(e0[0])-float(e1))/2, (float(e0[0])+float(e1))/2
        if not numpy.isfinite(delta) or delta <= 64*numpy.finfo(float).eps*max(1., abs(center)):
            raise ValueError('Oscillator calibration requires a resolved positive spectral delta')
        if abs(e1-opposite) > 1e-10*max(1., abs(e1), abs(off)):
            raise ValueError('Oscillator calibration failed its parity check')
        return grid, diagonal, off, delta, center, numpy.abs(vectors[:, 0])

    def _preparePhonon(self, model, p):
        torch = model.torch
        grid, diagonal, off, delta, center, ground = self._calibrate(p)
        pairs = model.size//2
        internal = numpy.zeros(pairs)
        matching = (model.heads % 2 == 0) & (model.tails == model.heads+1)
        internal[model.heads[matching]//2] = model.values[matching]
        outside = model.values.copy()
        outside[matching] = 0
        outsideMatrix = model.makeMatrix(outside)

        def tensor(value):
            return torch.tensor(value, dtype=model.h.dtype, device=model.h.device)

        signs = tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        scale = tensor(numpy.abs(internal)/delta)[:, None, None, None]
        source = signs[None, :, 0] - tensor(numpy.sign(internal))[:, None]*signs[None, :, 1]
        mediatorDiagonal = scale*(tensor(diagonal)[None, None, None, :]-center
                                  -p['mediator_coupling']*source[:, None, :, None]*tensor(grid))
        return dict(outside=outsideMatrix, signs=signs, diagonal=mediatorDiagonal,
                    off=scale*off, ground=tensor(ground), delta=delta,
                    pairs=int(numpy.count_nonzero(internal)))

    @staticmethod
    def _applyLocal(torch, psi, field, gamma, template):
        signs = template['signs']
        diagonal = (template['diagonal'] + field[::2, :, None, None]*signs[None, None, :, 0, None]
                    + field[1::2, :, None, None]*signs[None, None, :, 1, None])
        applied = diagonal*psi
        applied[:, :, :, 1:] += template['off']*psi[:, :, :, :-1]
        applied[:, :, :, :-1] += template['off']*psi[:, :, :, 1:]
        applied -= gamma*(psi[:, :, [2, 3, 0, 1]]+psi[:, :, [1, 0, 3, 2]])
        return applied, diagonal

    @staticmethod
    def _magnetizations(torch, psi, signs):
        probabilities = psi.square().sum(-1)
        moments = probabilities @ signs
        return moments.permute(0, 2, 1).reshape(-1, psi.shape[1]), probabilities

    def _run(self, model, problem, p, start, width, collect):
        torch = model.torch
        if not hasattr(model, 'phonon'):
            model.phonon = self._preparePhonon(model, p)
        template = model.phonon
        initial = self._initial(model, problem, p, start, width, 4)[::2]
        psi = (1+.05*initial[:, :, :, None])*template['ground'][None, None, None, :]
        psi /= psi.square().sum((2, 3), keepdim=True).sqrt()
        moments, probabilities = self._magnetizations(torch, psi, template['signs'])
        product = torch.mm(template['outside'], moments)
        stepSize = torch.full((width,), p['time_step'], dtype=psi.dtype, device=psi.device)
        acceptedCount = torch.zeros(width, dtype=torch.int64, device=psi.device)

        def decode():
            sectors = probabilities.argmax(-1)
            # Select the joint sector; independent marginal signs can lose the
            # pair correlation. Lowest sector index resolves probability ties.
            bits = template['signs'][sectors] > 0
            samples = bits.permute(1, 0, 2).reshape(width, model.size)
            collect(moments, samples)

        decode()
        for step in range(p['steps']):
            from ..observation import poll
            poll()
            gamma = p['transverse_start']*(1-smoothSchedule(step+1, p['steps']+1))
            applied, diagonal = self._applyLocal(torch, psi, product+model.h, gamma, template)
            expectation = (psi.double()*applied.double()).sum((2, 3), keepdim=True).to(psi.dtype)
            # A stoquastic H has nonpositive off-diagonals. This local diagonal
            # bound keeps the explicit trial nonnegative without clipping it.
            stiffness = (diagonal-expectation).clamp_min(0).amax((0, 2, 3)).clamp_min(1e-12)
            attemptedStep = torch.minimum(stepSize, .9/stiffness)
            trial = psi-attemptedStep[None, :, None, None]*(applied-expectation*psi)
            trial /= trial.square().sum((2, 3), keepdim=True).sqrt().clamp_min(torch.finfo(psi.dtype).tiny)
            trialMoments, trialProbabilities = self._magnetizations(torch, trial, template['signs'])
            trialProduct = torch.mm(template['outside'], trialMoments)
            trialApplied, _ = self._applyLocal(torch, trial, trialProduct+model.h, gamma, template)
            oldEnergy = ((psi.double()*applied.double()).sum((0, 2, 3))
                         -.5*(moments.double()*product.double()).sum(0))
            trialEnergy = ((trial.double()*trialApplied.double()).sum((0, 2, 3))
                           -.5*(trialMoments.double()*trialProduct.double()).sum(0))
            accepted, stepSize = guardedStep(torch, moments, trialMoments, oldEnergy, trialEnergy,
                                              attemptedStep, p['time_step'])
            accepted &= torch.isfinite(trial).all((0, 2, 3)) & (trial >= 0).all((0, 2, 3))
            psi = torch.where(accepted[None, :, None, None], trial, psi)
            moments = torch.where(accepted, trialMoments, moments)
            probabilities = torch.where(accepted[None, :, None], trialProbabilities, probabilities)
            product = torch.where(accepted, trialProduct, product)
            acceptedCount += accepted
            if (step+1) % p['candidate_interval'] == 0 or step+1 == p['steps']:
                decode()
        return moments, dict(runs=width, accepted=acceptedCount.cpu().tolist(),
            stalled=(stepSize <= 1e-8).cpu().tolist(), spectralDelta=template['delta'],
            retainedPairs=template['pairs'], terminalTransverseField=gamma,
            blockNormError=float((psi.square().sum((2, 3))-1).abs().max().cpu()))


