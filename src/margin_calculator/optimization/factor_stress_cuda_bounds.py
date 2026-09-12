"""Optional float64 CUDA execution for the existing per-asset ball bounds."""
from __future__ import annotations

import math
from time import perf_counter
import numpy as np

from .factor_stress_bounds import BallErrorBound, _margin_bracket_with_bound
from .factor_stress_reference import referenceArray, referenceRadius


_PREAMBLE = r'''
#define FSTRESS_INF __longlong_as_double(0x7ff0000000000000LL)
#define FSTRESS_MAX 1.7976931348623157e308
#define FSTRESS_EPS 2.2204460492503131e-16
__device__ double positive_bound_exp(double x) {
    return x > log(FSTRESS_MAX) ? FSTRESS_INF : fmax(nextafter(0., 1.), exp(x));
}
__device__ double positive_remainder(double a) {
    if (a == 0.) return 0.;
    if (a > log(FSTRESS_MAX)) return FSTRESS_INF;
    if (a > 1.) return expm1(a)-a-.5*a*a;
    double term=(a/6.)*a*a;
    if (term == 0.) return nextafter(0., 1.);
    double total=term;
    int degree=3;
    while (true) {
        double following=term*a/(degree+1);
        double tail=following/(1.-a/(degree+2));
        if (tail <= FSTRESS_EPS*total) return nextafter(total+tail, FSTRESS_INF);
        total+=following;
        term=following;
        ++degree;
    }
}
'''

_OPERATION = r'''
double row=0.;
if (exposure != 0.) {
    for (int j=0; j<dimension; ++j) row=hypot(row,directions[i*dimension+j]);
}
shock=radius*row;
taylor=0.; lipschitz=0.;
if (exposure != 0. && row != 0.) {
    if (!isfinite(shock)) {
        taylor=FSTRESS_INF; lipschitz=FSTRESS_INF;
    } else {
        double log_weight=log(fabs(exposure))+center;
        double remainder=positive_remainder(shock);
        if (shock < 1e-100) {
            taylor=positive_bound_exp(log_weight+3.*(log(radius)+log(row))-log(6.));
        } else if (isinf(remainder)) {
            taylor=FSTRESS_INF;
        } else {
            taylor=positive_bound_exp(log_weight+log(remainder));
        }
        lipschitz=positive_bound_exp(log_weight+shock+log(row));
    }
}
'''


class CUDABallBounds:
    """One worker/device adapter; imports CuPy only when explicitly selected.

    Each calculation transfers one immutable model, computes Taylor and
    Lipschitz contributions together, and copies the contributions back once.
    Host reductions and certificate identity/feasibility checks are retained.
    SLSQP, PCA, and the full-residual supporting bound are unchanged.
    """

    def __init__(self, device: int):
        if isinstance(device,bool) or not isinstance(device,int):
            raise TypeError('CUDA device index must be an integer')
        import cupy as cp
        if not 0 <= device < cp.cuda.runtime.getDeviceCount():
            raise ValueError('CUDA device index is unavailable')
        self.cp, self.device = cp, device
        self.kernel = cp.ElementwiseKernel(
            'raw float64 directions, float64 center, float64 exposure, int32 dimension, float64 radius',
            'float64 taylor, float64 lipschitz, float64 shock', _OPERATION,
            'factor_stress_ball_bounds_float64_v1', preamble=_PREAMBLE,
            options=('--fmad=false',))
        self.lastSeconds = 0.

    def diagnostics(self, model, radius, coordinates, quadratic, bits, *, scope):
        referenceRadius(radius)
        if isinstance(bits,bool) or not isinstance(bits,int) or not 2 <= bits <= 8:
            raise ValueError('bits per coordinate must be an integer from 2 to 8')
        arrays=[referenceArray(getattr(model,name),name) for name in ('directions','center','exposures')]
        started=perf_counter()
        cp=self.cp
        with cp.cuda.Device(self.device):
            a,c,w=[cp.asarray(value) for value in arrays]
            t,l,s=self.kernel(a,c,w,np.int32(model.dimension),np.float64(radius))
            # A single device-to-host copy synchronizes all kernel computation.
            host=cp.asnumpy(cp.stack((t,l,s)))
        self.lastSeconds=perf_counter()-started
        gross=float(np.abs(model.exposures).sum())
        shock=float(host[2].max(initial=0.))
        def bound(parts):
            with np.errstate(over='ignore'):
                total=float(parts.sum())
            valid=np.isfinite(total)
            return BallErrorBound(parts,total,None,None if gross==0 else total/gross,
                shock,'numerical' if valid else 'none','ok' if valid else 'bound_overflow')
        taylor,lipschitz=bound(host[0]),bound(host[1])
        distance=math.sqrt(model.dimension)*radius/((1 << (bits-1))-1)
        with np.errstate(over='ignore',invalid='ignore'):
            lattice=bound(host[1]*distance)
        bracket=_margin_bracket_with_bound(model,radius,coordinates,quadratic,taylor,scope=scope)
        return dict(taylor=taylor,lipschitz=lipschitz,lattice=lattice,margin_bracket=bracket)

    def metadata(self):
        return dict(type='cuda_ball_bounds',device=f'cuda:{self.device}',dtype='float64',
            cupy_version=self.cp.__version__,runtime_version=self.cp.cuda.runtime.runtimeGetVersion(),
            kernel='factor_stress_ball_bounds_float64_v1',last_transfer_and_kernel_seconds=self.lastSeconds,
            cpu_stages=['PCA','SLSQP','global_quadratic_TRS','exact_candidate_repricing','full_residual_bound'])
