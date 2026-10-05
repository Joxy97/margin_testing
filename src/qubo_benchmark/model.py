"""Reference-free symmetric QUBO and strict, independent objective scoring."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def binaryVector(sample, n):
    x = np.asarray(sample)
    if x.shape != (n,) or x.dtype.kind not in 'biuf' or not np.all(np.isfinite(x)) or not np.all((x == 0) | (x == 1)):
        raise ValueError(f'Expected exactly {n} finite binary values')
    return x.astype(np.int64)


@dataclass(frozen=True)
class Problem:
    n: int
    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    offset: float = 0

    def __post_init__(self):
        if self.n <= 0 or not np.isfinite(self.offset):
            raise ValueError('Invalid dimension or offset')
        r, c, v = np.asarray(self.rows), np.asarray(self.cols), np.asarray(self.values)
        if r.ndim != 1 or r.shape != c.shape or r.shape != v.shape or r.dtype.kind not in 'iu' or c.dtype.kind not in 'iu':
            raise ValueError('Invalid coordinate arrays')
        if np.any(r < 0) or np.any(c >= self.n) or np.any(r > c) or not np.all(np.isfinite(v)):
            raise ValueError('Expected finite upper-triangle coefficients')
        if len(set(zip(r.tolist(), c.tolist()))) != len(r):
            raise ValueError('Duplicate matrix coordinates')
        for name, a in [('rows',r),('cols',c),('values',v)]:
            a = a.copy(); a.flags.writeable = False
            object.__setattr__(self, name, a)
        # Scoring coefficients are immutable problem data, not solver state.
        # Establish overflow safety once instead of walking every coefficient
        # in Python for every independently audited candidate.
        factors = np.where(self.rows == self.cols, 1, 2)
        integer = self.values.dtype.kind in 'iu' and float(self.offset).is_integer()
        bound = None
        if integer:
            offset = int(self.offset)
            maximum = max(abs(int(v.min())), abs(int(v.max()))) if len(v) else 0
            coarse_bound = abs(offset) + maximum * int(factors.sum(dtype=np.int64))
            limit = np.iinfo(np.int64).max
            if coarse_bound <= limit:
                # The proof also makes conversion from unsigned and doubling
                # safe, including narrow input types such as int8 and uint8.
                coefficients = self.values.astype(np.int64) * factors
                bound = abs(offset) + int(np.abs(coefficients).sum(dtype=np.int64))
            else:
                # Use Python integers for INT64_MIN, uint64 and large sums.
                # Never take NumPy abs(INT64_MIN) or cast uint64 before proof.
                coefficients = tuple(int(value)*int(factor) for value,factor in zip(self.values,factors))
                bound = abs(offset) + sum(abs(value) for value in coefficients)
                if bound <= limit:coefficients = np.asarray(coefficients,dtype=np.int64)
        else:
            offset = self.offset
            coefficients = self.values.astype(np.float64) * factors
        if isinstance(coefficients,np.ndarray):coefficients.flags.writeable = False
        object.__setattr__(self,'_score_integer',integer)
        object.__setattr__(self,'_score_offset',offset)
        object.__setattr__(self,'_score_coefficients',coefficients)
        object.__setattr__(self,'_score_absolute_bound',bound)

    def score(self, sample):
        x = binaryVector(sample, self.n)
        selected = x[self.rows] * x[self.cols]
        coefficients = self._score_coefficients
        if self._score_integer:
            if isinstance(coefficients,np.ndarray):
                return self._score_offset + int(np.dot(coefficients,selected))
            return self._score_offset + sum(value for value,active in zip(coefficients,selected) if active)
        # Keep the original float64 reduction order, including zero terms.
        return float(self._score_offset + np.sum(coefficients*selected))

    def sparse(self):
        from scipy.sparse import coo_matrix
        off = self.rows != self.cols
        return coo_matrix((np.r_[self.values,self.values[off]],
                           (np.r_[self.rows,self.cols[off]],np.r_[self.cols,self.rows[off]])),
                          shape=(self.n,self.n)).tocsr()

    def dense(self):
        q = np.zeros((self.n,self.n), dtype=self.values.dtype)
        q[self.rows,self.cols] = self.values
        q[self.cols,self.rows] = self.values
        return q

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path, n=self.n, rows=self.rows, cols=self.cols,
                            values=self.values, offset=self.offset)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(int(z['n']),z['rows'],z['cols'],z['values'],z['offset'].item())
