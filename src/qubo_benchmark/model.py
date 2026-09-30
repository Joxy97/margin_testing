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

    def score(self, sample):
        x = binaryVector(sample, self.n)
        factors = np.where(self.rows == self.cols, 1, 2)
        selected = x[self.rows] * x[self.cols]
        if self.values.dtype.kind in 'iu' and float(self.offset).is_integer():
            bound = abs(int(self.offset)) + sum(abs(int(v))*int(f) for v,f in zip(self.values,factors))
            if bound <= np.iinfo(np.int64).max and self.values.dtype.kind != 'u':
                return int(self.offset) + int(np.sum(self.values*factors*selected, dtype=np.int64))
            return int(self.offset) + sum(int(v)*int(f)*int(s) for v,f,s in zip(self.values,factors,selected))
        return float(self.offset + np.sum(self.values.astype(np.float64)*factors*selected))

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
