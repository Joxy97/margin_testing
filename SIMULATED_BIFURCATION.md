# Simulated Bifurcation Machine

This CPU implementation follows Algorithm 1 of Orlando et al., *High-Parallel
FPGA-Based Discrete Simulated Bifurcation for Large-Scale Optimization*
(arXiv:2510.12407v2). It implements discrete simulated bifurcation (dSB), with
the paper's optional heating term:

- `solve_cpu`: sparse CSR matrix-vector updates, parallelized with OpenMP when
  available. Rows and trajectories are parallelized adaptively, while coupling
  reductions and time evolution use SIMD lanes. `solve_cpu_batch` additionally
  schedules independent small QUBOs across cores.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/sbm_solve model.qubo
```

## Compact QUBO format

The solver consumes a sparse coordinate format. Variables are zero-based.
Repeated records are summed, and diagonal `q` records are folded into linear
biases.

```text
p qubo <number-of-variables>
o <offset>
l <variable> <linear-bias>
q <variable-u> <variable-v> <quadratic-bias>
```

It represents

`E(q) = offset + sum_i l_i q_i + sum_(i<j) Q_ij q_i q_j`.

The conversion uses `q_i=(s_i+1)/2` and the paper's convention
`H(s)=offset - 1/2 s^T J s - h^T s`. Linear fields are evaluated as couplings
to an ancillary spin fixed at `+1`, as prescribed for dSB in the paper, without
materializing that extra row in CPU/GPU memory.

## D-Wave BQM and CQM input

Install the Python bridge and export a model serialized by `BQM.to_file()` or
`CQM.to_file()`:

```bash
python -m pip install -r requirements.txt
python tools/export_dwave_qubo.py model.bqm model.qubo
./build/sbm_solve model.qubo --steps 10000 --runs 32
```

LP files containing a CQM are also accepted by the exporter. CQM input is
converted with `dimod.cqm_to_bqm`; `--penalty` controls its Lagrange multiplier.
The exporter writes variable labels and, for CQM input, the sample inverter to
`model.qubo.metadata.json`. D-Wave's CQM-to-BQM conversion requires linear
constraints and non-negative lower bounds for integer variables.

## Algorithm parameters

`a` increases linearly from zero to `a0`. At every step, all interactions are
computed from the same position-sign snapshot before position and momentum are
updated. `c0<=0` uses the paper's estimate based on the standard deviation of
the Ising matrix. `gamma=0` gives ordinary dSB; a positive value enables the
heating update. Multiple randomized runs are evaluated against the original
QUBO energy, and the best sample is returned.
