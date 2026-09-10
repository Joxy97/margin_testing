# QOBLIB integer/binary QUBO preparation and eligibility inventory

Pinned QOBLIB revision: `2b400f43c197bb0eb9bc9802efa2b28b818ab63c`.
The dataset is CC-BY-4.0 with upstream per-instance attribution. Source compressed
files and their Git blob/SHA-256 hashes are retained. No SAC run is launched here.

The inventory prefers supplied QS QUBOs where available; otherwise it converts
the supplied LP models. Alternative topology formulations are tried separately
but counted once per class/filename if any is eligible. Counts refer to available
model cases, including parameterizations, not the website's 1,264 base instances.

Continuous variables are rejected, including fixed continuous variables. Bounded
integer variables are encoded as a lower-bound offset plus bounded binary weights
covering exactly their integer domain. Rational linear rows are rescaled to
primitive integer residuals without rounding. Inequalities get bounded integer
slack encoded in binary. Squared residual penalties use a multiplier strictly
larger than a conservative range bound of the binary objective. Conditional on
original feasibility, exact minima satisfy all encoded linear constraints.
No finite-iteration heuristic feasibility guarantee is implied.

Coefficient storage is float64; algebraic equivalence is subject to floating-point
representation. Very large exact residuals are rejected with a numerical guard,
not silently rounded. Unbounded variables and unsupported quadratic/general/SOS
constraints are explicitly recorded. Topology's alternate linear models remain
candidates. Integer mappings and normalized residual matrices are saved for later
decoding and feasibility checks.

Memory admission uses the repository's SBM/SVL/TRF estimates with 32 trajectories,
batches of 8, float64, 10,000 steps, SVL noise chunks 16, TRF sparse matrix format
and candidate batches 128. The cap is 80% of 11 GiB. This is not a measured GPU
allocation guarantee, nor permission to raise batches/use dense matrices without
re-admission. No solver trajectories or unit tests run during preparation.

Before expensive sparse products, conservative penalty-clique size bounds are
used. A bound rejection does not prove the final overlapping matrix cannot fit.
Per-formulation limits are 180 seconds, 12 GiB process address space and 2 GiB
decompressed LP input. Eight concurrent preparations fit within the host's roughly
128 GiB RAM. Timeouts, numerical guards and unsupported transformations are
reported as unresolved/excluded, not claimed inherently impossible.

Outputs: `manifest.json`, incremental `status.json`, `inventory.csv`, final
`summary.json`, compressed QUBOs, source models, and per-case preparation logs.

Reference command on the server:

```bash
OPENBLAS_NUM_THREADS=1 /venv/main/bin/python tools/prepare_qoblib_qubos.py --tree tree.json --output results --workers 8
```
