# Verified dense-200 tuning presets

This folder preserves the selected settings for **144 solvers** on `gka5e`:
130 solvers underwent two tuning passes; 14 user-excluded solvers retain their
archived settings. The families are LM23 (23), SM23 (23), SM20 (20), and SM86
(78 compatible solvers). Loading these files is explicit; library defaults and
historical benchmark configurations are unchanged.

The accompanying numerical changes expose controls in four modules:
`hamiltonian_discrete.py` adds vacancy, temperature, pinning and proposal-mixture
controls; `open_system_discrete.py` adds vacancy initialization, cooling duration
and triple-proposal cadence; `compendium_iii_a.py` adds continuous initialization
scale/noise; and `compendium_iii_c.py` adds terminal schedule fraction. Prior
default values are preserved. Four pre-existing C3 integration changes in
`qubo_benchmark/adapters.py`, `runtime/continuous_calibration.py`,
`runtime/selection.py`, and `runtime/worker.py` are also included because they
belong to the tested runtime snapshot. These runtime changes are distinct from
the new mathematical controls. Copying the JSON files onto the older base
commit is insufficient; use the matching source checked by `source_manifest.json`.

The problem is a dense, 200-variable binary QUBO with published optimum
**−35154**. The objective is `offset + x.T @ Q @ x`; stored upper-triangle
off-diagonal coefficients contribute twice. Every trial has fresh random state,
one worker per GPU, original-source scoring, disabled TF32, and checkpoints at
0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, and 20 seconds.

The **20-second limit is a solution-certification deadline**, with cooperative
solver return. Operations may finish later, but captures completing at or after
20 seconds cannot replace the certified incumbent. A certified optimum stops
that trial; its value is propagated to later checkpoints. Natural solver
termination also propagates the incumbent without more search. Parameters and
the per-solver `wall` or `finite_step` schedule must be used together.

## Recover the single input

Install this checkout with `python -m pip install -e '.[benchmark,dev]'` and a
CUDA-enabled Torch build appropriate for the GPU. Recover only the 103 KB public
`gka5e` source and normalize it using the existing library:

```bash
python fine_tuning/stage_input.py
```

The helper verifies the pinned source checksum and canonical numeric COO
fingerprint before publishing
`benchmark_data/continuous12/normalized/gka5e.npz`. Existing matching inputs are
reused; different inputs are rejected. NPZ compression bytes may differ between
environments, so numeric identity is authoritative and both tested/current byte
hashes are reported. No input payload or generated result is included here.

## Replay one selected preset

Validate the preset, all 100 frozen source files, and the input without GPU work:

```bash
python fine_tuning/run.py --solver lib_c3_rydberg_pxp --seed 900001 \
  --input benchmark_data/continuous12/normalized/gka5e.npz --device cuda:0 \
  --output results/fine_tuning_smoke --dry-run
```

Remove `--dry-run` to execute that one seed. Use `--seeds 900001 900002` for
multiple explicit seeds. The command never defaults to all solvers or a large
campaign, and it refuses an existing output directory. Original hardware is not
enforced; the current GPU and environment are recorded. Hardware and package
differences can change timing and stochastic outcomes.

## Files and evidence

| File | Contents |
| --- | --- |
| `selected_settings.json` | Complete parameters, schedules, selections, and 14 unchanged exclusions. |
| `solvers/<canonical_id>.json` | One preset, archive differences, selected candidate IDs, and bounded observations. |
| `protocol.json` | Input identity, scoring, deadlines, checkpoints, and all first/second-pass seed sets. |
| `source_manifest.json` | Exact hashes of the 100 tested source files and published settings. |
| `environment.json` | Nine anonymous hardware profiles, 36 GPUs, and recorded Python/package versions. |
| `provenance.json` | Base commit, upstream evidence hashes, per-solver selection history, and bundle hashes. |

Only eight machine-specific archive-origin paths were converted to portable
labels in the aggregate settings. Every other field, including every parameter,
schedule, selection ID and archive checksum, matches the verified final export.
The original export byte hash and execution-payload digest are retained.

Each pass used three screening seeds, 12 finalist-validation seeds, and 24 fresh
paired confirmation seeds. The second pass compared against the first pass's
selected settings. These are **selection-affected 24-seed observations**;
confirmation influenced promotion. They are not an independent final test or
the historical 100-seed campaign, and the two passes must not be pooled.
Single-instance tuning does not establish transfer to unseen QUBOs. Readout,
step-count, and coupled-population changes can also affect reported gains.

Exchange-cascade graph mode intentionally retains `memory_limit_bytes: null`:
its CUDA capture memory estimates are heuristic, not guaranteed allocation
bounds. Preserve this setting when replaying the verified preset.

Publication checks passed: 119 focused source tests passed with 3 skipped, and
16 bundle/replay tests passed. A single CUDA replay on an RTX 5060 Ti,
`lib_c3_rydberg_pxp` seed `900001`, reached **−35154 in 2.19747 seconds**.
This is a one-seed replay smoke test, not an additional quality campaign.
Single-input recovery also passed on the GPU host with identical canonical
arrays; its differing NPZ compression bytes were accepted as documented.
