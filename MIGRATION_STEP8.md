# Step 8: numerical qualification

All edits, checks, artifacts and progress updates belong on **`mmcv2`**.
The two selected seed-0, single-GPU EPA/FDA experiments pass numerical
qualification. This establishes checkpoint inference and short-update
compatibility; modern 5,000-update accuracy comparison remains step 9.
The MR -> CT legacy accuracy reference needs replacement as described below.

The unchanged experiment specifications are:

- `configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`
- `configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`

The stacks remain Python 3.8/Torch 1.7.1/MMCV 1.3.7 and the pinned
Python 3.11/Torch 2.1/MMCV 2.1/MMEngine 0.10.7 environment. Both checks use
FP32, TF32 for matmul/cuDNN, cuDNN benchmark enabled, and nondeterministic
algorithms, matching the legacy contract. Deterministic resume is a
separately labeled control.

## Frozen tolerances

Three legacy processes per direction ran before modern numerical inspection.
Each evaluated three target and three source slices from the final legacy
checkpoint, then ran two initial updates and three updates starting at an
artificial `local_iter=1999`, using the saved actual four-pair medical batch.
This boundary fixture tests the first target/prototype update; it does not
simulate 1,999 prior training updates.

Budgets are five times the maximum observed legacy spread or the predeclared
FP32/TF32 floor, whichever is larger. RMS error is normalized by reference
RMS; maximum error is normalized by the maximum reference magnitude.
Elementwise relative errors near zero are reported but do not define a gate.

| Category | Normalized RMS limit | Normalized maximum limit |
| --- | ---: | ---: |
| FDA input | 0.00001 | 0.0001 |
| Logits | 0.0159145247403 | 0.0251254809834 |
| Loss/log scalars | 0.0048828125 | 0.0048828125 |
| Selected gradients/AdamW moments | 0.0979107455090 | 0.153944629072 |
| Selected parameter/buffer deltas | 0.250175049405 | 9.92952499115 |
| Selected student/teacher/buffer/feature state | 0.0048828125 | 0.0300756994480 |
| Pseudo weights | 0.001 | 0.001 |
| Declared ETF tensors | 0 | 0 |

The very small first learning rate quantizes some deltas to FP32 units and
makes their normalized maximum poorly conditioned. That delta limit alone
is weak; gradients, optimizer moments, resulting weights, initialization,
LRs and update ordering are checked alongside it.

Label disagreement is limited to `0.0048828125`. Full-split same-checkpoint
absolute mDice difference is limited to `0.001`, with prediction disagreement
at most `0.0048828125`. Metrics computed from identical predictions must
agree to absolute `1e-10`, with identical NaN patterns.

The frozen `tolerances.json` SHA256 is
`68a5b02fc351d40738ac60d9ab26921189dd7e51a108f886cf16e2b60a4cbbd5`.
No limit was changed after inspecting modern results. After discovering the
legacy asset-path issue, three additional declared-asset legacy processes
per direction confirmed that the same frozen limits still apply.

## Compatibility fixes and results

The first probes failed because the modern FDA branch used
`torch.fft.ifft2(...).real`, whereas the legacy branch used a real inverse.
The corner replacement does not preserve Hermitian symmetry, so these
operations give different inputs. The modern helper now uses `irfft2` on
the nonnegative last-axis frequencies, with the original spatial shape.
This reproduces the configured legacy 256x256/L=0.01 transform. See the
[PyTorch 2.1 real inverse documentation](https://docs.pytorch.org/docs/2.1/generated/torch.fft.irfft2.html).
The failed comparisons are preserved in `fda_failure_comparisons.json`.

PyTorch CPU linspace rounding also changed derived DropPath rates by up to
approximately `7.45e-9`. `legacy_drop_path_rates` preserves the frozen
FP32 AVX chunk/scalar arithmetic in construction and reset. Fourteen
rate/depth combinations match the legacy outputs exactly, including the
selected 52-block, 0.1-rate model. The implementation follows the
[Torch 1.7.1 linspace kernel](https://raw.githubusercontent.com/pytorch/pytorch/v1.7.1/aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp).

Both default modern probes pass without dropout replay:

| Maximum observed normalized RMS / disagreement | CT -> MR | MR -> CT |
| --- | ---: | ---: |
| FDA input RMS | 5.87e-7 | 4.48e-7 |
| Logit RMS | 0.00405416 | 0.00202037 |
| Loss/log scalar error | 0.00071230 | 0.00038835 |
| Selected gradient/moment RMS | 0.01984921 | 0.00925302 |
| Pseudo-label pixel disagreement | 0.00052261 | 0.00049210 |
| Pseudo-weight error | 0 | 0 |
| Observed DropPath/Dropout2d mask disagreement | 0 | 0 |

Fixed inference inputs and metadata values, declared ETF tensors, seeded
initialization, optimizer groups, learning rates, counters, backward counts
and stochastic module rates agree. Python, NumPy and CPU Torch RNG states
match after every captured update. CUDA Philox seed/offset matches; Torch
1.7 and 2.1 serialize it into different-sized RNG containers (816/16 bytes).
The capture unwraps legacy metadata tuples into lists and resolves frozen
data symlinks for comparison; actual transport types were checked in step 5.

Both stacks evaluated the same final legacy checkpoint over each complete
target split, batch one, sequentially, with the actual test pipeline and
student inference. Direct dataset fetching avoids worker startup in this
numerical diagnostic; four-worker transport is covered separately. Runtime
here is not a training-performance comparison.

| Direction | Target slices | Legacy mDice | Modern mDice | Absolute delta | Changed pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| CT -> MR | 2,400 | 0.685727668499 | 0.685730186867 | 0.000002518368 | 673 |
| MR -> CT | 1,200 | 0.802287016602 | 0.802286682397 | 0.000000334205 | 139 |

Prediction disagreement fractions are `4.28e-6` and `1.77e-6`, respectively.
All ten Dice/ASSD metrics agree exactly when fed identical saved predictions.
Checkpoint identities are `ct2mr_epa_fda_seed0_5k/iter_5000.pth` and
`mr2ct_epa_fda_seed0_5k/iter_5000.pth` under `work_dirs/migration_baseline`;
full SHA256, per-class results and NaN controls are in `qualification.json`.
The MR checkpoint remains valid for this same-checkpoint inference test;
its training result is superseded as an accuracy reference.

Native runner checks were repeated after the fixes with four workers:
two initial updates per direction, a separate three-update artificial
boundary run per direction, target validation, best/full checkpoint
selection, debug images, precision logging and exact EMA equations.
All gradients/losses are finite, teacher gradients are absent, teacher BN
updates once per update, ETF tensors have shape 5x768, and target losses
activate at zero-based iteration 2,000. Backward counts change from two to
four, followed by one optimizer step.

Save/resume restores model, optimizer, scheduler, EPA, RNG and loader state
exactly. CPU and deterministic GPU continuation through the boundary are
bitwise exact. Matched nondeterministic CUDA continuation retains floating
drift (maximum parameter difference about `6.24e-4` in the final boundary
check); its earlier strict diagnostic allclose fails and remains recorded.
It is not presented as bitwise continuation or hidden by new tolerances.
Structural restore, data/draw sequences and the deterministic control pass.

Seed-0 two-slice diagnostics after two updates, using the configs listed
above and checkpoints under `native_training_final`:

| Direction | Checkpoint | Target mDice (GPU) | Source mDice (CPU) |
| --- | --- | ---: | ---: |
| CT -> MR | `ct2mr_initial/iter_2.pth` | 0.009849691424 | 0.029872192851 |
| MR -> CT | `mr2ct_initial/iter_2.pth` | 0.033317513052 | 0.012588630543 |

These small subsets are diagnostics, not an accuracy comparison. Per-class
metrics/ASSD and list identities are in `source_target_smoke.json` and
`native_training_final/validation_lists`. CPU source evaluation ran on the
host for working worker IPC; the stalled sandbox attempts were terminated.

The selected configurations disable MIC, dynamic prototypes, AMP and
distributed execution; those variants are unqualified. Exploratory CPU FDA
checks at 8x8/9x11 with L=0.2 differ from the legacy backend and fail the
configured FDA limits. Those geometries/strengths remain unsupported by
this qualification; the passing scope is the selected 256x256/L=0.01 GPU
experiments, with a matching CPU control at that geometry.

## Legacy reference correction

The isolation step exposed a legacy root-search bug: it required ignored
`AGENTS.md`. The frozen MR -> CT checkout lacked that marker and generated
ETF assets under `mmseg/models/utils/etf_prototypes` at its first update.
Although those tensors differ from the declared tensors by only `1.49e-8`,
generation consumed 3,840 NumPy draws. Gaussian blur draws from the same
parent stream, including before target losses start. Consequently the
original MR -> CT final/best scores cannot be the matched accuracy reference.
CT -> MR loaded the declared assets and remains valid.

Modern root search now uses tracked `README.md` plus `configs/_base_`.
Clean-checkout and frozen-root controls pass without `AGENTS.md`.
The frozen source and both original runs are preserved. Calibration used
the already-cached legacy fallback tensors; declared-asset repeat controls
pass the immutable budgets, and modern comparisons use declared-asset
oracles with exact ETF tensors.

`tools/repair_legacy_baseline.py` launches a replacement MR -> CT run from
the frozen legacy source, pins the declared asset path and rejects any
regeneration. Its seed/config/data/initialization/batch/workers/precision,
validation interval and 50,000 horizon are unchanged; it stops at 5,000.
It then evaluates final/best checkpoints on both target and source splits.
The supervisor and trainer were verified alive on the GPU host, and its
first log confirms **Loaded ETF prototypes** from the declared directory.

Replacement output: `work_dirs/migration_baseline/mr2ct_epa_fda_seed0_declared_5k`.
Durable status: `work_dirs/migration_baseline/mr2ct_declared_baseline_status.json`.
Do not use the old MR -> CT 0.802287/0.806971 scores for the step-9 accuracy
gate. Finish and audit the replacement before matched modern retraining.

## Evidence and reproduction

Artifacts live in `work_dirs/migration_baseline/step8_numerics_20261011`:
`tolerances.json`, calibration/declared-repeat comparisons, default modern
comparisons, full-split prediction arrays and metrics, `cpu_checks.json`,
`native_training_final/verification.json`, and `qualification.json`.
`source_target_smoke.json` reports the final two-update checkpoint diagnostics.
The earlier native-training diagnostic directory is retained separately.
No generated artifacts are intended for commits.

With the modern environment active:

```bash
python -u tools/verify_mmcv2_numerics.py --phase modern --reuse
python -u tools/verify_mmcv2_numerics.py --phase full --reuse
python -u tools/verify_mmcv2_numerics.py --phase compare
```

Fresh calibration requires a fresh `--out-dir` under the baseline directory.
Run phases `legacy`, `oracle`, `modern`, `cpu`, `full` in that order; run
`tools/verify_mmcv2_training.py --cuda --out-dir AUDIT/native_training_final`,
then phase `compare`. Use the same audit path on each numerical invocation.
Never recalibrate an existing frozen policy after modern inspection.
Phase `source-smoke` reproduces the supplemental two-slice CPU source checks
and records the GPU target diagnostics from native training.

Inspect replacement status on the GPU host with:

```bash
/home/akash/miniconda3/envs/epa/bin/python -u \
  tools/repair_legacy_baseline.py --status
```

AST compilation and focused YAPF/isort/Flake8 checks pass; full pre-commit is
not run. Main's reference, 247 frozen files, 107 original configs, six
declared asset files, and all 81 legacy/78 modern packages remain unchanged.

## Subsequent accuracy diagnosis

The step-8 reports remain unchanged evidence for their scoped short-probe
budgets. The 5,000-update runs did not pass accuracy acceptance. Longer
real-data controls exposed near-zero decoder gradients amplified by the
modern cuDNN/TF32 path and substantial within-legacy CUDA repeat drift.
See [MIGRATION_DIAGNOSIS.md](MIGRATION_DIAGNOSIS.md) for measurements,
control interventions and remaining full-run validation. Short numerical
qualification must not be used as a claim of completed accuracy parity.
