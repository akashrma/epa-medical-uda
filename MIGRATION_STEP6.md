# Step 6: native training, UDA updates, and resume

Completed on `mmcv2`, 2026-10-11. All migration edits and progress belong on
this branch. `main`, the legacy environment, original experiment configs,
and the 247-file frozen legacy source remain unchanged.

## Training contract

`BaseSegmentor` now implements MMEngine `BaseModel` training, validation,
and prediction interfaces. `DACS.train_step(data, optim_wrapper)` transfers
the plain tensor batch, clears gradients with `set_to_none=False`, executes
the retained UDA forward/backward sequence, and steps AdamW once. Native
loops receive scalar log variables and do not perform another backward.

`UDAOptimWrapper` counts optimizer updates rather than backward calls.
Gradient accumulation and AMP are outside the selected FP32 contract.
`LegacyOptimWrapperConstructor` preserves all 2,134 named-parameter groups,
including detached teacher and dormant head parameters, custom-key precedence,
head LR multiplier, decay rules, and non-fused AdamW defaults.
Its scheduler/logging interfaces expose actual optimizer groups rather than
MMEngine's synthetic default-LR group.

The student pre-logit feature hook is installed when constructing prototype
UDA models. Training no longer depends on a launcher installing this hook.
EPA/FDA math, pseudo-label threshold/weighting, strong augmentation and random
draws retain the existing implementations. At zero-based iterations 1999,
2000 and 2001 the selected configs perform respectively 2, 4 and 4 backward
passes, with exactly one optimizer step each. Target segmentation and target
ETF first activate at update 2001.

Teacher parameters are copied at the first update and subsequently updated
before forwards using the legacy EMA formula. Teacher dropout and DropPath
are disabled while BatchNorm remains in training mode. BN buffers are not
copied or EMA-updated with teacher parameters. Selected EMA tensors match
the formula exactly, teacher gradients stay absent, and teacher BN counters
advance once per training update in the qualification runs.

`EPARunner` initializes weights on CPU before CUDA transfer and defers loader
startup until initialization/resume is complete. This preserves legacy
seeded head initialization: all 2,143 tensors and their order match in both
directions. Its prebuilt-model path retains both the caller's model identity
and a serializable architecture config without rewinding the post-build RNG.
The public training API requires a native config; translating the original
configs and making CLI launchers usable is step 7.

`EPAIterBasedTrainLoop` uses native iteration hooks and validation loops,
with separate `max_iters=50000` and `stop_after=5000` settings. The cap does
not shorten the learning-rate or UDA horizon. Final validation precedes the
stopped checkpoint. Native `MedicalMetric` uses `target/mDice` for strict
greater best-checkpoint selection. Source diagnostics run separately.
Validation/test-only configs disable the training checkpoint hook.

`LegacyPolyLR` and `LegacyParamSchedulerHook` reproduce the legacy polynomial
and overlapping warmup arithmetic before each update. Both complete
50,000-by-2 LR curves are exactly equal to their frozen legacy traces.
Checkpoints retain the last used LR; the next iteration sets its own LR.
`PrecisionLoggerHook` emits native logging plus unrounded JSON scalars with
completed training iteration counters for validation records.

The reference baseline uses one GPU. Native unwrapped single-GPU UDA is
qualified. Multi-rank UDA is explicitly rejected by the runner/wrapper;
distributed synchronization across the multiple backward passes has not
been qualified. Disabled MIC, feature-distance, and dynamic-prototype branches
retain their algorithms and route backwards through the wrapper, but their
training behavior is outside this EPA/FDA gate.

## Checkpoint and data replay contract

Full stopped/periodic checkpoints contain native model, optimizer, scheduler,
and message-hub state plus versioned `epa_training` state:

- UDA iteration/configuration signature and horizon; fixed ETF tensors,
  cached source class means and class probabilities; student, teacher and
  registered prototype/BN buffers in the unchanged model state dict.
- Python, NumPy, CPU Torch and all CUDA RNG states, GPU count and backend
  precision/determinism flags.
- Loader/datalist/pipeline/sampling signature, epoch-start RNG state, epoch
  and consumed batch count, including subset indices for diagnostic runs.
- Parameter names with allocated gradients. Resuming restores zero tensors
  for these parameters before the next `zero_grad(set_to_none=False)`, so
  intermittently unused parameters retain the original AdamW behavior.

`EPACheckpointHook` defers a due periodic checkpoint until scheduled validation
has finished, even when the best score does not improve. Validation consumes
loader RNG, so saving only before it would resume from a different stream.
Final saves occur after validation and best selection as well.

`ReplayDataIterator` reconstructs the current epoch using its starting RNG.
With workers, it replays consumed batches to rebuild transforms/prefetch;
without workers, it advances indices only, then restores the checkpoint-time
parent RNG. It preserves subsequent model RNG draws and epoch transitions.
The qualified loader uses the legacy sampler, global generator, four
nonpersistent workers, default Linux fork/prefetch, and no pinning. Persistent
workers or explicit loader/sampler generators are rejected. Worker Python
and NumPy seeds use `legacy_worker_init`; the DataLoader-generated Torch
worker seed stays intact.

Replay uses pinned Torch 2.1 private sampler-iterator and worker-shutdown
interfaces. Large medical epochs can require replaying many consumed batches
on resume; this costs time but avoids serializing worker processes. Different
Torch versions or loader policies need qualification.

Resume requires a full native checkpoint, matching model/data/horizon/precision
settings, optimizer and scheduler state, and strict model key/shape loading.
Checkpoint loading stays on CPU before state transfer. Legacy checkpoints
support the already-qualified inference/weights-initialization path and cannot
resume native UDA training: they omit the necessary UDA/RNG/data state.
Native best checkpoints follow MMEngine's weights-only policy and are also
inference/initialization artifacts; resume from a full periodic/stopped
checkpoint. Unsupported resumes fail explicitly.

## Qualification and limits

Evidence is under
`work_dirs/migration_baseline/step6_training_20261011/` (ignored artifacts).

| Check | Result |
| --- | --- |
| Complete LR curves | Exact for both directions, all 50,000 iterations |
| Loader resume | Exact batches and parent RNG; 0/4 workers; mid-epoch and across epoch boundaries |
| Native CPU runner | Uninterrupted 10 updates versus stop at 2/resume to 10: every model/optimizer/scheduler/UDA/RNG/data tensor and counter exact, both worker counts; dropout, BN and intermittent unused parameters exercised |
| Real EPA/FDA initialization/groups | 2,143 tensors and 2,134 groups exact in both directions |
| Real early training | Two full updates per direction, batch 4 source + 4 target, four workers, finite gradients/losses, exact EMA and one step per update |
| Target-loss boundary | Three real updates per direction at artificial local iterations 1999/2000/2001; source/target loss keys, 2/4/4 backward counts, EMA/BN and LR verified |
| Native CUDA resume | Saved model, optimizer and UDA tensors load exactly; subsequent RNG/data/counters exact; legacy-precision floating continuation differences retained as diagnostics |
| Deterministic CUDA resume | CT -> MR artificial boundary, checkpoint at 2000/resume through 2002: all 2,143 model tensors, optimizer/scheduler/UDA tensors and RNG/data state exactly equal to uninterrupted control |
| Public API | Prebuilt model identity, serializable config and two native updates passed |
| Checkpoint evaluation | Native ValLoop target and TestLoop source evaluation passed in both directions, two real validation slices per domain |
| Unsupported resume | Legacy weights and native best/weights-only checkpoints rejected |
| Python checks | 21 changed files compile and pass Flake8; standalone isort/YAPF applied; full pre-commit suite not run |

GPU qualification uses saved real crops repeated into a 12-pair diagnostic
epoch, target validation every update on two slices, and a 50,000-step
schedule. Boundary runs deliberately start an initialized model at local
iteration 1999; they are control-flow tests, not 2002-step trained models.
No model has yet been retrained for 5,000 updates on the modern stack.

With the matched legacy settings (TF32 and cuDNN benchmark enabled,
determinism disabled), CUDA continuation is not bitwise reproducible.
The boundary diagnostic has a maximum parameter difference of about
`6.12e-4` and fails the diagnostic `atol=1e-5, rtol=1e-4` check. Its loaded
state and RNG/data are exact. The separate deterministic control disables
benchmarking, enables deterministic CUDA algorithms and sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; it passes exact continuation. These control
settings do not change the selected matched-training precision policy.
Step 8 must calibrate numerical probes before modern accuracy results;
the existing final/best mDice acceptance thresholds remain unchanged.

At the modern two-update checkpoints, two-slice diagnostic target/source
mDice is approximately CT -> MR `0.009665/0.031682` and MR -> CT
`0.033216/0.012478`. These small subsets and untrained checkpoints establish
executable evaluation, not accuracy parity. Exact raw class metrics, configs,
seed and checkpoints are recorded in `evaluation_smoke.json`.

Reproduce on the host GPU from the repository root and branch `mmcv2`:

```bash
conda activate epa-mmcv2
python -u tools/verify_mmcv2_training.py --cuda
python -u tools/verify_mmcv2_training.py --api-only
python -u tools/verify_mmcv2_training.py --evaluation-smoke-only
```

`--control-only` runs the LR/loader/native CPU checks. `verification.json`,
`control.json`, per-run `observations.json`, full/best checkpoints,
`public_api.json`, `evaluation_smoke.json`, `python_checks.json` and
`manifest.json` contain the reproducible evidence. Original/native configs,
precision modes and artificial offsets are recorded with each diagnostic.

The implementation follows the qualified MMEngine 0.10.7
[model migration](https://mmengine.readthedocs.io/en/v0.10.7/migration/model.html),
[runner migration](https://mmengine.readthedocs.io/en/v0.10.7/migration/runner.html),
and [scheduler migration](https://mmengine.readthedocs.io/en/v0.10.7/migration/param_scheduler.html)
interfaces. Retained custom training, scheduler and loop behavior is explicit.

## Completed legacy baselines

Both frozen-source, seed-0 runs stopped at 5,000 with the 50,000-step horizon
preserved. Final and best checkpoints were evaluated on their full current
target validation slices and diagnostic source validation slices:

| Direction / selection | Target mDice | Source mDice | Target mASSD |
| --- | --- | --- | --- |
| CT -> MR final/best at 5000 | 0.6857276685 | 0.8769931395 | 2.2916152708 |
| MR -> CT final at 5000 | 0.8022870166 | 0.8292970936 | 1.6273337966 |
| MR -> CT best at 4000 | 0.8069711912 | 0.8180215443 | 1.6160183938 |

Full class metrics and hashes are in `legacy_baselines_complete.json` and
the original evaluation summaries linked from it. Their generic historical
capture utility labels every evaluation as an existing best checkpoint;
the explicit checkpoint path and final/best selection in this manifest are
authoritative. Step 1 is now complete. Step 7 translates original configs
(including native nested UDA dataset construction) and ports the launchers;
steps 8/9 establish numerical and measured accuracy parity.
