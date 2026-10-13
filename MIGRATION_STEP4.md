# Step 4: foundational APIs and model construction

Completed on `mmcv2`, 2026-10-11. All edits and progress updates must remain
on `mmcv2`; leave `main` unchanged.

Both selected CT -> MR and MR -> CT EPA/FDA configurations build the local
DACS/SegFormer MiT-B5 model under Python 3.11.17, Torch 2.1.0+cu118, MMCV
2.1.0 and MMEngine 0.10.7. Each model matches every saved legacy
initialization tensor exactly. Full data loading and UDA runner integration
remain for steps 5 and 6.

## Changes

- Added explicit `mmseg` registry scopes for models, datasets, transforms,
  metrics, hooks, pixel samplers, weight initializers, optimizers and optimizer
  wrappers/constructors. Existing builder aliases and registered class names
  remain available.
- Moved model `BaseModule`/`Sequential`, initialization helpers, checkpoint
  loading, wrapper detection, logging, config parsing, file I/O and generic
  utility imports to MMEngine or native Torch. Retained MMCV CNN/image
  operations. The
  [official API mapping](https://mmcv.readthedocs.io/en/latest/get_started/api_reference.html)
  documents these relocated interfaces.
- Removed model precision decorators. The qualified baseline uses FP32;
  AMP requires separate qualification. The
  [MMEngine model migration guide](https://mmengine.readthedocs.io/en/latest/migration/model.html)
  describes modern model/training integration; the UDA `train_step` port is
  intentionally handled in step 6.
- Record pretrained backbone missing/unexpected keys. Both student and EMA
  backbones load `mit_b5_mmseg.pth` with no missing or unexpected keys.
- Decoupled model imports from the obsolete runner evaluation/logger hooks.
  Those hook implementations will be replaced in steps 5/6; they are no
  longer eagerly exported from the core or utility packages.
- Marked the local fork `0.16.0+mmcv2` and restricted its dependency guard to
  the qualified MMCV 2.1.0/MMEngine 0.10.7 pins. This identifies the source
  port; it does not claim completed training or accuracy qualification.

The old `DataContainer`, collation/scatter, runner, hook and CLI precision
paths still require their scheduled ports. Modern training/evaluation
entrypoints are not ready yet. The source-model verification below bypasses
those paths using saved plain-tensor batches.

## Legacy baseline isolation

Before changing tracked source, `tools/isolate_legacy_baselines.py` copied
and verified 247 legacy source/utility/prototype files into
`work_dirs/migration_baseline/legacy_source/`. The manifest is
`work_dirs/migration_baseline/legacy_source_manifest.json`. The snapshot
shares data, pretrained weights and branch metadata through symlinks; its
source and prototype contents are frozen and hashed.

The waiting supervisor was handed off without restarting or signaling the
active CT -> MR trainer. That process retains its already loaded legacy
modules and workers. Future checkpoint evaluations and MR -> CT training
run from the frozen source with the unchanged `epa` environment. The new
supervisor checks the frozen hashes before launches, and verifies completed
updates, UDA local iteration, and configured horizon against 5,000/50,000.
The source snapshot imports legacy models/datasets/APIs and resolves both
original configs successfully.

Status remains at `work_dirs/migration_baseline/selected_baselines_status.json`.
Handoff provenance is in `selected_baselines_status_before_isolation.json`;
the new supervisor log is `isolated_supervisor.log`. Final/best evaluations
are generated in the snapshot's `work_dirs`, with links from the original
baseline run directories after successful evaluation. Never modify the
frozen source files. Use its legacy tools and working directory for further
legacy captures rather than importing the ported source with Python 3.8.

## Validation

Evidence root: `work_dirs/migration_baseline/step4_models_20261011/`.
The final model qualification is `qualification_v2/verification.json`.

| Check | CT -> MR | MR -> CT |
| --- | --- | --- |
| Initialized state keys/order/shapes/dtypes/values | All 2,143 tensors exact | All 2,143 tensors exact |
| Named parameters | 2,134, original names | 2,134, original names |
| MMEngine optimizer group names/order/settings | All 2,134 groups match legacy | All 2,134 groups match legacy |
| Student/EMA pretrained backbone loading | No missing/unexpected keys | No missing/unexpected keys |
| BatchNorm | Two native BN2d modules, original eps/momentum/affine/statistics | Same |
| CUDA source-training smoke | One update, finite FP32 loss/gradients | One update, finite FP32 loss/gradients |
| Historical checkpoint student loading | Strict, original keys/shapes | Strict, original keys/shapes |
| Fixed source-slice checkpoint prediction | 100% argmax agreement | 100% argmax agreement |

The smoke uses one saved source image, native MMEngine `OptimWrapper` and
AdamW with `foreach=False`, `fused=False`, and `zero_grad(set_to_none=False)`.
Its group inventory matches the legacy optimizer. It exercises source
segmentation, backward and one update; full FDA/ETF/EMA update order is
tested after the step-6 port. It does not constitute a matched training run.

Both configs retain 50,000 as the configured runner/UDA horizon. BatchNorm
is native `BatchNorm2d`, eps `1e-5`, momentum `0.1`, affine enabled and
running statistics enabled. Float state remains FP32. No checkpoint key
conversion is needed for student inference; extracting the existing
`model.` prefix follows the legacy evaluator. Full UDA/optimizer resume
conversion is addressed in step 6.

Checkpoint logits on the fixed source slice have maximum absolute
differences of about `0.00316` (CT -> MR) and `0.00541` (MR -> CT), while all
predicted pixels agree. These are diagnostics with the frozen TF32/cuDNN
flags; numerical thresholds are calibrated in step 8, and complete split
accuracy comparisons remain in step 9. Peak Torch allocation during the
CUDA probes was about 2.44 GiB and 1.64 GiB respectively, with a 25% allocator
cap. The existing cuDNN `nvrtc.so` workaround warning remains for step 8's
performance/numerical investigation.

Additional checks passed for config CLI overrides, Python/JSON dumps and
model graph construction, parameter utility/experiment module imports,
logger identity, class/palette helpers and MMEngine environment collection.
Changed Python files compile; isort follows `setup.cfg`; targeted Flake8
passes with an E402 exception for existing scripts that bootstrap the
repository import path. Formatting/lint tools were installed in `/tmp`,
leaving both training environments unchanged. The full pre-commit hook
suite was not run. Utility evidence is in `utility_checks/`.

Reproduce the model gate from the repository root on `mmcv2` with host GPU
access and the step-1 artifacts available, using an unused output directory:

```bash
/home/akash/miniconda3/envs/epa-mmcv2/bin/python -u tools/verify_mmcv2_model.py \
  --out-dir work_dirs/migration_baseline/model_recheck --device cuda
```

The verifier checks the `mmcv2` branch and the local package location before
building the models. Artifacts, datasets and checkpoints remain ignored.
Step 5 is next: replace data transport and preserve preprocessing and
medical metric semantics, including the legacy ASSD formula.
