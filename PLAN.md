# MMCV 2.x and Python migration plan

Created: 2026-10-11
Completed and accepted by the user: 2026-10-13

## Objective and completion criteria

Migrate the local EPA/MIC/MMSegmentation fork to Python 3.11 and MMCV
2.0+ with MMEngine, then train EPA/FDA in both CT -> MR and MR -> CT from the same initial
weights and establish that segmentation accuracy remains comparable.
Preserve the experiment's mathematical behavior while changing framework
interfaces. Successful imports or a short training run alone do not establish
accuracy parity.

This document covers the implementation and validation work to follow.
Repository inspection, documentation research, a legacy data/environment
audit, both supplied reference-checkpoint evaluations, and fixed-batch legacy
training updates for three candidate configs have been completed. Step 3 is
complete: the isolated Python 3.11/MMCV 2 environment is installed, locked,
and qualified on CUDA. Step 4's foundational API/model port passed exact
initialization, checkpoint-loading and source-training smoke checks. Data
transport and evaluation passed step 5. Step 6 qualified the native UDA runner
for the selected single-GPU EPA/FDA contract. Both legacy baselines finished
at the user-requested
5,000-iteration cap with `runner.max_iters=50000` retained.
The corrected MR -> CT reference also reached 5,000; its final/best
evaluations are complete after recovering a frozen-helper output-path rejection.
Both modern campaigns completed 5,000 updates and planned evaluations.
The user accepted the migration on 2026-10-13. Version 5 accepts each stack's
best target checkpoint, allows improvements, excludes source/final scores,
and treats lower ASSD as better. CT -> MR best improves mean Dice 4.499
points and mean ASSD 4.73%. MR -> CT best loses 1.215 mean Dice points and
3.965 AA points, with mean ASSD increasing 3.34%; these differences are
explicitly accepted by the user. Earlier automatic failures remain preserved.
No further accuracy runs are required. Migration is complete for the pinned
single-GPU EPA/FDA configs and 5,000-update comparison scope.
See `MIGRATION_ACCEPTANCE.md` for the decision, evidence and repository handoff.

Completion requires:

- A reproducible, pinned modern environment and working train/test entrypoints.
- Equivalent resolved experiment settings, data preprocessing, losses,
  optimizer groups, learning-rate schedule, and UDA update order.
- Evaluation parity on fixed predictions and an existing checkpoint.
- Modern-stack training in both directions, stopped at 5,000 iterations, compared
  with matching legacy baselines while retaining the 50,000-iteration schedule.
- Results recording config, seed, initialization, evaluation checkpoint,
  best target metrics, environment, and remaining limitations.

## Branch isolation

All edits and all progress updates for this migration must be made on the
`mmcv2` branch. This includes code, configs, dependencies, documentation,
`PLAN.md`, task checkboxes, status tables, experiment records, and progress
log entries. The branch was created from `main` at commit `4a589d1` on
2026-10-11. Leave `main` unchanged and keep the legacy environment available
for baseline comparisons.

Before edits or commits, verify `git branch --show-current` reports `mmcv2`.
If another branch is active, switch to `mmcv2` before making any edits or
recording progress.
Keep migration commits on this branch; merging into `main` is a separate
action requiring a later user instruction. Include `PLAN.md` and migration
documentation in the migration commit before switching branches, since
uncommitted files can follow a checkout.

## Findings from the initial inspection

| Area | Observed state | Migration implication |
| --- | --- | --- |
| Local framework | `mmseg/version.py` reports 0.16.0; this is a customized fork | Installing upstream MMSeg alone will not port EPA/MIC behavior |
| Installed legacy environment | `/home/akash/miniconda3/envs/epa/bin/python`: Python 3.8.20, Torch 1.7.1+cu110, torchvision 0.8.2+cu110, mmcv-full 1.3.7 | Preserve this environment for baseline runs |
| Other installed packages | NumPy 1.19.2, timm 0.3.2, MedPy 0.4.0; MMEngine absent | Audit compatibility beyond MMCV |
| Version guard | `mmseg/__init__.py` accepts MMCV 1.3.7 through 1.4.0 | Change the guard only alongside a real API port |
| Framework dependencies | Runner, optimizer construction, hooks, registries, checkpoints, parallel wrappers, `DataContainer`, collate/scatter | Requires structural changes, not only import substitutions |
| UDA training | `DACS.train_step()` zeroes gradients and steps the optimizer; `forward_train()` performs multiple backward passes | Ensure exactly one optimizer step and preserve backward/EMA ordering |
| Data and initialization | `data/data_np/`, `data/datalist/`, and `pretrained/mit_b5_mmseg.pth` exist | Verify list resolution and hashes before training |
| Reference weights | `pretrained/MR2CT_best_mDice.pth` and `pretrained/CT2MR_best_mDice.pth` exist | Inspect metadata/config compatibility before using as references |
| Prior run records | No `work_dirs/` or `configs/generated/` directory found | No accuracy baseline has been established from local run logs |
| GPU visibility | Sandbox hides GPU devices; host access exposes an RTX 4080 Laptop GPU, driver 595.99.02, 12,282 MiB | Run GPU checks/training with host access; legacy CUDA forward/backward verified |

The on-disk data/weights differ from the absence note in `AGENTS.md`.
Dataset integrity and checkpoint compatibility have been audited in step 1.

MMCV 2.x removes framework facilities including `mmcv.runner` and
`mmcv.parallel`; their migration destinations are documented in the
[official API reference](https://mmcv.readthedocs.io/en/latest/get_started/api_reference.html).
MMEngine supplies the modern training infrastructure, and MMSegmentation's
[migration guide](https://mmsegmentation.readthedocs.io/en/latest/migration/interface.html)
describes its modern config and data interfaces.

## Proposed technical direction

Port the existing local fork onto MMCV 2.x and MMEngine, using MMSegmentation
1.x interfaces as a reference. Keep the existing MiT backbone, SegFormer head,
losses, EPA prototypes, DACS, FDA, and MIC implementations during the first
parity pass. Replacing them with similarly named upstream implementations
would introduce another source of accuracy changes.

Use native MMEngine runner/optimizer/evaluator interfaces. Add small, explicit
adapters where the existing UDA tensor/metadata contract needs preservation.
Choose either modern segmentation data samples or a documented custom UDA
batch representation; do not depend on removed `DataContainer` machinery.
Keep the local `mmseg` package authoritative and verify `mmseg.__file__` so an
installed upstream package cannot silently shadow it.

### Qualified environment (step 3)

The following stack passed dependency and CUDA-operation checks on the
training host. Exact setup commands and qualification evidence are in
`ENVIRONMENT_MMCV2.md`; all Python distributions are pinned in
`requirements-mmcv2.lock.txt`. Source and full-model migration remain pending.

| Component | Qualified version | Reason |
| --- | --- | --- |
| Python | 3.11.17 | Newer than the current 3.8; matching MMCV wheel verified |
| PyTorch / torchvision | 2.1.0+cu118 / 0.16.0+cu118 | Official pair; CUDA forward/backward verified |
| CUDA runtime | 11.8 | Working on the RTX 4080 Laptop GPU and current driver |
| MMCV | 2.1.0 (`mmcv`, with compiled ops) | Conservative 2.x starting point compatible with the upstream reference |
| MMEngine | 0.10.7 | Both selected configs load and dump |
| NumPy | 1.26.4 | Dependency checks and saved numerical probes passed |
| Remaining dependencies | Individually audited and pinned | Includes timm, Kornia, MedPy, SciPy, OpenCV, logging and visualization |

The [official MMCV wheel index](https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html)
contains a Linux x86_64 Python 3.11 wheel for MMCV 2.1.0 with Torch 2.1/CUDA
11.8. The [PyTorch version instructions](https://pytorch.org/get-started/previous-versions/)
list the corresponding Torch/torchvision pair. CUDA convolution, NMS,
ROIAlign forward/backward, and FFT roundtrip passed on the training host.

The [MMCV main-branch version file](https://github.com/open-mmlab/mmcv/blob/main/mmcv/version.py)
currently reports 2.2.0. However, the
[MMSegmentation 1.2.2 version guard](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/__init__.py)
requires MMCV below 2.2.0. That upstream constraint is relevant when borrowing
its interfaces, but does not automatically define our local fork's tested
support. Version 2.1.0 passed the environment gate; treat 2.2.0 or a newer
source revision as a separate qualification if needed. This stack upgrades the legacy
stack; it does not claim to use the newest release of every dependency.

For 2.x, the full package is named `mmcv`; the reduced package is
`mmcv-lite`. Install one distribution, following the
[MMCV installation guidance](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
Prefer a matching binary wheel. A GitHub checkout/build is a fallback for a
missing wheel or a specific required fix, not a prerequisite for this plan.

## Stepwise execution plan

### 1. Freeze the exact experiment and legacy baseline

- [x] User selected EPA/FDA in both CT -> MR and MR -> CT. Use the current
  `configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`
  and `configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`.
  Freeze both fully resolved configs, including runtime model settings.
  Use seed 0, one GPU, and no hyperparameter overrides; only work directories
  differ. The earlier CE-only candidate is outside the selected comparison.
- [x] Record repository revision, environment package list, GPU/driver/CUDA,
  datalist hashes, dataset file manifest, and initialization/prototype hashes.
- [x] Check all listed paths, paired image/label counts, label IDs, and train,
  validation, and test split boundaries.
- [x] Inspect the supplied best-checkpoint metadata. Use a checkpoint only
  with a compatible architecture/config and known evaluation split.
- [x] Train both legacy baselines to the user-requested stop at 5,000
  iterations. Keep `runner.max_iters=50000` and its LR horizon unchanged.
  Save iteration-5,000 and best checkpoints, learning-rate/loss curves,
  per-class Dice, mDice, and ASSD.
- [x] Capture fixed inputs, transformed batches, model outputs, and optimizer
  traces for later comparison. Keep these artifacts outside version control.

Both selected configs use MiT-B5/SegFormer, five classes, seed 0,
256x256 crops, four samples and four workers per GPU, AdamW at `6e-5`,
50,000 iterations, 1,500 warmup iterations, linear polynomial decay, and
validation every 500 iterations. They use no-mix DACS, source/target ETF
losses (weight 0.2, temperature 0.05), Dice+CE (0.5 each), FDA source-to-target
(probability 1, L=0.01), target-training start at iteration 2,000, EMA alpha
0.99, and pseudo-label threshold 0.968. Preserve these settings unchanged.

The primary checkpoint-selection and migration-comparison protocol uses the
current configs: MR validation slices (2,400) for CT -> MR, and CT validation
slices (1,200) for MR -> CT; foreground mDice selects the best checkpoint.
Historical MR -> CT CT test-volume results remain supplemental evidence,
and must not be used as the accuracy target for the slice-validation run.
Run the two baselines sequentially on the single GPU, from the original
MiT weights rather than resuming the supplied best checkpoints. Per the user's
updated instruction, stop each run after exactly 5,000 optimizer updates. This
is a fixed iteration cap; validation still runs every 500 iterations. Keep
`runner.max_iters=50000`, the LR schedule, and UDA horizon unchanged, so the
first 5,000 updates retain the original schedule rather than compressing decay.
Apply the same stopping rule to both migrated runs. Accuracy comparisons are
for the 5,000-step protocol; 50,000-step accuracy is outside the current scope.

`tools/train_legacy_baseline.py` adds a final-priority stop hook around the
existing legacy trainer. After validation and logging at update 5,000, it saves
`iter_5000.pth` and `latest.pth`, records `early_stop.json`, and closes the hooks.
The original CT -> MR process could not accept a new hook while running, so it
was stopped and restarted from seed 0. Its partial logs/checkpoint remain in
`ct2mr_epa_fda_seed0/`; the replacement runs use `_seed0_5k/` directories.
`selected_baselines_status_before_5k.json` preserves the superseded queue.
A real MMCV runner control check verified exact stopping, checkpoint iteration,
optimizer state, hook ordering/cleanup, and preservation of the 50,000-step LR
horizon: audit `early_stop_control_check/control_check.json`.

**Deliverable/gate:** A baseline manifest identifying the exact experiment,
checkpoint selection rule, data, and measured accuracy before claiming parity.
Baseline capture can proceed while independent migration work is prepared.

Step 1 evidence is under
`work_dirs/migration_baseline/step1_audit_20261011/` (ignored run artifacts):

- `manifest.json`, `environment.json`, `packages.txt`, `source_files.json`,
  and `prototypes.json`: code/environment and initialization provenance.
- `configs/`: three resolved candidates with runtime model settings; both
  EPA/FDA configs are now selected. The CE-only capture is diagnostic only.
- `data_audit.json` and `dataset_files.jsonl`: hashes and array checks for all
  43,208 files, including all image/label pairs and eight NPZ test volumes.
  All files passed readability, finite-value, label-ID, shape, and pairing
  checks. No shared file paths were found between train/validation/test lists.
  The one shared image hash per domain between train and validation consists
  of constant-valued slices; patient-level overlap remains unverified because
  slice filenames do not identify patients.
- `checkpoints.json` and `checkpoints/`: supplied checkpoint metadata, original
  embedded configs, tensor inventories, and SHA-256 hashes.
- `reference_ct2mr/`: strict student checkpoint load, fixed raw/transformed
  inputs and logits, complete target/source predictions, and medical metrics.
- `reference_mr2ct/`: the historical CT test-volume evaluation, plus separate
  MR validation-slice source diagnostics and fixed parity inputs/outputs.
- `reference_mr2ct_current_slices/`: the same supplied checkpoint evaluated on
  the selected current CT validation-slice protocol; target mDice 0.8394315943
  and mASSD 1.4787344584. No delta against historical volume metadata is defined.
- `final_trace_ct2mr_epa/`, `final_trace_ct2mr_nomix/`, and
  `final_trace_mr2ct_epa/`: initialized UDA state, exact optimizer parameter
  groups, all 50,000 LR values, and two actual gradient/update snapshots on
  saved four-sample batches. Student/teacher EMA parameters and BatchNorm
  buffers are captured; finite gradients, exact EMA updates, and batch-counter
  increments were verified in `trace_verification.json`. These are diagnostic
  traces, not full-run training evidence. Earlier `trace_*` captures are
  superseded by these final captures.
- `legacy_source.tar.gz` and `capture_scripts/`: preserved baseline source and
  utilities. Each final training trace records its utility/config/batch hashes.

The supplied CT -> MR checkpoint matches the current
`ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py` model, loss,
optimizer, UDA settings, and MR validation-slice evaluation. It is seed 0,
best at iteration 3,500, and records mDice 0.7121535424. Re-evaluation on
the host GPU gives target mDice 0.7121525915 and diagnostic CT source mDice
0.8730193886. This supports checkpoint inference parity; fresh iteration-5,000
and best checkpoint results are now recorded in the accuracy table below.

The MR -> CT checkpoint is seed 0, best at iteration 8,000, with historical
mDice 0.8522451077. Re-evaluation gives target mDice 0.8522439763 and diagnostic
MR source mDice 0.8320427866. Its embedded config evaluates CT NPZ test volumes, while
the current top-level MR -> CT FDA config evaluates CT validation slices.
These evaluation protocols must remain distinct. Historical checkpoint
selection used the test-volume list, so preserve that provenance explicitly
as supplemental evidence. Both new baselines use the current validation-slice protocol.

Reproducible audit/capture utilities were added in `tools/`; each checks that
the active branch is `mmcv2` and writes artifacts under `work_dirs/`.
No existing training, dataset, model, or experiment-config implementation has
been changed. The first two CT -> MR EPA updates used about 4.35 GB peak
allocated GPU memory; later target-training branches remain to be exercised.
The legacy environment enables TF32 and cuDNN benchmarking; minor repeated-run
numerical differences were observed. Preserve and document these settings in
the comparison contract rather than assuming deterministic execution.

### 2. Specify behavior parity and acceptance thresholds

- [x] Freeze architecture, initialization, losses and weights, class mapping,
  label ignore value, transforms, optimizer settings, iterations, and batch
  size. Config syntax may change; effective experiment settings must match.
- [x] Freeze augmentation probabilities, source/target pairing, sampler,
  worker seeding, `drop_last`, validation frequency, and inference resizing.
- [x] Record precision and determinism settings. Use FP32 initially and
  explicitly control TF32, cuDNN, and AdamW execution options; postpone AMP,
  compilation, fused optimizers, and performance tuning.
- [x] Set checkpoint selection on the same validation metric and interval.
  Compare both final checkpoints and validation-selected best checkpoints.
- [x] Adopt the version-1 working accuracy tolerance below before inspecting
  modern training results; record any changes to this protocol with their reason.

The primary accuracy metric is the repository's foreground-only `mDice` over
`Myo`, `LAC`, `LVC`, and `AA`, excluding background. `MedNpyListDataset`
aggregates class intersections/areas across slices; `MedNpzVolumeDataset`
reconstructs volumes and uses its own aggregation. Preserve both definitions
and do not compare scores computed on different splits or aggregation units.

Current version-5 acceptance, explicitly authorized by the user on
2026-10-13: the current best-target results are acceptable and migration is
successful. Earlier automatic Dice tolerances remain historical evidence;
MR -> CT measured deviations are user-accepted. Source/final scores are
excluded; lower ASSD is better. No additional accuracy runs are required.
Structural/numerical gates are unchanged. See `MIGRATION_ACCEPTANCE.md`.

Frozen version-1 working criteria, adopted before modern results:

- Primary target-domain mDice: absolute difference no greater than **0.01**
  on the 0-1 scale, equivalent to **1 Dice percentage point**.
- Per-class foreground Dice: flag differences greater than **0.02** for
  investigation; the aggregate score must not hide a class regression.
- Source-domain metrics and ASSD: record alongside target Dice and investigate
  material regressions. ASSD remains diagnostic with no accuracy cutoff in
  version 1; record any justified revision before inspecting modern results.
- First comparison: the exact selected seed (0 in both directions). If the result
  misses tolerance or baseline variability is unknown, run matched seeds
  0/1/2 on both stacks and report paired differences, means, and standard
  deviations. Do not use variance to waive a discrepancy without diagnosis.
- A substantially better result outside the symmetric tolerance also warrants
  inspection for changed metrics, split leakage, or altered training behavior.

For identical fixed predictions, finite Dice/ASSD values must agree within
absolute `1e-10`, with identical NaN locations and empty-mask outcomes. For
logits/losses/gradients under changed Torch/CUDA kernels, record
maximum absolute/relative error and prediction disagreement; set numerical
tolerances using legacy repeatability before judging the modern outputs.

**Deliverable/gate:** A written comparison contract with thresholds, seeds,
metric units, splits, and best/final checkpoint rules.

Version 1 is frozen in [MIGRATION_CONTRACT.md](MIGRATION_CONTRACT.md) and
`work_dirs/migration_baseline/comparison_contract_v1.json`. Both resolved
configs, all 10 datalist hashes, identical 2,134-group optimizer inventories,
and identical full LR curves were checked. Settings inferred from actual
legacy code include the 80% outer color-jitter probability, non-persistent
workers under Torch 1.7, separate worker seeding rules, and
`zero_grad(set_to_none=False)`. The migrated runs must preserve these details.
Numerical logit/loss/gradient tolerance calibration remains an explicit step-8
gate to complete on legacy repeatability before viewing modern outputs.

This document work did not alter the `epa` environment or shared legacy
training source. Isolated modern environment preparation can proceed while
step 1 runs. The queued jobs verify tracked legacy source hashes before
launching: defer shared framework edits and adding new tracked files until
they finish, or explicitly isolate and reconfigure their baseline execution.

### 3. Create and verify an isolated modern environment

- [x] Verify training-host GPU visibility, driver support, GPU architecture,
  memory, and storage. Resolve the observed GPU visibility failure before
  starting GPU tests or baseline training.
- [x] Preserve the existing `epa` environment and create a separate Python
  3.11 environment, for example `epa-mmcv2`.
- [x] Install the selected Torch/torchvision pair first, then MMEngine and a
  matching MMCV 2.x wheel. Check installed distribution names and versions.
- [x] Audit every old requirement for Python/Torch compatibility. Inspect
  timm import paths, Kornia transforms, MedPy behavior, NumPy aliases, image
  resize behavior, and optional imports loaded by `mmseg` initialization.
- [x] Record exact pins in a modern environment/requirements file; run
  dependency consistency checks and small CUDA/compiled-op checks.
- [x] If a source build is required, clone MMCV into an isolated scratch
  directory, select a release tag or exact commit, match Torch/CUDA/compiler
  and GPU architecture settings, and retain the build log and wheel.

Completed 2026-10-11: `epa-mmcv2`, Python 3.11.17, Torch 2.1.0+cu118,
MMCV 2.1.0 and MMEngine 0.10.7 qualified. The official wheel worked, so the
conditional source build was unnecessary. Evidence:
`work_dirs/migration_baseline/step3_environment_20261011/verification_v2/verification.json`.
Dependency consistency passed and all 78 Python packages are locked.
Original tracked source and all 81 legacy package versions match the audit.
MedPy's newer ASSD aggregation differs; the verified legacy formula must be
implemented in step 5. Torch's cuDNN `nvrtc.so` workaround warning is recorded
for full-model performance/numerical checks in step 8. See
`ENVIRONMENT_MMCV2.md` for reproduction and the complete dependency audit.

Use the [official source-build instructions](https://mmcv.readthedocs.io/en/latest/get_started/build.html)
for the fallback, including its installation validation script. If the needed
fix exists only on the latest GitHub branch, pin that commit and rerun the
compatibility gates; an unpinned branch is not a reproducible dependency.

**Deliverable/gate:** Locked modern environment with working Torch CUDA and
MMCV ops on the intended training host. CPU-only checks can advance the port,
but do not satisfy this gate.

### 4. Port foundational APIs and model construction

- [x] Replace `Config`/`DictAction`, registries, logging, file I/O, environment
  collection, distributed helpers, and checkpoint utilities with their
  documented MMEngine counterparts.
- [x] Establish explicit project registries/scopes for models, datasets,
  transforms, metrics, and hooks. Preserve UDA construction and class names.
- [x] Port `BaseModule`/`Sequential`, initialization helpers, and backbone
  checkpoint loading. Replace obsolete precision decorators deliberately
  while preserving initial FP32 behavior.
- [x] Review retained MMCV CNN/image utilities against the selected 2.x API;
  remove reliance on obsolete registry and wrapper internals.
- [x] Verify pretrained weight keys, shapes, BN configuration, initialization
  ordering, and loaded/missing keys. Keep model parameter names stable where
  possible and document explicit checkpoint key conversion if required.
- [x] Update the local version guard and fork version metadata to describe
  tested support after import/model construction checks pass.

Completed 2026-10-11 on `mmcv2`. Both selected EPA/FDA models build on the
modern stack and exactly match all 2,143 legacy initialization tensors.
All 2,134 optimizer groups match in names/order/settings; student and EMA
pretrained backbones load with no missing/unexpected keys. CUDA source
training smoke and strict historical student checkpoint loading passed in
both directions. Fixed source-slice predictions agree at every pixel;
floating-point logit differences remain for step-8 calibration.
Evidence: `MIGRATION_STEP4.md` and
`work_dirs/migration_baseline/step4_models_20261011/qualification_v2/verification.json`.

Before source edits, 247 legacy files were copied and verified in
`work_dirs/migration_baseline/legacy_source`. The supervisor handoff preserved
the active CT -> MR process; subsequent evaluations and MR -> CT training
use frozen source. Source hash guards now validate that snapshot. Both
training environments and original configs remain unchanged. DataContainer,
runner/hook and training/evaluation CLI contracts are handled in steps 5-7;
obsolete hooks are no longer eagerly imported by the model package.

Primary files: `mmseg/__init__.py`, model/dataset builders,
`mmseg/models/backbones/mix_transformer.py`, decode heads, base segmentors,
`mmseg/utils/`, and auxiliary tools that import legacy APIs.

**Deliverable/gate:** Local package imports and builds the exact baseline
model on the modern stack and loads its initialization correctly.

### 5. Port data transport and medical evaluation

- [x] Replace `DataContainer`, legacy collation/scatter, and formatting with
  an explicit modern packing/collation/device-transfer contract for both
  source and target tensors and their metadata.
- [x] Preserve NPY scaling (`input_range='-1_1'`), channel replication/order,
  means/stds, transform order, crop/flip RNG, ignore value 255, and label IDs.
  Apply normalization and padding exactly once.
- [x] Preserve `img_norm_cfg` metadata used by DACS/FDA/MIC, inference image
  sizes, and all currently used UDA batch keys.
- [x] Preserve NPZ slicing, axis flips, slice ordering, and volume
  reconstruction. Keep the original dataset path resolution behavior.
- [x] Implement MMEngine-compatible medical evaluators preserving foreground
  exclusion, aggregation, ASSD units, NaN handling, and empty-mask fallback.
  Do not substitute a generic metric without demonstrating equivalence.
- [x] Preserve source/target prefixes when the selected config has multiple
  evaluation datasets. Add separate source evaluation reporting if needed,
  without changing the target validation/checkpoint selection rule.
- [x] Compare legacy/modern transformed tensors for fixed crop/flip decisions
  and metric outputs for identical saved predictions.

Primary files: `mmseg/datasets/builder.py`, `uda_dataset.py`, `med_npy.py`,
`med_npz_volume.py`, pipeline formatting/loading/transforms, and evaluation
utilities. Native packing is described in the
[MMSeg package migration guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/migration/package.md).

**Deliverable/gate:** Fixed-input preprocessing and fixed-prediction medical
metrics match the legacy behavior, including class and volume edge cases.

Completed 2026-10-11 on `mmcv2`. Real four-pair batches in both directions
match all legacy tensors and metadata exactly, including CUDA transfer.
Three crop/flip seeds, variable-size padding, both worker seeds, and six
sampler orders/RNG states pass. Six NPZ axis/flip cases and empty-mask cases
pass. Five full historical prediction sets (8,224 slices) and the fresh
CT -> MR 5,000-step source/target sets reproduce every medical metric
exactly. Real historical student checkpoint inference passes both APIs.
Evidence: `MIGRATION_STEP5.md` and
`work_dirs/migration_baseline/step5_data_20261011/`.

The port explicitly preserves old RandomSampler RNG semantics, effectively
unpinned legacy containers, MedPy 0.4 directional-mean ASSD, and NPZ
both-empty Dice=0. Native training integration, full-size run sequence and
distributed execution qualification remain in subsequent steps.

### 6. Port training, UDA updates, and checkpoints

- [x] Replace the MMCV runner with MMEngine iteration-based training,
  validation/test loops, hooks, and logging. Follow the
  [runner migration guide](https://mmengine.readthedocs.io/en/latest/migration/runner.html).
- [x] Implement DACS's modern `train_step(data, optim_wrapper)` contract.
  Preserve multiple backward passes and one optimizer update per iteration;
  use the wrapper's gradient/step facilities without a second automatic
  backward pass or optimizer step. See the
  [model migration guide](https://mmengine.readthedocs.io/en/latest/migration/model.html)
  and [OptimWrapper API](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.optim.OptimWrapper.html).
- [x] Preserve selected source/target losses, target start thresholds,
  pseudo-label weighting, EMA timing, teacher dropout/BN behavior, EPA
  prototypes and FDA. Disabled feature-distance, MIC and dynamic-prototype
  algorithms are retained; enabling them needs separate qualification.
- [x] Preserve AdamW parameter-group membership, head LR multiplier 10,
  normalization/position decay exclusions, optimizer defaults, and gradient
  clearing semantics. Compare group names and effective settings.
- [x] Port polynomial LR and warmup according to the
  [scheduler migration guide](https://mmengine.readthedocs.io/en/latest/migration/param_scheduler.html),
  then compare actual LR values across all iterations. Check warmup overlap
  and zero/one-based boundaries rather than assuming equivalence from names.
- [x] Replace single-GPU wrappers and result collection; qualify unused
  parameters and multiple backwards on the baseline's one GPU. Use native
  ordinary-model DDP infrastructure and explicitly reject multi-rank UDA;
  its gradient synchronization is outside the selected comparison.
- [x] Save/restore student, EMA teacher, prototype buffers, UDA iteration,
  optimizer, scheduler, and required RNG/sampler state. `local_iter` is
  currently a plain Python attribute; explicitly handle it during resume.
- [x] Separate old-checkpoint inference/initialization compatibility from
  full training resume. Document conversions and reject unsupported resumes.

Primary files: `mmseg/apis/train.py`, train/test APIs,
`mmseg/core/ddp_wrapper.py`, evaluation hooks, `mmseg/models/uda/`, and
base segmentors. Preserve upstream attribution and licenses.

**Deliverable/gate:** One complete UDA iteration has correct gradient and
update order; checkpoint round-trip/resume does not reset UDA state.

Completed 2026-10-11 on `mmcv2` for the selected single-GPU EPA/FDA runs.
Native training passes real four-pair updates and the iteration-2000 target
loss boundary in both directions. Initialization (2,143 tensors), optimizer
groups (2,134) and all 50,000 LR values match the frozen legacy contract.
Loader resume is exact with 0/4 workers; native CPU continuation and a
deterministic CUDA boundary continuation match every saved tensor/state
exactly. Legacy-precision CUDA reload/RNG/data state is exact, with
nondeterministic floating continuation differences retained for step 8.
Public native training API and target/source validation/test-loop smoke
checks pass. See `MIGRATION_STEP6.md` and
`work_dirs/migration_baseline/step6_training_20261011/verification.json`.

These are short fixed-batch diagnostics, including artificial iteration
offsets for the boundary checks; they are not modern accuracy results.
Original-config translation, native nested UDA dataset construction and
CLI integration are step 7. Both full legacy 5,000-step baselines and their
final/best evaluations have finished, completing step 1 as well.

### 7. Translate configs and keep launchers usable

- [x] Translate old `data` settings into train/val/test dataloaders;
  `runner`/`evaluation` into train/val/test loops and evaluators;
  `optimizer` into `optim_wrapper`; `lr_config` into `param_scheduler`; and
  runtime settings into hooks, randomness, environment, and visualization.
- [x] Translate `uda` construction and `max_iters` injection. Preserve debug
  interval propagation currently supplied through model training settings.
- [x] Provide a documented translation path so the original config remains
  a reproducible experiment specification. Fail on unsupported settings
  rather than silently dropping them.
- [x] Update `tools/train.py`, `tools/test.py`, `run_experiments.py`,
  `experiments.py`, inference, config printing, parameter counting, and
  representation-analysis utilities affected by removed APIs.
- [x] Verify the resolved modern config against the frozen legacy manifest,
  including all inherited model/UDA/dataset/schedule values.

**Deliverable/gate:** Both direct and launcher-generated runs build the same
effective experiment; evaluation arguments and resume behavior are documented.

Completed on `mmcv2`, 2026-10-11. The checked translator preserves both
frozen specifications and all 2,143 initial tensors per direction. Direct
and generated launches resolve to the same effective experiment. Real
4+4/four-worker CLI training stopped at two updates in both directions,
with the 50,000 horizon intact; full native resume and original/native
source/target evaluation passed. CPU schema controls agree exactly; small
GPU process differences remain recorded under the legacy precision policy.
A one-update original-config run also passed native target validation and
best/full checkpoint selection. NPY inference, strict weight loading,
parameter counting, native config printing and pre-logit extraction passed.
See `MIGRATION_STEP7.md` and step-7 verification artifacts. Modern matched
5,000-update training is not launched; numerical qualification is step 8.

### 8. Run numerical checks and training smoke checks

- [x] Load the same legacy evaluation checkpoint in both environments;
  compare logits/predictions on fixed slices and the same complete evaluation
  split. Resolve key/shape/preprocessing/metric mismatches before retraining.
- [x] Compare seeded initial weights, forward losses, selected gradients,
  first optimizer updates, EMA updates, and pseudo-label statistics on saved
  batches. Record numerical tolerances and discrepancy summaries.
- [x] Run a short real-data GPU training smoke check, then exercise a run
  crossing the selected config's target/prototype/MIC start boundaries.
  A two-iteration run does not exercise the selected 2,000-iteration
  target-training transition.
- [x] Verify finite losses/gradients, expected teacher/prototype updates,
  working validation, checkpoint selection, debug output, and logging.
- [x] Compare uninterrupted versus save/resume training through a UDA
  boundary. Test the configured distributed execution if applicable.
- [x] Exercise the selected active EPA/FDA branches. MIC, dynamic prototypes,
  AMP, distributed execution and other FDA geometries remain unqualified;
  their limitations are explicit in `MIGRATION_STEP8.md`.

Smoke-only configs may shorten intervals/iterations but must use separate run
directories. The accuracy run must restore the original experiment settings and use the
matched 5,000-update stop with the declared 50,000-iteration schedule.
Add focused regression checks for metric equivalence, LR traces, packing, and
UDA resume/update ordering, rather than a broad suite that mirrors code.

**Deliverable/gate:** Documented numerical and GPU smoke results covering the
baseline's active training branches; no unresolved behavior mismatch.

Completed on `mmcv2`, 2026-10-11, for both selected single-GPU configurations.
Legacy repeats froze tolerances before modern inspection; extra declared-asset
repeats fit those immutable limits. Corrected real-inverse FDA semantics,
legacy DropPath-rate rounding and repository asset lookup. Both fixed-batch
and complete target-split same-checkpoint comparisons pass; identical
predictions give exactly matching Dice/ASSD. Native validation, logging,
debug/checkpoints, EMA and deterministic boundary resume pass after the fixes.
Matched nondeterministic CUDA continuation drift remains recorded separately.
See `MIGRATION_STEP8.md` and `step8_numerics_20261011/qualification.json`.
Step 1's MR -> CT accuracy-reference correction is complete: the replacement
stopped at 5,000 with its 50,000 horizon and final/best evaluations intact.
Step 9's sequential modern training queue is now running.

### 9. Retrain to the agreed stopping point, compare accuracy, and document support

- [x] Train both selected EPA/FDA directions from the original initialization
  with the same seed, effective batch size, GPU count, and data. Stop after
  5,000 updates while retaining the original 50,000-iteration LR/UDA horizon.
  Preserve a separate legacy/modern comparison per direction.
- [x] Evaluate final and best checkpoints using the frozen metric/split
  protocol. Record primary target and diagnostic source/per-class metrics,
  ASSD, runtime, and peak GPU memory.
- [x] Apply acceptance independently to CT -> MR and MR -> CT best targets.
  The user accepted both on 2026-10-13, including measured MR -> CT deviations.
  Source/final evaluations are excluded; additional paired runs are optional.
- [x] Do not retune losses, augmentation, or learning rates to conceal a
  migration discrepancy. Record any intentional algorithm change separately.
- [x] Update README environment/setup/train/test/resume instructions and
  repository guidance to match the tested stack.
- [x] Record supported configs and dependency ranges, checkpoint conversion
  limitations, and the exact reproduction commands. Keep datasets, weights,
  generated configs, and run outputs out of commits.
- [ ] Optional future scope: if a newer MMCV/Torch/Python stack is required, qualify
  it after the initial migration with the same import, numerical, smoke,
  and accuracy gates; record it as another tested stack.

**Deliverable/gate:** A migration report demonstrating comparable measured
accuracy and a reproducible modern setup. Do not mark the migration complete
while agreed training or accuracy comparison remains outstanding.

Completed and accepted on `mmcv2`, 2026-10-13; see `MIGRATION_ACCEPTANCE.md`.
The following execution notes retain original campaign history.
Both original configs resolved with
seed 0, batch 4, four workers, one GPU, no resume/weight overrides,
`stop_after=5000`, and `max_iters=50000`. The corrected legacy MR -> CT
final/best source/target evaluations are complete. The durable queue completed both directions at exactly 5,000 and all
final/best source/target evaluations. It ended in `awaiting_review` with
failed accuracy/class gates; no hyperparameters were retuned. Initial CT -> MR updates have finite losses and load the
declared ETF assets.
The versioned run-input manifest and queue control check are under
`work_dirs/mmcv2/`. Commands, provenance, automated outputs, acceptance
checks and pending review are described in `MIGRATION_STEP9.md`.
The setup/train/test/resume and tested-version/config guidance is complete
in `README.md`, `ENVIRONMENT_MMCV2.md` and `MIGRATION_SUPPORT.md`.
These documents qualify the interfaces and pinned stack; measured full-run
accuracy and its final review remain pending.

The first queue launch correctly stopped on the failed legacy evaluation:
the frozen helper required outputs under its own nested `work_dirs`.
The recovery wrapper changes only artifact-root resolution at runtime and
keeps frozen imports/configs and saved checkpoints intact. Failed queue
diagnostics are preserved; the modern queue was restarted with no training
or checkpoint overwritten. Training settings and calibrated thresholds
remain unchanged.

## Baseline commands to preserve

Historical baseline commands, run from the frozen legacy source in the
unchanged `epa` environment. The MR -> CT command below produced the
superseded regenerated-asset reference. The corrected reference uses
`tools/repair_legacy_baseline.py`, which pins the frozen declared ETF assets
and writes `mr2ct_epa_fda_seed0_declared_5k`. Its final/best evaluations are
complete. Retain the original command record for provenance; do not rerun
these commands over existing artifacts. Model,
data, optimizer, UDA, and the configured iteration/LR
horizon remain unchanged. The wrapper adds only the fixed 5,000-update stop.

```bash
cd work_dirs/migration_baseline/legacy_source

/home/akash/miniconda3/envs/epa/bin/python -u tools/train_legacy_baseline.py \
  --stop-after 5000 \
  configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir /home/akash/projects/epa-medical-uda/work_dirs/migration_baseline/ct2mr_epa_fda_seed0_5k

/home/akash/miniconda3/envs/epa/bin/python -u tools/train_legacy_baseline.py \
  --stop-after 5000 \
  configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir /home/akash/projects/epa-medical-uda/work_dirs/migration_baseline/mr2ct_epa_fda_seed0_5k
```

Each run saves its resolved config, text/JSON/TensorBoard logs, best checkpoint,
and `iter_5000.pth`/`latest.pth`, plus the stop evidence in `early_stop.json`. Queue status and exact commands
are recorded in `work_dirs/migration_baseline/selected_baselines_status.json`. Final and
best checkpoint evaluations are recorded for each direction.
To inspect live status on the training host:

```bash
/home/akash/miniconda3/envs/epa/bin/python -u \
  work_dirs/migration_baseline/run_selected_baselines.py --status
tail -n 20 work_dirs/migration_baseline/ct2mr_epa_fda_seed0_5k/console.log
```

Do not change legacy source or switch branches while the baseline queue runs.
Modern dry-run, train, test and resume commands are documented in
`MIGRATION_STEP7.md` and `MIGRATION_SUPPORT.md`. Step 8 passed; the matched
modern queue is running as described in `MIGRATION_STEP9.md`.

## Progress tracker

Make every update to this tracker, the experiment record, the accuracy
comparison, and the progress log only while on the `mmcv2` branch.

Update statuses to `Not started`, `In progress`, `Blocked`, or `Done`.
Include evidence paths and the next action; check individual tasks above as
they are completed. A blocked GPU training gate need not stop independent
API/config/evaluator work.

Current migration status: Done, accepted by the user on 2026-10-13.
Version 5 accepts both independently selected best targets, including the
measured MR -> CT differences. Source/final results are outside acceptance;
lower ASSD is better. No repeat/matched-seed training is required.
The user authorized the migration commit on `mmcv2`. Commit validation passes
for 80 Python files and whitespace; full pre-commit is unavailable. Remaining
handoff: run that suite when installed and back up ignored validation evidence.
See `MIGRATION_ACCEPTANCE.md`; merging into `main` remains a separate action.

Both modern campaigns and all final/best source/target evaluations are
complete at exactly 5,000 updates with the 50,000 horizon retained. The
current candidate disables convolution TF32 and retains matmul TF32.
Original version-1 comparisons and queue statuses remain intact; the
separate candidate `accuracy_review_v5.json` records user acceptance; earlier versions are preserved.
No further training was launched during this review. Numerical and native
training/UDA/checkpoint/CLI checks pass for the two selected one-GPU configs.
The original CT reference and corrected declared-asset MR reference are
used; the superseded MR baseline remains provenance only. Full source,
metric and reference-correction history is preserved in the reports and
progress log. All work remains on `mmcv2`; `main` and frozen artifacts are
unchanged.

| Step | Status | Evidence / result | Next action / blocker |
| --- | --- | --- | --- |
| 0. Inspect repository and research migration | Done | Findings and official links in this plan | Apply documented migration interfaces |
| 1. Freeze experiment and legacy baseline | Done | CT -> MR reference valid; corrected declared-asset MR -> CT final/best source/target evaluations complete; original MR results retained as superseded provenance | Use corrected MR reference and original CT reference in step 9 |
| 2. Freeze comparison contract | Done | `MIGRATION_CONTRACT.md`; `comparison_contract_v1.json`; both configs, datalist hashes, optimizer inventories and LR curves checked | Apply contract in config translation and step-8 checks |
| 3. Isolated environment | Done | `ENVIRONMENT_MMCV2.md`; direct pins and 78-package lock; dependency/CUDA checks passed; legacy source/packages unchanged | Apply MedPy legacy ASSD formula in step 5; full-model checks remain in step 8 |
| 4. Foundational APIs and models | Done | `MIGRATION_STEP4.md`; exact initialization (2,143 tensors) and optimizer inventories (2,134 groups) for both configs; CUDA source updates and strict checkpoint probes passed; frozen baseline source | Step 5 data transport/evaluation; full UDA update and CLI ports follow in steps 6/7 |
| 5. Data and medical evaluation | Done | `MIGRATION_STEP5.md`; exact fixed tensors/metadata; sampler, worker and CUDA checks; historical and fresh-baseline fixed-prediction metrics identical | Integrate preprocessor/evaluator with native training in step 6; accuracy parity remains pending |
| 6. Training, UDA, checkpoints | Done | `MIGRATION_STEP6.md`; exact initialization/groups/LR; real UDA updates and target threshold; exact CPU/deterministic CUDA resume; native API/evaluation passed | Step 7 original configs/CLI; step 8 legacy-precision numerical calibration |
| 7. Configs and entrypoints | Done | `MIGRATION_STEP7.md`; frozen configs/native initialization exact; direct/generated equivalence; both CLI smoke/resume/source-target evaluations; native validation/best selection; inference/representation utilities | Step 8 legacy-repeat calibration and cross-stack numerical qualification |
| 8. Numerical and smoke checks | Done | `MIGRATION_STEP8.md`; immutable legacy-calibrated limits; both default probes/full target splits pass; fixed-prediction metrics exact; native validation/EMA/checkpoints and deterministic resume pass | Step 9 full matched accuracy comparison |
| 9. Matched 5,000-step retraining and migration report | Done | Both campaigns, best-target review and documentation complete; user accepted measured results/deviations on 2026-10-13. MIGRATION_ACCEPTANCE.md and version-5 JSON record completion | Migration committed on mmcv2; full pre-commit unavailable; archive ignored evidence; no further accuracy runs required |

### FP32-convolution candidate progress

All candidate edits, runs and progress must remain on `mmcv2`.

- [x] Record the authorized precision policy and preserve original sources/results.
- [x] Disable convolution TF32 in training/evaluation; retain matmul TF32.
- [x] Pass immutable numerical budgets and native training/checkpoint checks.
- [x] Launch fresh sequential CT -> MR and MR -> CT runs with 5,000 cap and 50,000 horizon.
- [x] Complete both runs and final/best source/target evaluations.
- [x] Review best target Dice/ASSD and record version-5 user acceptance; source/final evaluations excluded.
- [x] Preserve prior repeat/matched-seed plan as optional future work; user acceptance requires no further runs.

### Experiment record

Both directions are required; preserve a separate run and metric record for each.

| Field | Value |
| --- | --- |
| Config / resolved config path | Both EPA/FDA configs named in step 1; resolved copies under each run directory and audit `configs/` |
| Overrides / direction / active UDA features | CT -> MR and MR -> CT; no hyperparameter overrides; no-mix EPA/ETF + FDA + DiceCE |
| Repository revision / environment lock paths | `4a589d1`; audit `environment.json`, `packages.txt`, `source_files.json` |
| Initialization and prototype hashes | Audit `checkpoints.json` and `prototypes.json`; MiT-B5 SHA-256 `dfbee875dac625eb1f634bab3bdefa2510d1d0afe8d5bc6512e72d280bc6cdcf` |
| Datalist hashes / evaluation split and metric definition | Audit `data_audit.json`; NPY validation slices and NPZ test volumes are separate protocols; foreground Dice |
| GPU model / driver / CUDA / GPU count | RTX 4080 Laptop, 595.99.02, Torch CUDA 11.0, cuDNN 8.0.5, one host-visible GPU |
| Seed(s) / effective batch / worker and precision settings | Seed 0 each; batch 4; workers 4; FP32 with legacy TF32 and cuDNN benchmark enabled; deterministic disabled |
| Iterations / validation interval / best selection rule | Configured/LR horizon 50,000; stop at 5,000 updates / validate every 500 / best foreground target-validation mDice, independently per direction |
| Final acceptance decision | Version 5: user accepts current best target results, including MR measured deviations; source/final excluded; higher Dice/lower ASSD better; no further accuracy runs. Earlier automatic tolerances and fixed-prediction/numerical evidence preserved |

### Accuracy comparison

Use raw 0-1 Dice values. Enter separate rows for each seed and final/best
selection. Use `N/A` with a reason for an unavailable metric rather than
leaving ambiguity about whether it was measured.

| Stack / seed | Selection / checkpoint | Target mDice | Source mDice | Myo / LAC / LVC / AA Dice | Target mASSD | Delta vs legacy | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Legacy CT -> MR EPA/FDA / 0 | Fresh final at 5,000 / `ct2mr_epa_fda_seed0_5k/iter_5000.pth` | 0.6857276685 | 0.8769931395 | 0.5201158991 / 0.7389125593 / 0.8250178512 / 0.6588643644 | 2.2916152708 | Reference | Completed; source/target evaluated; modern fixed-prediction metrics exact |
| Legacy MR -> CT EPA/FDA / 0 | Original final at 5,000 / `mr2ct_epa_fda_seed0_5k/iter_5000.pth` | 0.8022870166 | 0.8292970936 | 0.7025279241 / 0.8675543462 / 0.8334452216 / 0.8056205746 | 1.6273337966 | N/A: superseded | Asset lookup regenerated ETF tensors and advanced blur RNG; preserve for provenance only |
| Legacy MR -> CT EPA/FDA / 0 | Declared-asset replacement final / `mr2ct_epa_fda_seed0_declared_5k/iter_5000.pth` | 0.8027866677 | 0.8281325378 | 0.6963521303 / 0.8182923841 / 0.8166535104 / 0.8798486459 | 1.5840689929 | Reference | Completed at exactly 5,000; source/target evaluated; output-path recovery retained frozen source/checkpoints |
| Modern CT -> MR EPA/FDA / 0 | Final at 5,000 / `mmcv2/ct2mr_epa_fda_seed0_5k/iter_5000.pth` | 0.6251673327 | 0.8678568804 | 0.5463024590 / 0.5720703923 / 0.7816812099 / 0.6006152695 | 2.4923604681 | -0.0605603358 | Target mDice outside tolerance; per-class investigation required |
| Modern MR -> CT EPA/FDA / 0 | Final at 5,000 / `mmcv2/mr2ct_epa_fda_seed0_5k/iter_5000.pth` | 0.7928191797 | 0.7833636407 | 0.7442993212 / 0.8602143285 / 0.8442872263 / 0.7224758427 | 1.7081418482 | -0.0099674880 | Mean mDice within tolerance; per-class investigation required |
| Legacy CT -> MR EPA/FDA / 0 | Fresh best at 5,000 / `ct2mr_epa_fda_seed0_5k/best_mDice_iter_5000.pth` | 0.6857276685 | 0.8769931395 | 0.5201158991 / 0.7389125593 / 0.8250178512 / 0.6588643644 | 2.2916152708 | Reference | Completed; both evaluations match final checkpoint |
| Legacy MR -> CT EPA/FDA / 0 | Original best at 4,000 / `mr2ct_epa_fda_seed0_5k/best_mDice_iter_4000.pth` | 0.8069711912 | 0.8180215443 | 0.6988241604 / 0.8663558056 / 0.8212037075 / 0.8415010913 | 1.6160183938 | N/A: superseded | Preserved as provenance; use corrected declared-asset best below |
| Legacy MR -> CT EPA/FDA / 0 | Declared-asset replacement best at 4,000 / `mr2ct_epa_fda_seed0_declared_5k/best_mDice_iter_4000.pth` | 0.8432316865 | 0.8394168571 | 0.7573315915 / 0.8745539314 / 0.8323257785 / 0.9087154447 | 1.4677211894 | Reference | Completed; independent target-mDice selection and full source/target evaluation |
| Legacy CT -> MR EPA/FDA / 0 | Supplied best at 3,500 / `pretrained/CT2MR_best_mDice.pth` | 0.7121525915 | 0.8730193886 | 0.5791900906 / 0.7537407962 / 0.8236226480 / 0.6920568313 | 2.2119541744 | -0.0000009509 vs checkpoint metadata | Reproduced reference on the selected MR validation-slice protocol |
| Legacy MR -> CT EPA/FDA / 0 | Supplied best at 8,000 / `pretrained/MR2CT_best_mDice.pth` | 0.8522439763 | 0.8320427866 | 0.7997424407 / 0.8612157581 / 0.8675094207 / 0.8805082857 | 3.0699790653 | -0.0000011314 vs checkpoint metadata | Reproduced historical CT test-volume reference; current config uses validation slices |
| Legacy MR -> CT EPA/FDA / 0 | Supplied best at 8,000 / current CT validation slices | 0.8394315943 | 0.8320427866 | 0.7900957438 / 0.8333361024 / 0.8534989573 / 0.8807955738 | 1.4787344584 | N/A: historical metadata uses another split | Supplemental supplied-checkpoint reference; fresh results above |
| Modern CT -> MR EPA/FDA / 0 | Best at 2,500 / `mmcv2/ct2mr_epa_fda_seed0_5k/best_target_mDice_iter_2500.pth` | 0.6750950841 | 0.7765594884 | 0.6009536641 / 0.5854465173 / 0.8418297707 / 0.6721503844 | 2.2462948457 | -0.0106325844 | Target mDice outside tolerance; per-class investigation required |
| Modern MR -> CT EPA/FDA / 0 | Best at 4,000 / `mmcv2/mr2ct_epa_fda_seed0_5k/best_target_mDice_iter_4000.pth` | 0.7942362434 | 0.8298503640 | 0.7186551744 / 0.8583707114 / 0.8406549849 / 0.7592641029 | 1.6805179639 | -0.0489954432 | Target mDice outside tolerance; per-class investigation required |

### Progress log

Append one row per meaningful update. Include commands, result artifacts,
decisions, or blockers so another person can resume the work.

| Date | Step | Update / evidence | Next action |
| --- | --- | --- | --- |
| 2026-10-11 | 0 | Inspected legacy APIs, installed package metadata, configs, medical metrics, assets, and official migration docs; created plan | Identify exact baseline config/result and GPU host |
| 2026-10-11 | 0 | Created and switched to `mmcv2` from `main` at `4a589d1`; migration work is isolated to this branch | Keep `main` unchanged; commit the plan on `mmcv2` when requested |
| 2026-10-11 | 1 | Verified host GPU access and legacy CUDA forward/backward; sandbox hid GPU nodes | Use host access for GPU work |
| 2026-10-11 | 1 | Audited/hashed 43,208 data files; no integrity/pairing errors; preserved three config candidates and checkpoint metadata | Confirm experiment and historical split provenance |
| 2026-10-11 | 1 | Reproduced CT -> MR best target mDice 0.7121525915 and source mDice 0.8730193886; captured fixed inputs/logits and initial EPA updates | Finish reference captures and obtain full-run evidence |
| 2026-10-11 | 1 | Reproduced historical MR -> CT CT-volume target mDice 0.8522439763 and MR-slice source mDice 0.8320427866; verified final traces for three configs, including EMA and BN buffers | Select exact experiment; obtain full legacy run/final-checkpoint evidence |
| 2026-10-11 | 1 | User selected current EPA/FDA configs for both adaptation directions; seed 0, one GPU, no hyperparameter overrides; current validation-slice protocols frozen | Start sequential 50,000-step legacy runs and record final/best metrics |
| 2026-10-11 | 1 | Re-evaluated MR -> CT supplied checkpoint on current CT validation slices: mDice 0.8394315943; launched durable sequential training/evaluation queue, supervisor PID 71102; CT -> MR running, MR -> CT queued | Monitor `selected_baselines_status.json` and each `console.log`; complete final/best evaluations |
| 2026-10-11 | 1 | CT -> MR reached scheduled validation at iteration 500; finite training logs verified; saved trainer config matches original settings; all tracked legacy source hashes and `main` revision unchanged | Keep both baselines running; record completed final/best evaluations before marking step 1 done |
| 2026-10-11 | 1 | User changed both runs to stop at 5,000 updates while keeping the configured 50,000 horizon. Verified stop-hook control check; stopped supervisor 71102 and retained partial artifacts; launched revised queue with supervisor 76131, CT -> MR restarted from seed 0 and MR -> CT queued | Complete iteration-5,000 and best evaluations for both directions; use the same cap for modern comparisons |
| 2026-10-11 | 2 | Completed version-1 comparison contract for both directions and both stopped/best selections; verified config/list hashes, optimizer groups and LR curves; recorded actual augmentation, sampling, precision and metric semantics | Prepare isolated modern environment while baseline queue continues; retain shared legacy source until queued jobs finish |
| 2026-10-11 | 3 | Created isolated conda environment `epa-mmcv2` (Python 3.11.17); official MMCV 2.1.0/Torch 2.1/cu118 CPython 3.11 wheel exists; audited old timm/MedPy API blockers and selected explicit dependency pins | Finish installation and compiled-op/dependency checks; keep legacy environment/source unchanged |
| 2026-10-11 | 3 | Completed isolated environment qualification: Python 3.11.17, Torch 2.1.0+cu118, MMCV 2.1.0, MMEngine 0.10.7; 78-package lock; CUDA NMS/ROIAlign and Conv/BN/AdamW checks passed; saved timm probes exact; legacy ASSD formula verified; original source/packages unchanged | Step 4 source port after baseline completion or verified source isolation; preserve MedPy aggregation and investigate cuDNN workaround in full-model checks |
| 2026-10-11 | 4 | Verified 247-file frozen legacy source and handed off the supervisor without restarting CT -> MR. Completed MMEngine registries/utilities/model initialization port; both selected models exactly match 2,143 legacy state tensors and 2,134 optimizer groups; CUDA source updates and strict historical checkpoint probes passed; fixed source predictions agree | Step 5 data packing and medical evaluators; continue both legacy baselines to 5,000 updates; full UDA training and accuracy gates remain pending |
| 2026-10-11 | 5 | Completed plain tensor data transport, native preprocessor/evaluator, legacy shuffle/pinning and MedPy ASSD/empty-Dice compatibility; fixed inputs exact and all fixed-prediction metrics identical; inference APIs passed. Fresh legacy CT -> MR finished at 5,000 (target mDice 0.6857276685); MR -> CT started from frozen source | Step 6 native training/UDA/checkpoints; await MR -> CT 5,000-step final/best evaluations |
| 2026-10-11 | 1 | Both legacy runs and final/best source/target evaluations completed at the 5,000-step stop; CT -> MR final/best target mDice 0.6857276685; MR -> CT final 0.8022870166 and best at 4,000 0.8069711912; full class metrics/hashes in step-6 `legacy_baselines_complete.json` | Use separate final/best reference rows for modern accuracy comparisons |
| 2026-10-11 | 6 | Completed native BaseModel/UDA optimizer, iteration loop, scheduler, precision logging and full-state checkpoint replay on `mmcv2`; both configs pass real 4+4 updates and 1999/2000/2001 threshold; all LR values exact; CPU and deterministic CUDA continuation exact; native API and source/target evaluation smoke passed. Legacy-precision CUDA floating continuation drift is diagnostic, not a parity claim | Step 7 translate configs and native UDA construction/CLI; step 8 numerical calibration; then matched 5,000-step modern runs |
| 2026-10-11 | 7 | Completed checked legacy-to-native config translation and public CLI/utility ports on `mmcv2`; both 2-update real-data runs retain the 50,000 horizon; full-state resume and native target validation/best selection pass; CPU config-schema metrics/predictions agree; small GPU process differences retained as diagnostics | Step 8 legacy-repeat numerical calibration before full modern retraining |
| 2026-10-11 | 8 | Completed legacy-repeat calibration before modern inspection; fixed declared assets for numerical oracles; corrected FDA inverse semantics, exact legacy DropPath rates and clean-checkout root resolution; both default probes and complete target-split same-checkpoint tests pass (mDice deltas 2.52e-6/3.34e-7); identical predictions give exact metrics; native finite gradients, EMA, validation/debug/checkpoints and deterministic resume pass | Complete corrected legacy MR -> CT reference, then step 9 modern matched training; all work/progress stays on `mmcv2` |
| 2026-10-11 | 1 | Reopened MR -> CT reference after finding ETF regeneration in frozen source advanced 3,840 parent NumPy draws used by blur. Preserved original results; launched replacement from frozen source with declared assets, seed 0, horizon 50,000 and 5,000 cap; supervisor/trainer verified alive and first log confirms declared assets loaded | Monitor `mr2ct_declared_baseline_status.json`; finish final/best source/target evaluation; supersede original accuracy rows |
| 2026-10-11 | 9 | Comparison queue failed: AssertionError; see supervisor.log and comparison_status.json | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Prepared both original-config modern 5,000-update runs with unchanged 50,000 horizon; CPU queue controls and focused style checks passed; durable supervisor 157682 waits for recovered legacy evaluations | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Corrected legacy MR -> CT final/best source/target evaluations completed; modern sequential queue released | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Started modern ct2mr EPA/FDA from seed 0, original initialization; stop 5,000, horizon 50,000 | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Corrected MR -> CT reference complete: final target/source mDice 0.8027866677/0.8281325378, best at 4,000 0.8432316865/0.8394168571. Modern CT -> MR passed complete 2,400-slice validation at 500 and saved its best checkpoint; MR -> CT queued; main/frozen inputs/environment guarded | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Completed modern supported-version/config and setup/train/test/resume guide in MIGRATION_SUPPORT.md; updated README/environment/plan guidance and corrected historical MR baseline command provenance; documentation links/resolved settings/input guards pass; accuracy review still awaits both completed runs | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Documentation closeout completed on mmcv2; modern CT -> MR crossed the real iteration-2,000 target-training boundary, with finite target ETF/pseudo-label losses in the 2,050-update log; MR -> CT queued; full final/best accuracy review remains pending | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Comparison queue failed: AssertionError; see supervisor.log and comparison_status.json | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Fixed post-training installed-package inventory: 78 pins unchanged; archived original manifest/status/log; CT -> MR 5,000 checkpoint retained for evaluation; MR -> CT queued | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Corrected legacy MR -> CT final/best source/target evaluations completed; modern sequential queue released | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Recovered completed ct2mr at 5,000; evaluate saved checkpoints without retraining or resuming | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Checked stalled queue: CT -> MR full checkpoint finite at 5,000, installed 78 pins unchanged; fixed setuptools-vendor inventory false failure and restarted saved final/best evaluations, supervisor 287925; MR -> CT queued; CT validation accuracy outside tolerance and must be diagnosed | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Modern ct2mr stopped exactly at 5,000; full final/best source/target evaluations completed | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Started modern mr2ct EPA/FDA from seed 0, original initialization; stop 5,000, horizon 50,000 | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Recovered CT -> MR final/best evaluations complete: target mDice 0.6251673327/0.6750950841, source 0.8678568804/0.7765594884; target deltas exceed 0.01; MR -> CT trainer 290665 running with finite early losses; accuracy diagnosis pending, no hyperparameter changes | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Modern mr2ct stopped exactly at 5,000; full final/best source/target evaluations completed | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-11 | 9 | Both modern runs/evaluations finished; accuracy comparison written; target/class gates passed=False; source/ASSD and final documentation review pending | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-12 | 9 | Status audit: both modern directions and final/best source/target evaluations complete at exactly 5,000 (50,000 horizon retained). CT final/best target mDice 0.6251673327/0.6750950841; MR final/best 0.7928191797/0.7942362434. Complete acceptance contract fails for all selections; MR final average passes but class deltas exceed 0.02. Diagnosis pending; main and frozen inputs unchanged | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-12 | 9 | Completed discrepancy diagnosis: full 80.64M shuffle and startup RNG exact; 700 real batches/all RNG streams identical; modern cuDNN TF32 raises near-zero decoder bias gradients 40-68x, FP32 convolutions suppress them 117-212x; legacy repeat drift also substantial; modern CT final scores agree under legacy inference within 4.46e-7. MIGRATION_DIAGNOSIS.md records evidence/limits; no configs/thresholds/production precision changed; full accuracy recovery still pending | Inspect `work_dirs/mmcv2/comparison_status.json` |
| 2026-10-12 | 9 | User authorized convolution TF32 off candidate on mmcv2; archived previous source/provenance; changed training/evaluation policy, retained matmul TF32 and all other settings. MIGRATION_PRECISION_EXPERIMENT.md freezes new qualification/run plan; original refs and accuracy thresholds preserved | Qualify immutable numerical budgets, then launch fresh sequential 5,000-update runs |
| 2026-10-12 | 9 | FP32-convolution candidate passed immutable numerical qualification in both directions: optimizer/EMA/target-boundary probes and full target splits (2,400/1,200 slices), mDice deltas 5.87e-6/3.60e-8; identical-prediction metrics exact. Main, 247 frozen files, 107 configs, 6 prototype files and both package inventories unchanged | Finish native training/resume checks and launch candidate accuracy queue |
| 2026-10-12 | 9 | FP32-convolution native training/target-boundary/full-state resume checks passed; deterministic replay exact. Prepared separate campaign with unchanged seed 0, original initialization/configs, 50,000 horizon and 5,000 stop; all runtime sources and immutable budgets pinned. Existing legacy evaluation-helper repair recorded separately; original failed results preserved | Monitor candidate comparison_status.json; final/best source/target evaluations and accuracy comparison run automatically |
| 2026-10-12 | 9 | FP32-convolution candidate: Corrected legacy MR -> CT final/best source/target evaluations completed; modern sequential queue released | Inspect `work_dirs/mmcv2/fp32_conv_seed0_5k/comparison_status.json` |
| 2026-10-12 | 9 | FP32-convolution candidate: Started modern ct2mr EPA/FDA from seed 0, original initialization; stop 5,000, horizon 50,000 | Inspect `work_dirs/mmcv2/fp32_conv_seed0_5k/comparison_status.json` |
| 2026-10-12 | 9 | Fresh FP32-convolution campaign launched, supervisor 506658; CT -> MR trainer 506745 running with finite losses through update 250; MR -> CT queued. Candidate runtime/package/frozen-source guards pass; original comparison artifacts unchanged | Monitor `work_dirs/mmcv2/fp32_conv_seed0_5k/comparison_status.json`; automatic final/best evaluations and comparison follow |
| 2026-10-12 | 9 | Candidate progress check: CT -> MR active through 4050/5,000, finite losses, best scheduled target mDice 0.7307 at 3500; MR -> CT queued; no training/supervisor errors seen. Accuracy recovery remains under evaluation | Wait for both stopped/final-best source-target evaluations and unchanged acceptance review |
| 2026-10-12 | 9 | Candidate CT -> MR stopped at exactly 5,000 (UDA counter 5,000; 50,000 LR horizon retained); final scheduled target mDice 0.6957, best 0.7307 at 3500. Full final/best source/target checkpoint evaluations running; MR -> CT queued; immutable runtime/package/legacy guards pass | Await formal checkpoint metrics and automatic MR -> CT start; accuracy acceptance remains pending |
| 2026-10-12 | 9 | FP32-convolution candidate: Modern ct2mr stopped exactly at 5,000; full final/best source/target evaluations completed | Inspect `work_dirs/mmcv2/fp32_conv_seed0_5k/comparison_status.json` |
| 2026-10-12 | 9 | FP32-convolution candidate: Started modern mr2ct EPA/FDA from seed 0, original initialization; stop 5,000, horizon 50,000 | Inspect `work_dirs/mmcv2/fp32_conv_seed0_5k/comparison_status.json` |
| 2026-10-12 | 9 | Candidate CT -> MR training and all final/best source/target evaluations complete: target mDice 0.6957273925/0.7307150253, source 0.8694851428/0.8593366677. Final mean delta +0.009999724 barely fits 0.01; all four final class deltas exceed 0.02; best mean delta +0.044987357 exceeds matching limit. MR -> CT active through 400/5,000; no errors; immutable guards pass | Finish MR -> CT and formal comparison; investigate class/best-score differences before acceptance |
| 2026-10-12 | 9 | FP32-convolution candidate: Modern mr2ct stopped exactly at 5,000; full final/best source/target evaluations completed | Inspect `work_dirs/mmcv2/fp32_conv_seed0_5k/comparison_status.json` |
| 2026-10-12 | 9 | FP32-convolution candidate: Both modern runs/evaluations finished; accuracy comparison written; target/class gates passed=False; source/ASSD and final documentation review pending | Inspect `work_dirs/mmcv2/fp32_conv_seed0_5k/comparison_status.json` |
| 2026-10-12 | 9 | Candidate complete: both runs and final/best source/target evaluations stopped at 5,000 with horizon 50,000. CT target final/best 0.6957273925/0.7307150253; MR 0.8228788727/0.8310812836. All target means improved versus original modern runs, but all four complete accuracy/class gates fail; status awaiting_review. Original inputs/packages/main guards pass; no run errors | Review MIGRATION_PRECISION_RESULTS.md and define paired repeats/matched seeds to quantify remaining variability |
| 2026-10-13 | 9 | Completed class/source/ASSD review; user explicitly permits improvements. Recorded version-2 regression-only acceptance in contract and separate policy/review JSON, preserving version-1 outputs. CT best and MR final pass target gates; CT final LAC -4.068 points and MR best mean/AA -1.215/-3.965 fail. Source/ASSD diagnostic regressions remain; no code/config/checkpoint changes or additional training | Resolve remaining regressions using documented paired seed-0 repeats and conditional matched seeds 1/2; inspect class predictions and ASSD empty-mask strata |
| 2026-10-13 | 9 | User excluded source evaluations from acceptance/follow-up and confirmed lower target ASSD is better. Recorded version-3 target-only policy/review; earlier JSON/metrics preserved. CT best and MR final pass target Dice gates and improve mean ASSD; CT final LAC and MR best mean/AA remain unresolved. No training launched | Resolve target-only regressions; lower ASSD is improvement, higher ASSD is regression; no source blockers |
| 2026-10-13 | 9 | User restricted acceptance to independently selected best target checkpoints. Version-4 policy/review preserves earlier evidence and excludes all source/final results. CT best: mean Dice +4.499 points, ASSD -4.73%, Dice gates pass. MR best: mean Dice -1.215 points, AA -3.965 points, ASSD +3.34%, unresolved. No training launched | Focus only on MR best target regressions; inspect AA predictions and optionally repeat MR seed 0 on both stacks |
| 2026-10-13 | 9 | User accepted current results and declared migration successful. Recorded portable MIGRATION_ACCEPTANCE.md and separate version-5 policy/review, preserving automatic failures and measured MR best deviations. Marked step 9/migration Done; no further accuracy runs required; main unchanged | Repository handoff: full pre-commit, commit mmcv2 changes, back up ignored run evidence; merge remains separate |
| 2026-10-13 | Closeout | User authorized committing the completed migration on mmcv2. Commit includes 100 source/docs/dependency files; all 80 Python files parse and whitespace checks pass. No generated artifacts included; full pre-commit executable unavailable. Git history records commit identity | Back up ignored run evidence; full pre-commit follow-up when available; main remains unchanged |
| YYYY-MM-DD | N | Add progress here on `mmcv2` | Add next action here |
