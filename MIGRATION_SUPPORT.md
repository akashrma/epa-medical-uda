# Modern stack and reproduction guide

All migration edits and progress belong to `mmcv2`. Preserve `main`, the
`epa` environment and frozen legacy source. Both modern accuracy campaigns
and planned evaluations are complete. The user accepted the migration on
2026-10-13; see [MIGRATION_ACCEPTANCE.md](MIGRATION_ACCEPTANCE.md).

## Tested stack and scope

| Component | Qualified version |
| --- | --- |
| Python | 3.11.17 |
| PyTorch / torchvision | 2.1.0+cu118 / 0.16.0+cu118 |
| MMCV / MMEngine | 2.1.0 / 0.10.7 |
| NumPy / SciPy | 1.26.4 / 1.11.4 |
| timm / Kornia | 0.4.12 / 0.5.8 |
| MedPy | 0.5.2 with the local legacy metric adapter |

Use the complete 78-package `requirements-mmcv2.lock.txt`. Installation
and CUDA verification are in [ENVIRONMENT_MMCV2.md](ENVIRONMENT_MMCV2.md).
Use compiled-op `mmcv` and keep the repository's local `mmseg` authoritative.
These are exact qualified versions, not a wider compatibility range. The
local guard accepts MMCV 2.1.0 and MMEngine 0.10.7. Newer versions or a
GitHub build require separate qualification; the official wheel sufficed.

The selected configurations are:

- `configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`
- `configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`

Both use one GPU, MiT-B5/SegFormer, five classes, no-mix EPA/ETF and FDA,
256x256 crops and FDA L=0.01. Steps 4-8 qualified initialization, data,
optimization/UDA ordering, numerical behavior, native checkpoints and CLI
smoke checks. The user accepted the 5,000-update best-target results in step 9.
MIC, dynamic prototypes, AMP, distributed UDA, other FDA geometries and
other experiment configurations are outside this qualification.

## Reproduction

Run from the repository root on `mmcv2` with `epa-mmcv2` activated. Replace
the placeholders below with one of the original configs above and actual
paths. The active queue owns its work directories; use fresh output
directories for another reproduction and run sequentially on the single GPU.

```bash
conda activate epa-mmcv2
python tools/train.py CONFIG --work-dir FRESH_WORK_DIR --stop-after 5000
python tools/test.py CONFIG CHECKPOINT --eval mDice --metrics-out METRICS.json
```

Complete direction-specific commands are in
[MIGRATION_STEP9.md](MIGRATION_STEP9.md) and
[MIGRATION_STEP7.md](MIGRATION_STEP7.md). Training uses original MiT weights,
declared ETF assets, seed 0, batch 4 and four workers. The cap preserves
`max_iters=50000`, LR/UDA horizon and validation every 500 updates. Preserve
FP32 storage, cuDNN benchmark, nondeterministic execution and every original
loss/augmentation/optimizer setting. The current authorized precision
candidate disables convolution TF32 while retaining matmul TF32, for both
training and evaluation. The original TF32-enabled comparison failed
accuracy acceptance; its source/provenance and results are preserved.
The candidate specification and isolated queue commands are in
[MIGRATION_PRECISION_EXPERIMENT.md](MIGRATION_PRECISION_EXPERIMENT.md).
Do not use resume, weight overrides,
`--debug`, `--startup-test` or `--no-validate` for the matched accuracy runs.

`--dry-run` resolves configs without building a model or starting GPU work.
Runs retain original/native configs, launch manifest, source archive, logs,
best checkpoint and final full checkpoint. In `early_stop.json`, runner and
UDA counters must equal 5,000 and the configured horizon must equal 50,000.

Evaluate each final `iter_5000.pth` and independently selected best
`best_target_mDice_iter_N.pth` with the same original config. Evaluation
loads the student. Best selection uses strict-greater target foreground
mDice over scheduled validations. Use separate metrics files per selection
and domain; `--out FILE.pkl` retains predictions.

| Direction | Target validation split | Diagnostic source split |
| --- | --- | --- |
| CT -> MR | `val_mr`: 2,400 slices | `val_ct`: 1,200 slices |
| MR -> CT | `val_ct`: 1,200 slices | `val_mr`: 2,400 slices |

For source evaluation, override the original config's `data.test.img_list`
and `data.test.ann_list` to the source validation image/GT pair. Native
configs use `test_dataloader.dataset.img_list` and `.ann_list` instead.
Use unchanged preprocessing, evaluation batch 1 and four workers. The
adapter returns raw 0-1 foreground/per-class Dice and ASSD with legacy
aggregation and empty-mask rules. Historical CT test volumes, test-time
augmentation, slide inference and FP16 change the matched protocol.

## Checkpoint compatibility

| Checkpoint | Evaluation / weight initialization | Full native resume |
| --- | --- | --- |
| Legacy checkpoint with matching student architecture | Strict load / `--load-from` | Unsupported |
| Native best checkpoint | Strict load / `--load-from` | Unsupported: optimizer/scheduler absent |
| Native full iteration checkpoint | Strict load / `--load-from` | Supported with matching runtime/data contract |

Loading handles an anchored `module.` prefix and explicitly selects
`model.` student keys from UDA checkpoints. All student keys and tensor
shapes must match. `--load-from` starts fresh optimization/UDA/sampling state.

```bash
python tools/train.py CONFIG --work-dir WORK_DIR \
  --resume-from FULL_ITER_CHECKPOINT --stop-after 5000
```

`--resume` without a path uses the work directory's last full checkpoint.
Resume restores student/teacher/prototypes, optimizer, scheduler, UDA
counters, RNG, data stream and gradient-presence state. GPU count, horizon,
precision and dataset/sampling contracts must match. CPU and deterministic
CUDA continuation passed exact checks. Under the matched nondeterministic
CUDA policy, restored state is exact but floating continuation can differ.

The original periodic checkpoint interval is 50,000; the matched run saves
its full stopped checkpoint at 5,000. Intermediate best checkpoints cannot
resume training. A separate shorter diagnostic stop can create a full
resume checkpoint with the 50,000 horizon retained.

## Accuracy review and completion

Migration was accepted by the user on 2026-10-13 for the two tested configs.
Version 5 accepts both independently selected best target checkpoints,
including the measured MR -> CT mean/AA Dice and ASSD differences. Source
and final evaluations are excluded; higher Dice and lower ASSD are better.
Earlier automatic tolerance failures stay preserved as evidence. No further
accuracy training is required. See
[MIGRATION_ACCEPTANCE.md](MIGRATION_ACCEPTANCE.md) for the decision and scope.

The original campaign comparisons/statuses and version-1/2/3/4 reviews
remain unchanged. Current user acceptance is saved separately as
`work_dirs/mmcv2/fp32_conv_seed0_5k/accuracy_review_v5.json`; portable metrics
and acceptance are in `MIGRATION_ACCEPTANCE.md` because `work_dirs/` is ignored.
Step 9 and the migration are marked complete in `PLAN.md`.

The migration commit on `mmcv2` includes code, dependencies and acceptance
documentation. Full pre-commit is unavailable; syntax checks for all 80 Python
files and whitespace checks pass. Remaining handoff is full pre-commit when
installed and backup of ignored checkpoints/logs and validation evidence. A merge into `main` requires a separate user instruction.
Keep datasets, weights and run artifacts out of commits. Broader feature or
version qualification is optional future work outside the accepted scope.
