# Step 7: original configs and native entrypoints

All edits, verification, and progress updates belong on **`mmcv2`**. The
`main` reference and frozen legacy checkout remain unchanged. This step is
qualified for the two selected medical EPA/FDA experiments on one GPU:

- `configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`
- `configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`

The original files and their inherited bases remain the experiment
specifications. `mmseg/engine/config.py::translate_legacy_config` creates a
separate native runtime config, retains the resolved specification under
`migration.legacy_config`, and does not mutate the caller's config.
Unsupported legacy runtime settings raise an error. Other model, dataset,
AMP, distributed UDA, optimizer, and augmentation combinations need their own
qualification; successful translation alone is not an accuracy claim.

## Translation

| Original setting | Native equivalent |
| --- | --- |
| `model` and `uda` | Nested DACS/student model; original UDA values and injected `max_iters` |
| `data.train` | Native nested UDA dataset; 4 source + 4 target, 4 workers |
| Legacy training sampler/transport | `LegacyRandomSampler`, `legacy_worker_init`, `uda_collate`; drop last; effective unpinned batches; no persistent workers |
| `data.val`/`data.test` | Sequential batch-one dataloaders and medical evaluators |
| Iteration runner | `EPAIterBasedTrainLoop`: 50,000 horizon; separate 5,000-update cap |
| `evaluation` | Target `mDice` every 500 updates and at the cap; strict-greater best selection |
| AdamW/custom keys | `UDAOptimWrapper` and `LegacyOptimWrapperConstructor` |
| Polynomial LR/warmup | `LegacyPolyLR`, iteration-based hook; original arithmetic and horizon |
| Logging/checkpoints | Precision JSON logging, native checkpoint hook, full EPA/RNG/loader state |
| Seed/precision | Seed 0; legacy FP32, TF32, cuDNN benchmark and nondeterministic policy |
| Model runtime settings | Work directory and original `log_config` passed to the student; UDA debug interval retained |

`UDADataset` now also accepts native nested dataset configs. The legacy
builder path remains available. Class/palette metadata is exposed to the
native runner. Native configs can be passed back to the tools, with work-dir,
stop-cap and validation overrides applied explicitly.

The implementation follows the official
[MMCV 2 compatibility notes](https://mmcv.readthedocs.io/en/latest/compatibility.html)
and [MMEngine runner migration guide](https://mmengine.readthedocs.io/en/v0.10.7/migration/runner.html).
The installed stack remains the pinned Python 3.11/MMCV 2.1.0 stack in
`ENVIRONMENT_MMCV2.md`; no newer build was needed for this step.

## Commands

Run from the repository root with `epa-mmcv2` activated, or replace `python`
with `/home/akash/miniconda3/envs/epa-mmcv2/bin/python`.

Resolve and inspect the original experiment without initializing weights,
starting workers, or accessing CUDA:

```bash
python tools/train.py \
  configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir work_dirs/mmcv2/ct2mr_epa_fda_seed0_5k --dry-run

python tools/print_config.py \
  configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --native --out work_dirs/mmcv2/ct2mr_resolved.py
```

The training launcher writes `experiment_spec.py`, `native_config.py`, and
`launch_manifest.json` with input/native config hashes, branch/revision,
seed, horizon, stop cap, initialization/resume path and GPU selection. Real
training additionally archives source while excluding data, pretrained
weights, generated configs, run outputs, caches and Git internals.

The matched training commands, **for step 9 after step 8 passes**, are:

```bash
python tools/train.py \
  configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir work_dirs/mmcv2/ct2mr_epa_fda_seed0_5k --stop-after 5000

python tools/train.py \
  configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir work_dirs/mmcv2/mr2ct_epa_fda_seed0_5k --stop-after 5000
```

The cap defaults to 5,000. Changing `--stop-after` does **not** change
`max_iters`, warmup, polynomial decay or UDA start thresholds. Do not shorten
`runner.max_iters` to perform a smoke check.

The generated-child launcher reaches the same training implementation:

```bash
python run_experiments.py \
  --config configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir work_dirs/mmcv2/mr2ct_launcher_preview --dry-run
```

`--startup-test` stops after two updates and disables validation for a short
startup diagnostic, while retaining the original horizon. Generated Python
configs preserve tuples and inheritance. `--debug` explicitly changes log,
image and validation intervals; it retains the medical metric. Neither flag
belongs in the matched accuracy runs. The predefined medical path is
qualified; the nonmedical experiment catalogue is not qualified by this step.

Full native resume uses an iteration checkpoint:

```bash
python tools/train.py work_dirs/mmcv2/ct2mr_epa_fda_seed0_5k/native_config.py \
  --resume-from work_dirs/mmcv2/ct2mr_epa_fda_seed0_5k/iter_1000.pth \
  --stop-after 5000
```

This example assumes a prior controlled stop at 1,000 in that work directory.
The original full-checkpoint interval remains 50,000, so the matched run
saves its full stopped checkpoint at 5,000; intermediate best checkpoints
serve inference rather than full resume. Use an existing full iteration
checkpoint or create a controlled stop with `--stop-after` for a resume test.

`--resume` without a path uses the work directory's last checkpoint. Resume
requires model/optimizer/scheduler, EPA/EMA state, RNG and replayable loader
state, matching horizon, precision, dataset and sampling contracts. Legacy
checkpoints and inference-only best checkpoints cannot resume native training.
`--load-from PATH` explicitly initializes weights and starts a fresh training
state. It is mutually exclusive with `--resume-from`. Resume to the already
reached cap is supported. Step 6 separately qualified continuation with
additional updates.

Checkpoint evaluation accepts original or native config schemas:

```bash
python tools/test.py \
  configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  work_dirs/mmcv2/ct2mr_epa_fda_seed0_5k/iter_5000.pth \
  --eval mDice --metrics-out work_dirs/mmcv2/ct2mr_target_metrics.json

python tools/test.py \
  configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  work_dirs/mmcv2/ct2mr_epa_fda_seed0_5k/iter_5000.pth \
  --eval mDice --metrics-out work_dirs/mmcv2/ct2mr_source_metrics.json \
  --options data.test.img_list=data/datalist/val_ct.txt \
            data.test.ann_list=data/datalist/val_ct_gt.txt
```

For MR -> CT, target is `val_ct` and source is `val_mr`. With a native config,
overrides use `test_dataloader.dataset.img_list`/`ann_list`. Target and source
lists must be declared explicitly; preserve the frozen NPY slice protocol.
`--test-set` changes list paths from `val` to `test` only when explicitly
requested and changes the evaluation protocol. `--aug-test`, slide mode and
FP16 do not belong in the matched comparison. `--device cpu` enables the CPU
control; the default is `cuda:0`. `--out` saves raw predictions. Medical
`mDice`, per-class Dice, ASSD and empty-class NaNs retain their existing
semantics; metrics JSON keeps unrounded values.

Inference and representation utilities share strict student loading:
anchored `module.` removal, then explicit `model.` selection for UDA
checkpoints, with complete student key/shape checks. Native and legacy
metadata are supported. `init_segmentor` accepts both schemas without
mutating its input; `inference_segmentor` supports NPY filenames through the
actual medical test pipeline. Ordinary image/array input retains the raw BGR
image API. NPY volume slicing is performed through the volume dataset API.
Parameter counting supports `--config`; config printing supports `--native`
and `--out`; pre-logit extraction uses native device preprocessing.

## Verification and evidence

The ignored artifact directory is
`work_dirs/migration_baseline/step7_entrypoints_20261011/`.

- `verification.json`: all inherited resolved values equal the frozen legacy
  specifications; neither source config is mutated; direct/generated native
  experiments agree after excluding run identity/path metadata. Native UDA
  construction produces the original 80,640,000 pair dataset; all 2,143
  initial tensors per direction match the frozen references exactly.
  Unsupported runtime settings and invalid stop caps are rejected.
- `cuda.json`: direct CT -> MR and generated MR -> CT each train two actual
  4+4 batches with four workers and save `iter_2.pth`. Both runner and UDA
  counters stop at two while the declared horizon remains 50,000. Explicit
  native resume restores the state at the existing cap. Separate original
  and native-schema source/target evaluations use two actual slices per
  domain; CPU medical results agree exactly, including NaN patterns.
- Separate GPU evaluations under the legacy benchmark/TF32 policy show a
  MR -> CT target `mDice` difference of approximately `5.09e-6` between
  processes. This is retained in `gpu_schema_deltas`; CPU controls establish
  schema equivalence. Step 8 still needs legacy-repeat calibration and
  cross-stack numerical checks. No accuracy threshold has been relaxed.
- `utilities.json`: same-model original/native NPY inference predictions
  agree; strict missing-weight rejection; parameter-count/config-print CLI
  checks; real pre-logit extraction yields `[1, 768, 64, 64]`.
- `validation.json`: original-config training with one update and two actual
  target validation slices exercises native metric/best-checkpoint selection
  and a full stopped checkpoint, retaining the 50,000 horizon.
- `runtime_overrides.json`: native seed overrides also update worker seeds;
  repeated launches retain the legacy experiment specification and capture
  the input config hash before overwriting an in-place runtime snapshot.
- `python_checks.json` and `manifest.json`: compile/style results and hashes;
  frozen legacy files, original configs, prototypes, environment packages and
  main reference are checked without modification. Full pre-commit is not run.

The source/target smoke metrics are diagnostic results after two updates on
two slices. They are **not** full-split accuracy or migration acceptance
results. Modern 5,000-update training has not been launched. Step 8 numerical
qualification and step 9 retraining/accuracy comparison remain outstanding.

Seed-0 GPU diagnostics using the configs listed above and two slices per
domain (`mDice` on the 0–1 scale):

| Direction | Checkpoint relative to artifact directory | Target mDice | Source mDice |
| --- | --- | --- | --- |
| CT -> MR | `smoke_ct2mr/iter_2.pth` | 0.011044848944778965 | 0.04094618324553249 |
| MR -> CT | `smoke_mr2ct/iter_2.pth` | 0.028358689157512523 | 0.011961433470082838 |

Full per-class Dice/ASSD and CPU controls are in `cuda.json`. Repeated MR -> CT
original-schema evaluation reproduced its original GPU score; an earlier
separate-process comparison had the two GPU scores in the opposite ordering.
The CPU controls agree exactly for both schemas in every domain. These
checks establish config/entrypoint equivalence and retain the numerical
variation for step 8; they do not establish cross-stack accuracy parity.

Reproduce the checks:

```bash
python -u tools/verify_mmcv2_entrypoints.py
python -u tools/verify_mmcv2_entrypoints.py --cuda-only
python -u tools/verify_mmcv2_entrypoints.py --utilities-only
python -u tools/verify_mmcv2_entrypoints.py --validation-only
```

CUDA checks require the GPU host and writable multiprocessing IPC. Utility
and validation checks use the checkpoints/absolute two-slice lists produced
by the CUDA check. `--reuse-smoke` can reuse those short training checkpoints
when checking evaluation; it is not a fresh-training parity check.
