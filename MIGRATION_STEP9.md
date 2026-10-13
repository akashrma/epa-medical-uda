# Step 9: matched accuracy runs

Current status: migration accepted by the user on 2026-10-13; see
[MIGRATION_ACCEPTANCE.md](MIGRATION_ACCEPTANCE.md). Best target checkpoints
only; source/final results excluded; measured MR -> CT differences accepted.
No further accuracy runs are required. Execution/status notes below are
historical campaign evidence and retain their original automatic outcomes.


Started 2026-10-11 on `mmcv2`. All edits, runs and progress updates must stay
on this branch. `main` remains at `4a589d1`. Accuracy acceptance is pending.

## Experiment and execution order

Use the original CT -> MR and MR -> CT EPA/FDA configs, seed 0, MiT-B5
initialization, declared ETF tensors, batch 4 and four workers on one GPU.
Train in FP32 with the qualified legacy TF32/cuDNN benchmark policy.
Stop exactly after 5,000 optimizer updates; retain the 50,000-step LR/UDA
horizon, warmup, losses, EMA, augmentations and validation every 500 updates.
Do not resume legacy checkpoints, shorten the LR horizon or retune settings.

The durable supervisor waits until the corrected legacy MR -> CT reference
has finished both final/best source/target evaluations. It then trains
CT -> MR, evaluates its final/best checkpoints, trains MR -> CT and evaluates
its final/best checkpoints. Only one GPU job in this queue runs at a time.

The corrected legacy run finished training at 5,000. Its first evaluation
attempt rejected the sibling output directory because the frozen helper's
root points at the isolated source tree. The recovery wrapper keeps all
frozen imports/configs and checkpoints and changes only output-root
resolution. The failed queue status/log are preserved. No modern training
or checkpoints had been created when that first queue stopped.

Recovery is complete. The corrected MR -> CT final checkpoint has target
mDice 0.8027866677 and source mDice 0.8281325378. Its independently selected
best checkpoint at 4,000 has target mDice 0.8432316865 and source mDice
0.8394168571. CT -> MR modern training has started with finite initial
losses; MR -> CT remains queued. Live status is authoritative for later
iteration counts and results.

CT -> MR subsequently completed exactly 5,000 updates and saved the full
final checkpoint and best checkpoint at 2,500. Its post-training package
check stopped the queue because setuptools adds bundled dependency metadata
to `sys.path`; the installed 78-package environment remained unchanged.
The inventory now scans the interpreter's installation directories.
Recovery archives the original supervisor, manifest, status and logs,
records the automation revision and checkpoint hashes, and evaluates the
saved CT checkpoints without repeating any updates. All model/config/data
and environment guards remain enforced. MR -> CT follows those evaluations.

CT -> MR's training-validation final/best target mDice was
0.6251682374 / 0.6750950841, against legacy final/best 0.6857276685.
Both absolute deltas exceed 0.01; the final difference is substantial.
Full checkpoint/source/target evaluation and discrepancy diagnosis remain
required. These observations do not establish migration acceptance.

## Commands and provenance

Run from the repository root with the qualified `epa-mmcv2` environment.
Preparation and launch have already been performed; do not launch duplicates.

```bash
/home/akash/miniconda3/envs/epa-mmcv2/bin/python -u \
  tools/run_mmcv2_comparison.py --prepare
/home/akash/miniconda3/envs/epa-mmcv2/bin/python -u \
  tools/run_mmcv2_comparison.py --launch
/home/akash/miniconda3/envs/epa-mmcv2/bin/python -u \
  tools/run_mmcv2_comparison.py --status
```

GPU training requires host access; a sandbox-only process may hide CUDA and
host process IDs. Read status/log files directly to inspect progress without
GPU access. The supervisor runs independently of the initiating terminal.

The queue invokes the public trainer with these unchanged specifications:

```bash
/home/akash/miniconda3/envs/epa-mmcv2/bin/python -u tools/train.py \
  configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir work_dirs/mmcv2/ct2mr_epa_fda_seed0_5k --stop-after 5000
/home/akash/miniconda3/envs/epa-mmcv2/bin/python -u tools/train.py \
  configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir work_dirs/mmcv2/mr2ct_epa_fda_seed0_5k --stop-after 5000
```

`work_dirs/mmcv2/comparison_manifest.json` records the runtime source,
config, datalist, initialization, prototype, qualification and contract
hashes, environment packages, exact commands and unchanged thresholds.
Each stage verifies those hashes, the frozen legacy source, `mmcv2` and
the unchanged `main` revision. Keep runtime code/config/assets and the
environment stable until the queue finishes; documentation/progress can
continue on `mmcv2`.

Each run retains original/native resolved configs, launch manifest, code
archive, complete logs, TensorBoard/debug outputs, independently selected
best checkpoint, full `iter_5000.pth` and exact stop/counter evidence.
`runtime_summary.json` records elapsed time including initialization,
training, scheduled validation and saves, plus peak allocated/reserved
GPU memory, device and actual precision/backend settings.

MMEngine logging resets allocator peak counters. The saved byte counters
cover the interval since its last reset; the full-run allocated peak is
also recorded from logged interval maxima in MiB, truncated by the logger.
CT -> MR's full-run logged allocated peak was 4,365 MiB. Preserve this
measurement scope when reporting memory; the final byte counter alone is
not the full-training peak.

## Evaluation and acceptance

Evaluate the student on the complete original validation-slice splits:
2,400 MR target slices for CT -> MR and 1,200 CT target slices for MR -> CT.
Use the opposite source validation split for diagnostics. Preserve inference
preprocessing, four workers, evaluation batch 1, foreground class averaging,
and the qualified legacy Dice/ASSD definitions.

For each final and best checkpoint, the supervisor invokes `tools/test.py`
and saves full predictions, unrounded mDice/mASSD and every class Dice/ASSD,
logs, summary and checkpoint SHA-256. Source evaluation changes only
`data.test.img_list` and `data.test.ann_list` to the source validation lists.
Source evaluation runs after training and does not affect training RNG.

Compare CT -> MR against the valid original CT reference and MR -> CT
against the declared-asset replacement. The original regenerated-asset
MR reference is preserved as provenance and excluded from accuracy gates.

Apply the frozen contract independently to both directions and both
selections: absolute target mDice delta <= 0.01; any class Dice delta > 0.02
requires investigation and prevents automatic acceptance. Retain source
and ASSD deltas for review without inventing new thresholds.

After both runs, automated outputs are:

- `work_dirs/mmcv2/accuracy_comparison.json`: full metrics, signed deltas,
  checkpoint identities and separate accuracy/class verdicts.
- `MIGRATION_RESULTS.md`: compact comparison table and artifact locations,
  written on `mmcv2`.
- `work_dirs/mmcv2/comparison_status.json`: live stage, PID, last logged
  training update, commands, completed evaluations or actionable failure.
- `PLAN.md`: progress log entries at each stage transition.

The supervisor ends in `awaiting_review` after producing results. It does
not mark the migration complete. Review source/ASSD diagnostics, investigate
any failed gate, finish reproduction/support documentation and only then
update the plan's accuracy rows and completion status. Additional versions,
MIC/dynamic prototypes, AMP, distributed UDA and other FDA geometries
remain outside the qualified configuration scope.

## Preparation checks

Both public CLI dry runs resolved seed 0, one GPU, cap 5,000/horizon 50,000
and fresh initialization with no resume or load override. CPU controls
verified that a failing direction/selection is not hidden by the other
three, positive/negative mDice deviations use the absolute threshold,
per-class deviations block acceptance, source commands choose the proper
validation split, and diagnostic review remains required. These controls
use synthetic metrics and are not accuracy results; evidence is
`work_dirs/mmcv2/queue_control_check.json`.

Focused YAPF/import-order/Flake8 checks passed for the supervisor and the
legacy evaluation recovery wrapper. Full repository pre-commit remains
unavailable in the current environments, as documented in earlier steps.
