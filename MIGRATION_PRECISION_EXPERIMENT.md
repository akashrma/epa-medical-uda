# FP32-convolution accuracy experiment

Current status: migration accepted by the user on 2026-10-13; see
[MIGRATION_ACCEPTANCE.md](MIGRATION_ACCEPTANCE.md). Best target checkpoints
only; source/final results excluded; measured MR -> CT differences accepted.
No further accuracy runs are required. Execution/status notes below are
historical campaign evidence and retain their original automatic outcomes.


Authorized on 2026-10-12. All edits, runs and progress must remain on
`mmcv2`; `main`, frozen legacy source/environments, original experiment
configs, numerical budgets and failed comparison artifacts remain intact.

Completed-run review on 2026-10-13 uses the user-authorized version-2
acceptance policy: improvements are allowed; target mean regressions remain
limited to 0.01 and class regressions greater than 0.02 require investigation.
The later version-3 instruction excludes source evaluations from acceptance
and required follow-up, and confirms lower target ASSD is better.
Version 4 further restricts acceptance and required follow-up to each stack's
independently selected best target checkpoint; final evaluations are excluded.
The version-1 execution specification below is preserved as provenance.
See [MIGRATION_METRIC_REVIEW.md](MIGRATION_METRIC_REVIEW.md) for the current
target class/ASSD review and unresolved regressions.

## Candidate and scope

The candidate disables `torch.backends.cudnn.allow_tf32` in the modern
training hook and public evaluator. Tensor storage remains FP32. Matmul
TF32, cuDNN benchmarking and the original nondeterministic policy remain
enabled. This is an explicit adjustment to the version-1 runtime precision
policy in `MIGRATION_CONTRACT.md`; the original contract and its accuracy
thresholds are preserved. Evaluation uses the candidate convolution policy
consistently with scheduled validation and best-checkpoint selection.

Diagnostic controls showed reduced near-zero decoder bias gradients with
this setting. They did not establish recovery of 5,000-update accuracy.
Legacy repeats also diverge; do not declare success from short checks or
dismiss an accuracy failure without complete-run evidence.

The unchanged experiment specifications are:

- CT -> MR: `configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`.
- MR -> CT: `configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`.
- Seed 0, one GPU, original MiT initialization and declared ETF assets.
- Stop at exactly 5,000 optimizer updates, retain the 50,000 horizon,
  original warmup/LR schedule and validation every 500 updates.
- Keep optimizer epsilon, losses, architecture, data and all other
  hyperparameters unchanged. Start fresh; do not resume earlier TF32 runs.

## Execution and acceptance

1. Preserve pre-change sources/provenance under
   `work_dirs/mmcv2/fp32_conv_transition/`.
2. Reuse immutable step-8 legacy numerical budgets and declared-asset
   oracles. Capture fresh modern fixed-input initial/target-boundary
   optimizer/EMA probes, full target-split inference and identical-prediction
   medical metrics with convolution TF32 disabled. Verify public runtime
   hook flags, frozen inputs and environments.
3. Prepare and launch a durable sequential training/evaluation queue in
   `work_dirs/mmcv2/fp32_conv_seed0_5k/`. Hash the candidate inputs and
   qualification, preserve the original manifests and results, and record
   exact commands. Train CT -> MR, evaluate, then train/evaluate MR -> CT.
4. Evaluate both final and independently selected best checkpoints on
   complete source/target validation splits. Compare to the valid original
   legacy references; write `MIGRATION_PRECISION_RESULTS.md` and the
   candidate's `accuracy_comparison.json`.
5. Apply the existing absolute target mDice difference <= 0.01 for each
   direction and final/best selection. Investigate per-class Dice differences
   > 0.02; review source Dice and ASSD. Success requires complete evidence
   and review. If discrepancies remain, document a paired repeat/additional
   matched-seed plan before expanding the training experiment.

The original legacy targets are CT -> MR final/best 0.6857276685, and
corrected MR -> CT final 0.8027866677 / best at 4,000 0.8432316865.
The original failed modern results remain in `MIGRATION_RESULTS.md` and
`work_dirs/mmcv2/accuracy_comparison.json`.

## Commands

Run from the repository root. GPU execution requires host access.
Preparation and launch are each performed once; inspect status instead of
launching another copy. The supervisor closes out both evaluations and
updates `PLAN.md` even after the initiating terminal exits.

```bash
/home/akash/miniconda3/envs/epa-mmcv2/bin/python -u \
  tools/verify_mmcv2_numerics.py --phase precision-candidate \
  --out-dir work_dirs/migration_baseline/step9_fp32_conv_qualification_20261012
/home/akash/miniconda3/envs/epa-mmcv2/bin/python -u \
  tools/run_mmcv2_comparison.py --campaign fp32-conv --prepare
/home/akash/miniconda3/envs/epa-mmcv2/bin/python -u \
  tools/run_mmcv2_comparison.py --campaign fp32-conv --launch
/home/akash/miniconda3/envs/epa-mmcv2/bin/python -u \
  tools/run_mmcv2_comparison.py --campaign fp32-conv --status
```

Candidate numerical evidence is under
`work_dirs/migration_baseline/step9_fp32_conv_qualification_20261012/`.
Candidate run state is
`work_dirs/mmcv2/fp32_conv_seed0_5k/comparison_status.json`.
The progress log in `PLAN.md` records execution status; this specification
is frozen with the training manifest once the queue is prepared.
