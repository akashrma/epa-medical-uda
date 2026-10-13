# MMCV 2 migration acceptance

Accepted by the user on 2026-10-13 on `mmcv2`. The current migration is
successful for the tested scope; no further accuracy runs are required.

## Accepted scope

- Python 3.11.17, PyTorch 2.1.0+cu118, MMCV 2.1.0 and MMEngine 0.10.7;
  full dependency lock in `requirements-mmcv2.lock.txt`.
- Original CT -> MR and MR -> CT EPA/FDA MiT-B5/SegFormer configs, seed 0,
  one GPU, 5,000 updates with the original 50,000-step LR/UDA horizon.
- Each stack's independently selected best target-validation checkpoint.
  Source and final-checkpoint evaluation do not affect acceptance.
- Modern convolution TF32 disabled; matmul TF32, cuDNN benchmark and
  nondeterminism retained. Architecture, data, losses, optimizer and
  immutable numerical budgets remain unchanged.

## Best target results

| Direction | Legacy best iteration | Modern best iteration | Legacy mDice | Modern mDice | Legacy mASSD | Modern mASSD | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CT -> MR | 5,000 | 3,500 | 0.6857276685 | 0.7307150253 | 2.2916152708 | 2.1831863543 | Accepted |
| MR -> CT | 4,000 | 3,000 | 0.8432316865 | 0.8310812836 | 1.4677211894 | 1.5168075542 | Accepted with measured deviations |

Higher Dice and lower ASSD are better. CT -> MR improves mean Dice and
mean ASSD; its AA ASSD increases 0.29552 pixel. MR -> CT mean Dice decreases
1.215 points, AA Dice decreases 3.965 points, and mean ASSD increases 3.34%.
The user explicitly accepts these observed differences. Earlier automatic
Dice criteria are not claimed to have passed; their failed results remain
preserved. Acceptance is a documented user decision for this campaign.

## Validation and evidence

Steps 1-8 qualify environment/CUDA operations, exact initialization and
optimizer groups, data/evaluator parity, UDA ordering, LR curves, native
checkpoints, deterministic resume, original-config entrypoints and scoped
numerical checks. Both modern campaigns and all planned evaluations are
complete. See `MIGRATION_STEP4.md` through `MIGRATION_STEP9.md`,
`MIGRATION_DIAGNOSIS.md` and `MIGRATION_METRIC_REVIEW.md` for evidence/limits.

Current machine-readable policy is
`work_dirs/migration_baseline/comparison_contract_v5.json`; current review
is `work_dirs/mmcv2/fp32_conv_seed0_5k/accuracy_review_v5.json`.
Original comparison/status files and version-1/2/3/4 reviews remain preserved
as execution history. Their failed gates or `awaiting_review` state do not
supersede this acceptance. The portable acceptance decision and metrics are
recorded in this document because `work_dirs/` is ignored by Git.

## Repository handoff

The migration is complete. Repository handoff status:

- Migration code, dependency files, documentation and `PLAN.md` are
  included in the migration commit on `mmcv2`. Datasets, checkpoints and
  generated run artifacts are excluded; the commit identity is in Git history.
- Full pre-commit has not been run: its executable is unavailable in the
  current environment. All 80 changed/new Python files pass AST parsing,
  `git diff --check` passes, and earlier focused style checks are preserved.
  The configured full suite remains a follow-up when tooling is installed.
- Back up best checkpoints, resolved configs, manifests, logs, metrics and
  environment/numerical qualification evidence from ignored `work_dirs/`.
  This review records their locations; no external backup has been made.
- Merge into `main` only under a separate user instruction.

MIC, dynamic prototypes, AMP, distributed UDA, other configs/dependency
versions and 50,000-update accuracy remain outside this completed scope.
They are optional future work rather than migration blockers. Legacy weights
support strict evaluation/initialization, not full modern resume; native best
weights cannot resume optimization. See `MIGRATION_SUPPORT.md`.
