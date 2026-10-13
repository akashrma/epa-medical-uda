# MMCV 2 FP32-convolution candidate accuracy results

All runs, edits and progress belong to `mmcv2`. Seed 0; 5,000 updates; original 50,000-iteration horizon.

Accepted by the user on 2026-10-13 under version 5. Acceptance uses only
independently selected best target checkpoints, allows improvements, excludes
source/final scores and treats lower ASSD as better. The measured MR -> CT
best-target deviations are accepted; no additional accuracy runs are required.
See [MIGRATION_ACCEPTANCE.md](MIGRATION_ACCEPTANCE.md) for the decision and
[MIGRATION_METRIC_REVIEW.md](MIGRATION_METRIC_REVIEW.md) for class/ASSD details.
Tables below preserve original measurements and version-1 automatic outcomes.

| Direction | Selection | Legacy target mDice | Modern target mDice | Delta | Modern source mDice | Modern target ASSD | Accuracy/class gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ct2mr | final | 0.6857276685 | 0.6957273925 | +0.0099997240 | 0.8694851428 | 2.4695023512 | False |
| ct2mr | best | 0.6857276685 | 0.7307150253 | +0.0449873568 | 0.8593366677 | 2.1831863543 | False |
| mr2ct | final | 0.8027866677 | 0.8228788727 | +0.0200922050 | 0.8261670718 | 1.5027930160 | False |
| mr2ct | best | 0.8432316865 | 0.8310812836 | -0.0121504030 | 0.8060577745 | 1.5168075542 | False |

Convolution TF32 disabled for candidate training and evaluation; matmul TF32 and benchmark/nondeterminism retained.

| Direction | Selection | Original modern target mDice | Candidate target mDice | Candidate minus original modern |
| --- | --- | --- | --- | --- |
| ct2mr | final | 0.6251673327 | 0.6957273925 | +0.0705600598 |
| ct2mr | best | 0.6750950841 | 0.7307150253 | +0.0556199412 |
| mr2ct | final | 0.7928191797 | 0.8228788727 | +0.0300596930 |
| mr2ct | best | 0.7942362434 | 0.8310812836 | +0.0368450402 |

Full per-class/source/target Dice and ASSD, checkpoint hashes and deltas: `work_dirs/mmcv2/fp32_conv_seed0_5k/accuracy_comparison.json`.

Runtime and peak allocated/reserved GPU memory are in `runtime_summary.json` under each run directory.

The migration is accepted and complete for the tested scope. Version-5
user acceptance is saved separately as `accuracy_review_v5.json` beside the
original comparison; automatic failures, earlier reviews and original
measurements remain preserved. No hyperparameters were retuned.
