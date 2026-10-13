# FP32-convolution candidate: best target checkpoint review

Updated 2026-10-13 on `mmcv2`. Seed 0; original 50,000-step schedule;
training capped at 5,000 updates. Same target validation splits and
corrected declared-asset MR -> CT legacy reference.

## Acceptance: version 5

The user accepted the migration on 2026-10-13, including the measured
MR -> CT best-target deviations. Only each stack's independently selected
best target checkpoint matters; source/final results are excluded.
Higher Dice and lower ASSD are better. Automatic version-4 Dice criteria and
measured deltas below remain preserved; acceptance does not claim every
numerical gate passed. Both directions are accepted and no further accuracy
runs are required. See [MIGRATION_ACCEPTANCE.md](MIGRATION_ACCEPTANCE.md).

## Best target summary

Dice deltas are percentage points. ASSD is in pixels; negative ASSD delta
means improvement.

| Direction | Legacy best iteration | Modern best iteration | Legacy mDice | Modern mDice | Dice delta | Legacy mASSD | Modern mASSD | ASSD change | Target Dice gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ct2mr | 5,000 | 3,500 | 0.68573 | 0.73072 | +4.499 | 2.29162 | 2.18319 | -4.73% (improvement) | Pass |
| mr2ct | 4,000 | 3,000 | 0.84323 | 0.83108 | -1.215 | 1.46772 | 1.51681 | +3.34% (regression) | Fail |

## Best target per-class changes

| Direction | Metric | Myo | LAC | LVC | AA |
| --- | --- | --- | --- | --- | --- |
| ct2mr | Dice points | +7.43072 | +1.27078 | +3.53777 | +5.75568 |
| ct2mr | ASSD pixels | -0.44678 | -0.19515 | -0.08731 | +0.29552 |
| mr2ct | Dice points | -1.18967 | -0.08264 | +0.37709 | -3.96495 |
| mr2ct | ASSD pixels | +0.08898 | +0.01186 | +0.04673 | +0.04878 |

## Outcome

- CT -> MR best passes target Dice criteria. Mean Dice improves 4.499
  points, every target class improves, and mean ASSD decreases 4.73%.
  AA ASSD increases 0.29552 pixel; recorded as a class diagnostic.
- MR -> CT best has user-accepted measured deviations. Mean Dice falls 1.215 points,
  exceeding the allowed one-point regression by 0.215 point. AA falls
  3.965 points, beyond the two-point class investigation tolerance.
  Mean ASSD increases 3.34%, with every class increasing. Higher ASSD
  is worse; this reinforces the target regression diagnosis.

One of two best target comparisons passes the earlier automatic Dice
criteria. Both are accepted by the user under version 5. MR -> CT mean/AA
Dice and ASSD differences remain recorded as accepted deviations; source
and final evaluations do not affect the decision. Migration is complete
for the tested single-GPU EPA/FDA scope.

Legacy slice ASSD uses a 1.0 fallback for empty/invalid masks. This
review has not stratified valid surface distances versus fallback cases;
aggregate changes do not establish their cause or full-run variability.

## Optional future validation

The user acceptance removes these earlier proposed validation tasks from
the required migration work. No additional training was launched. The prior
MR -> CT follow-up plan is retained below as optional future work:

1. Inspect best target AA predictions and validation trajectories, and
   stratify target ASSD into valid nonempty masks and fallback cases
   while preserving the primary metric definition.
2. If repeats are needed, record fresh manifests and repeat seed 0 once
   per stack for MR -> CT (two 5,000-update runs). Retain original
   initialization, declared prototypes, each stack's documented precision
   policy, 50,000 horizon and independently selected best target checkpoint.
   Neither source nor final-checkpoint evaluation is required.
3. If still unresolved, add matched seeds 1/2 to both MR -> CT stacks
   (four runs). Report best target per-seed values, paired differences,
   means and standard deviations; keep seed-0 repeats separate. Do not
   use variance alone to waive regressions or retune hyperparameters.
4. Reassess MR -> CT best mean/AA and record target ASSD findings under
   version 4; preserve the accepted CT -> MR best comparison.

## Evidence

- Original comparison: `work_dirs/mmcv2/fp32_conv_seed0_5k/accuracy_comparison.json`.
- Original SHA-256: `1ca96b43418d2d59a5cf48a665a1292faabf31ad738fad5af5df514eddb47f5f`.
- Current policy: `work_dirs/migration_baseline/comparison_contract_v5.json`.
- Current review: `work_dirs/mmcv2/fp32_conv_seed0_5k/accuracy_review_v5.json`.
- Earlier version-1/2/3/4 artifacts, source/final metrics, checkpoints and
  queue statuses remain preserved. Version 5 records the user acceptance.
