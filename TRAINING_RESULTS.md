# Training results

The Python 3.11/MMCV 2 migration was accepted on 2026-10-13 for the scope
below. These are independently selected best target-validation checkpoints;
source and final-checkpoint evaluations do not determine acceptance.

## Protocol

- Original `configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`
  and `configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`.
- Seed 0, one GPU, 5,000 updates with the original 50,000-step LR/UDA horizon;
  validation every 500 updates, batch size 4, four workers, 256x256 crops.
- MiT-B5/SegFormer, five classes, EPA/ETF and FDA L=0.01; original pretrained
  weights, architecture, data, losses and optimizer settings.
- Modern stack: Python 3.11.17, PyTorch 2.1.0+cu118, MMCV 2.1.0,
  MMEngine 0.10.7; full pins in `requirements-mmcv2.lock.txt`.
- FP32 storage, modern convolution TF32 disabled, matmul TF32 enabled,
  cuDNN benchmark and nondeterministic execution retained.
- CT -> MR target: `val_mr`, 2,400 slices. MR -> CT target: `val_ct`,
  1,200 slices. Foreground classes: Myo, LAC, LVC and AA.

## Best target results

| Direction | Legacy best iteration | Modern best iteration | Legacy mDice | Modern mDice | Legacy mASSD | Modern mASSD |
| --- | --- | --- | --- | --- | --- | --- |
| CT -> MR | 5,000 | 3,500 | 0.6857276685 | 0.7307150253 | 2.2916152708 | 2.1831863543 |
| MR -> CT | 4,000 | 3,000 | 0.8432316865 | 0.8310812836 | 1.4677211894 | 1.5168075542 |

Higher Dice and lower ASSD are better. CT -> MR improves mean Dice by
4.499 percentage points and mean ASSD by 4.73%; its AA ASSD increases
0.29552 pixel. MR -> CT mean Dice decreases 1.215 points, AA Dice decreases
3.965 points, and mean ASSD increases 3.34%. These observed deviations were
explicitly accepted. Earlier automatic Dice gates failed; this table records
the accepted campaign outcome and does not claim those gates passed.

## Evidence and limits

Environment/CUDA operations, initialization, optimizer groups, data and
evaluator parity, UDA ordering, LR curves, native checkpoint/resume behavior,
entrypoints and scoped numerical checks were qualified during migration.
The complete historical code and reports remain in Git at `a05cff5` (also
preserved on the local development branch `mmcv2`). This commit is an
ancestor of the published modern code.

Local ignored evidence is retained under:

- `work_dirs/migration_baseline/`: legacy runs and qualification fixtures;
  `comparison_contract_v5.json` records the accepted comparison policy.
- `work_dirs/mmcv2/fp32_conv_seed0_5k/`: modern runs, best/full checkpoints,
  logs and evaluations; `accuracy_review_v5.json` records acceptance.

These results predate the cleanup's use of ordinary FP32 `torch.linspace`
for DropPath rates and removal of obsolete FFT fallback branches. The
cleanup checks configuration equivalence, DropPath probabilities, exact CPU
FDA outputs/gradients and model/checkpoint loading; it does not repeat the
5,000-update accuracy campaign. Tiny DropPath rounding differences can
change a training trajectory. Do not attribute new accuracy measurements
to the cleaned code without running that experiment.

MIC, dynamic prototypes, AMP, distributed UDA, other dependency versions or
configs and 50,000-update accuracy remain outside the tested scope. See
[TRAINING.md](TRAINING.md) for commands and checkpoint limitations and
[ENVIRONMENT.md](ENVIRONMENT.md) for setup.
