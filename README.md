# Equiangular Prototype Alignment for Unsupervised Domain-Adaptive Medical Image Segmentation

Official Implementation for Equiangular Prototype Alignment for Unsupervised Domain-Adaptive Medical Image Segmentation accepted at IEEE International Conference on Image Processing (ICIP) 2026.

**Authors**: Akash Sharma, Arunima Sarkar, Mohanasankar Sivaprakasam


## MMCV 2 migration branch

All migration edits and progress are confined to **`mmcv2`**; the legacy
stack below remains the reference for the frozen baseline. The isolated
Python 3.11/MMCV 2.1.0 setup is documented in
[ENVIRONMENT_MMCV2.md](ENVIRONMENT_MMCV2.md). Original medical configs now
translate to native MMEngine training and evaluation through the existing
launchers; commands, supported scope and resume rules are in
[MIGRATION_STEP7.md](MIGRATION_STEP7.md). The consolidated version,
reproduction, evaluation and checkpoint guide is
[MIGRATION_SUPPORT.md](MIGRATION_SUPPORT.md).

Both EPA/FDA directions pass short real-data CLI checks. The matched runs
stopped at 5,000 updates while retaining the 50,000-iteration schedule.
Cross-stack numerical checks pass for these two configurations; results and
scope are in [MIGRATION_STEP8.md](MIGRATION_STEP8.md). The replacement MR -> CT
legacy reference corrected an asset-path/RNG issue in the isolated checkout;
its 5,000-update final/best evaluations are complete.
Both modern campaigns and all planned evaluations are complete. The user
accepted the migration on 2026-10-13 for the tested single-GPU EPA/FDA scope.
Acceptance uses independently selected best target checkpoints, permits
improvements, excludes source/final evaluations, and treats lower ASSD as
better. The measured MR -> CT best-target differences are accepted deviations.
Decision, results and scope are in
[MIGRATION_ACCEPTANCE.md](MIGRATION_ACCEPTANCE.md); class/ASSD details are in
[MIGRATION_METRIC_REVIEW.md](MIGRATION_METRIC_REVIEW.md).
Precision diagnosis and the authorized FP32-convolution candidate are in
[MIGRATION_DIAGNOSIS.md](MIGRATION_DIAGNOSIS.md) and
[MIGRATION_PRECISION_EXPERIMENT.md](MIGRATION_PRECISION_EXPERIMENT.md).
Original campaign execution is preserved in
[MIGRATION_STEP9.md](MIGRATION_STEP9.md). No further accuracy runs are required.

## Repository Layout

```text
.
|-- configs/
|   |-- _base_/                 # model, dataset, UDA, optimizer, schedule bases
|   `-- medical/                # MR <-> CT experiment configs
|-- data/
|   |-- data_np/                # copied medical .npy slices and .npz test volumes
|   `-- datalist/               # train/val/test file lists
|-- mmseg/                      # local MMSeg/MIC implementation
|-- pretrained/                 # root copy of pretrained weights
|-- tools/                      # train/test utilities
|-- run_experiments.py          # config wrapper and launcher
`-- requirements.txt
```

## Dataset and Pretrained Models

See this link: https://drive.google.com/drive/folders/1BcGtDlH80o15i-A_VTL-P6UPHNJnaI5D?usp=sharing for pretrained models and dataset.

The medical datalists use paths relative to `data/datalist/` into
`data/data_np/`, so the project can run without the original external dataset
location. The main experiment configs are in `configs/medical/`.

## Environment

On `mmcv2`, use the isolated `epa-mmcv2` environment: Python 3.11.17,
PyTorch 2.1.0+cu118, torchvision 0.16.0+cu118, MMCV 2.1.0 and MMEngine
0.10.7. Reproducible installation and CUDA qualification are in
[ENVIRONMENT_MMCV2.md](ENVIRONMENT_MMCV2.md); dependencies are pinned in
`requirements-mmcv2.txt` and `requirements-mmcv2.lock.txt`.

```bash
conda activate epa-mmcv2
```

The selected single-GPU CT -> MR and MR -> CT EPA/FDA configurations have
passed numerical and smoke qualification. Matched accuracy runs are complete;
the user accepted their best-target results. Other configurations/features require separate
qualification; the config catalogue below also includes historical experiments.

### Frozen legacy reference

The preserved `epa` environment and isolated legacy source use:

- Python 3.8.x
- CUDA 11.0 compatible PyTorch
- `torch==1.7.1+cu110`
- `torchvision==0.8.2+cu110`
- `mmcv-full==1.3.7`

Install the Python requirements from the repo root:

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7
```

For medical metric evaluation, `medpy` is also required:

```bash
pip install medpy
```

## Data

The medical data is already copied into `data/data_np/`:

- `train_mr/`, `gt_train_mr/`
- `train_ct/`, `gt_train_ct/`
- `val_mr/`, `gt_val_mr/`
- `val_ct/`, `gt_val_ct/`
- `test_mr/`
- `test_ct/`

The list files are in `data/datalist/`:

- `train_mr.txt`, `train_mr_gt.txt`
- `train_ct.txt`, `train_ct_gt.txt`
- `val_mr.txt`, `val_mr_gt.txt`
- `val_ct.txt`, `val_ct_gt.txt`
- `test_mr.txt`
- `test_ct.txt`

Training and slice-level validation use `.npy` image/label lists. The nested source-only configs under `configs/medical/mr2ct/` and `configs/medical/ct2mr/` also define volume-level evaluation on `.npz` test volumes.

## Pretrained Weights

SegFormer configs load:

```text
pretrained/mit_b5_mmseg.pth
```

That file is included.

DeepLabV2 configs use:

```text
open-mmlab://resnet50_v1c
```

MMCV must be able to resolve or download that checkpoint, unless it is already cached in your environment.

## Medical Configs

The top-level medical configs cover MR -> CT and CT -> MR UDA runs:

```bash
find configs/medical -maxdepth 1 -name "*.py" | sort
```

Common examples:

```text
configs/medical/mr2ct_selftrain_nomix_dicece_segformer_mitb5.py
configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py
configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda_mic.py
configs/medical/ct2mr_selftrain_nomix_segformer_mitb5.py
configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5.py
configs/medical/mr2ct_dacs_deeplabv2_r50.py
configs/medical/ct2mr_dacs_deeplabv2_r50.py
```

Source-only configs are also available in:

```text
configs/medical/mr2ct/
configs/medical/ct2mr/
```

## Training

Run commands from the repository root.

For the modern migration comparison, the sequential queue and exact
commands are documented in [MIGRATION_STEP9.md](MIGRATION_STEP9.md).
The original configs are translated to native MMEngine configs automatically.
`--stop-after 5000` caps optimizer updates and retains the 50,000-step
schedule. These launcher examples are part of the historical config catalogue.

Example MR -> CT SegFormer run:

```bash
python run_experiments.py --config configs/medical/mr2ct_selftrain_nomix_dicece_segformer_mitb5.py
```

Example CT -> MR SegFormer run:

```bash
python run_experiments.py --config configs/medical/ct2mr_selftrain_nomix_segformer_mitb5.py
```

Example MR -> CT MIC/FDA run:

```bash
python run_experiments.py --config configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda_mic.py
```

`run_experiments.py` writes generated child configs under `configs/generated/` and stores logs/checkpoints under `work_dirs/`.

## Direct Training Entrypoint

You can also bypass `run_experiments.py`:

```bash
python tools/train.py configs/medical/mr2ct_selftrain_nomix_dicece_segformer_mitb5.py
```

Use `--work-dir` to choose an output directory:

```bash
python tools/train.py configs/medical/mr2ct_selftrain_nomix_dicece_segformer_mitb5.py \
  --work-dir work_dirs/manual_mr2ct
```

## Evaluation

Evaluate a checkpoint with MMSeg's test entrypoint:

```bash
python -m tools.test CONFIG CHECKPOINT --eval mDice
```

In `epa-mmcv2`, add `--metrics-out PATH.json` to save unrounded medical
Dice/ASSD results. Full native checkpoints and legacy student weights load
strictly for evaluation. Source and target slice/volume protocols must match
the intended configuration; the migration compares validation slices.

Example:

```bash
python -m tools.test \
  configs/medical/mr2ct_selftrain_nomix_dicece_segformer_mitb5.py \
  work_dirs/local-medical/RUN_NAME/latest.pth \
  --eval mDice
```

For configs that evaluate multiple domains, the validation results are reported with domain prefixes such as `source.mDice` and `target.mDice`.

## Native training resume

Use a full native iteration checkpoint with the same experiment settings:

```bash
python tools/train.py CONFIG --work-dir WORK_DIR --resume-from ITER_CHECKPOINT
```

Native checkpoints restore optimizer, scheduler, teacher/UDA counters,
RNG and data-stream state. Under the matched nondeterministic CUDA policy,
floating continuation can differ. Legacy and best weights checkpoints are
for evaluation or `--load-from` initialization; they cannot resume native
training. The fresh accuracy runs use neither resume nor load overrides.
See [MIGRATION_STEP7.md](MIGRATION_STEP7.md) for supported CLI details.

## Notes

- `configs/medical/*segformer*` require `pretrained/mit_b5_mmseg.pth`.
- Medical classes are `background`, `Myo`, `LAC`, `LVC`, and `AA`.
- `work_dirs/` and generated configs are run artifacts and are not required to start a new experiment.
