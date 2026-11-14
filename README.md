# Equiangular Prototype Alignment for Unsupervised Domain-Adaptive Medical Image Segmentation

Official Implementation for Equiangular Prototype Alignment for Unsupervised Domain-Adaptive Medical Image Segmentation accepted at IEEE International Conference on Image Processing (ICIP) 2026.

**Authors**: Akash Sharma, Arunima Sarkar, Mohanasankar Sivaprakasam

The current runtime uses Python 3.11, PyTorch 2.1, MMCV 2.1 and MMEngine.
The two selected single-GPU EPA/FDA configurations have completed
5,000-update training comparisons in both directions.

- [Environment and installation](ENVIRONMENT.md)
- [Training, evaluation and resume](TRAINING.md)
- [Recorded results and tested scope](TRAINING_RESULTS.md)

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
`-- requirements-mmcv2.lock.txt # pinned modern dependencies
```

## Dataset and Pretrained Models

See this link: https://drive.google.com/drive/folders/1BcGtDlH80o15i-A_VTL-P6UPHNJnaI5D?usp=sharing for pretrained models and dataset.

The medical datalists use paths relative to `data/datalist/` into
`data/data_np/`, so the project can run without the original external dataset
location. The main experiment configs are in `configs/medical/`.

## Environment

Install the pinned modern stack using [ENVIRONMENT.md](ENVIRONMENT.md),
then activate it before training or evaluation:

```bash
conda activate epa-mmcv2
```

`requirements-mmcv2.txt` records direct dependencies and
`requirements-mmcv2.lock.txt` provides the full Python package lock.
`requirements.txt` is retained as a historical legacy-stack record.

## Data

Supply the medical data under `data/data_np/`:

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

Download this file separately; weights and datasets are excluded from Git.

DeepLabV2 configs use:

```text
open-mmlab://resnet50_v1c
```

MMEngine must be able to resolve or download that checkpoint, unless it is already cached in your environment. These historical DeepLabV2 configs are outside the tested modern scope.

## Medical Configs

The top-level medical configs cover MR -> CT and CT -> MR UDA runs:

```bash
find configs/medical -maxdepth 1 -name "*.py" | sort
```

Historical examples (only the two EPA/FDA configs in [TRAINING.md](TRAINING.md) are qualified on the modern stack):

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

## Training and evaluation

Run from the repository root using a fresh output directory:

```bash
python tools/train.py configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py --work-dir work_dirs/ct2mr_epa_fda --seed 0 --stop-after 5000
```

For MR -> CT, use
`configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`
and a separate work directory. The default cap is 5,000 optimizer updates;
the configured 50,000-step LR/UDA horizon is retained. Add `--dry-run` to
inspect resolved configs without building a model or starting GPU work.

Evaluate with the same config and an actual checkpoint path:

```bash
python tools/test.py CONFIG CHECKPOINT --eval mDice --metrics-out METRICS.json
```

Resume requires a full native iteration checkpoint:

```bash
python tools/train.py CONFIG --work-dir WORK_DIR --resume-from FULL_ITER_CHECKPOINT --stop-after 5000
```

Best weights and legacy checkpoints support strict evaluation or
`--load-from` initialization, but cannot resume native optimization.
`run_experiments.py --config CONFIG` remains available as a launcher that
writes generated configs under `configs/generated/` and runs under
`work_dirs/`. See [TRAINING.md](TRAINING.md) for the precision policy,
validation protocol, checkpoint selection and supported options.

## Notes

- Medical classes are `background`, `Myo`, `LAC`, `LVC`, and `AA`.
- Data, pretrained/ETF weights, generated configs and run artifacts are excluded from Git.
- Back up `work_dirs/` separately to retain checkpoints, logs and numerical evidence.
- The accepted migration and historical reports remain on `mmcv2` at `a05cff5`.
