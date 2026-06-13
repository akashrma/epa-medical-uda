# Equiangular Prototype Alignment for Unsupervised Domain-Adaptive Medical Image Segmentation

Official Implementation for Equiangular Prototype Alignment for Unsupervised Domain-Adaptive Medical Image Segmentation accepted at IEEE International Conference on Image Processing (ICIP) 2026.

**Authors**: Akash Sharma, Arunima Sarkar, Mohanasankar Sivaprakasam


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

The copied MIC segmentation code targets the older MMSegmentation stack:

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

Example:

```bash
python -m tools.test \
  configs/medical/mr2ct_selftrain_nomix_dicece_segformer_mitb5.py \
  work_dirs/local-medical/RUN_NAME/latest.pth \
  --eval mDice
```

For configs that evaluate multiple domains, the validation results are reported with domain prefixes such as `source.mDice` and `target.mDice`.

## Notes

- `configs/medical/*segformer*` require `pretrained/mit_b5_mmseg.pth`.
- Medical classes are `background`, `Myo`, `LAC`, `LVC`, and `AA`.
- `work_dirs/` and generated configs are run artifacts and are not required to start a new experiment.
