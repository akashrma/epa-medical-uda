# Equiangular Prototype Alignment for Unsupervised Domain-Adaptive Medical Image Segmentation

Official implementation of **Equiangular Prototype Alignment for Unsupervised
Domain-Adaptive Medical Image Segmentation**, accepted at ICIP 2026.

**Authors:** Akash Sharma, Arunima Sarkar, Mohanasankar Sivaprakasam

Train EPA for CT → MR and MR → CT segmentation using the local SegFormer/MiT-B5
implementation, fixed ETF prototypes, FDA and self-training. The current code
runs on **Python 3.11, PyTorch 2.1, MMCV 2.1 and MMEngine**. Existing medical
experiment configs are translated to MMEngine by the training entrypoint.

This README covers installation, the exact data layout, reproducing the
5,000-update comparison, evaluation, new training runs and resume.
Upstream attribution and licensing are in [LICENSES.md](LICENSES.md).

## 1. Install the environment

The tested platform is Linux with one NVIDIA GPU (RTX 4080 Laptop GPU,
about 12 GiB VRAM), Python 3.11.17, PyTorch 2.1.0+cu118, torchvision
0.16.0+cu118, MMCV 2.1.0 and MMEngine 0.10.7. A compatible NVIDIA driver is
required. Other platforms, GPUs and dependency versions are not qualified.

| Component | Qualified version |
| --- | --- |
| Python | 3.11.17 |
| PyTorch / torchvision | 2.1.0+cu118 / 0.16.0+cu118 |
| CUDA runtime / cuDNN | 11.8 / 8.7.0 |
| MMCV / MMEngine | 2.1.0 / 0.10.7 |
| NumPy / SciPy | 1.26.4 / 1.11.4 |
| timm / Kornia | 0.4.12 / 0.5.8 |
| MedPy / SimpleITK | 0.5.2 / 2.3.1 |
| OpenCV | 4.8.1.78 |

```bash
git clone https://github.com/akashrma/epa-medical-uda.git
cd epa-medical-uda
conda create --name epa-mmcv2 --override-channels --channel conda-forge python=3.11.17 pip --no-default-packages --yes
conda activate epa-mmcv2
python -m pip install pip==25.1.1 setuptools==75.1.0 wheel==0.44.0 packaging==24.1
python -m pip install numpy==1.26.4 torch==2.1.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-mmcv2.lock.txt --only-binary=mmcv
python -m pip check
```

The lock includes medical metrics and compiled MMCV operations. Run commands
from this repository root so Python uses the local `mmseg` package. Keep
`mmcv-full`, `mmcv-lite` and upstream `mmsegmentation` out of this environment.
`requirements-mmcv2.txt` lists direct dependencies; `requirements.txt` records
the historical Python 3.8/MMCV 1 environment.
The runtime enforces the exact MMCV and MMEngine versions. The lock pins
all 78 Python distributions, including packaging tools; it does not pin the
host driver or conda base packages.

Check local package resolution, CUDA forward/backward and a compiled
operation before training. This needs no dataset or model weights:

```bash
python - <<'PY'
from pathlib import Path

import mmcv
import mmengine
import torch
from mmcv.ops import nms

import mmseg

assert mmcv.__version__ == '2.1.0'
assert mmengine.__version__ == '0.10.7'
assert torch.__version__ == '2.1.0+cu118'
assert Path(mmseg.__file__).resolve() == Path('mmseg/__init__.py').resolve()
assert torch.cuda.is_available(), 'A visible NVIDIA GPU is required'
x = torch.randn(1, 3, 16, 16, device='cuda', requires_grad=True)
conv = torch.nn.Conv2d(3, 4, 3).cuda()
conv(x).square().mean().backward()
assert torch.isfinite(x.grad).all()
boxes = torch.tensor([[0., 0., 10., 10.]], device='cuda')
_, keep = nms(boxes, torch.tensor([.9], device='cuda'), .5)
assert keep.tolist() == [0]
print('Local package:', mmseg.__file__)
print('GPU:', torch.cuda.get_device_name(0))
PY
```

## 2. Prepare data and initialization weights

Download the prepared dataset and pretrained models from the
[project download folder](https://drive.google.com/drive/folders/1BcGtDlH80o15i-A_VTL-P6UPHNJnaI5D?usp=sharing).
Extract the prepared slice arrays and place the MiT-B5 initialization at
`pretrained/mit_b5_mmseg.pth`. Dataset arrays and large model checkpoints
are downloaded separately from the Git repository.

```text
epa-medical-uda/
├── data/
│   ├── data_np/
│   │   ├── train_ct/       # ct_train_slice0.npy, ...
│   │   ├── gt_train_ct/    # ct_train_slice0_gt.npy, ...
│   │   ├── train_mr/       # mr_train_slice0.npy, ...
│   │   ├── gt_train_mr/    # mr_train_slice0_gt.npy, ...
│   │   ├── val_ct/         # ct_val_slice0.npy, ...
│   │   ├── gt_val_ct/      # ct_val_slice0_gt.npy, ...
│   │   ├── val_mr/         # mr_val_slice0.npy, ...
│   │   └── gt_val_mr/      # mr_val_slice0_gt.npy, ...
│   └── datalist/          # generated below
├── pretrained/
│   └── mit_b5_mmseg.pth
└── etf_prototypes/
    ├── etf_prototypes_K5_D768.pt
    ├── etf_prototypes_K5_D768.npy
    └── etf_prototypes_K5_D768.json
```

The ETF assets are already tracked in Git. Preserve them for reproduction:
missing prototypes are generated automatically, which changes initialization
and consumes RNG state. An `.npy` prototype alone does not replace the `.pt`
and metadata files used by the loader.

Images are prepared single-channel NumPy slices with the configured `[-1, 1]`
input scale; the loader expands them to three channels and applies the
existing BGR normalization. Masks contain integer class IDs:

| ID | Class |
| --- | --- |
| 0 | background |
| 1 | Myo |
| 2 | LAC |
| 3 | LVC |
| 4 | AA |

Use the supplied preprocessing and splits to compare results. This repository
does not provide a raw NIfTI-to-training-slice preprocessing recipe.

### Generate the exact ordered datalists

```bash
python tools/prepare_medical_datalists.py
python tools/prepare_medical_datalists.py --check
```

The generator checks that every required array exists and writes eight lists
under `data/datalist/`. It preserves existing matching lists and refuses to
overwrite different lists. Paths are relative to each list file, for example:

```text
# train_ct.txt (one path per line; omit this comment in the actual list)
../data_np/train_ct/ct_train_slice0.npy
../data_np/train_ct/ct_train_slice1.npy
```

Image and mask lists have the same length and matching row order. The
recorded splits are:

| Modality | Training slices | Validation slices |
| --- | --- | --- |
| CT | 8,400 | 1,200 |
| MR | 9,600 | 2,400 |

The selected pipelines load target training masks, but `UDADataset` passes
only target images/metadata to UDA; target ground truth is not used by the
training loss. The supplied mask lists are still required by these configs.
Volume `.npz` test sets and historical source-only configs use a separate
protocol from this validation-slice comparison.

Check the initialization assets and generated lists against the recorded
SHA-256 hashes:

```bash
sha256sum -c resources/reproduction_assets.sha256
```

The expected list hashes assume the default layout shown above.

## 3. Reproduce the 5,000-update comparison

These two configs use batch size 4, four workers, 256×256 crops, EPA/ETF,
FDA L=0.01 and seed 0. Training uses FP32, convolution TF32 disabled, matmul
TF32 enabled and cuDNN benchmark enabled. Run the directions sequentially
on one GPU, using fresh work directories.

Resolve the first config without building a model or starting GPU work:

```bash
python tools/train.py configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir work_dirs/ct2mr_config_check --seed 0 --stop-after 5000 --dry-run
```

Train CT → MR:

```bash
python tools/train.py configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir work_dirs/ct2mr_epa_fda_seed0_5k --seed 0 --stop-after 5000
```

Train MR → CT:

```bash
python tools/train.py configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir work_dirs/mr2ct_epa_fda_seed0_5k --seed 0 --stop-after 5000
```

**The default cap is 5,000 optimizer updates.** `--stop-after` controls when
the run stops; the configured 50,000-step LR/UDA horizon stays unchanged.
Validation runs every 500 updates and selects the strictly highest target
foreground mDice. The commands above start fresh optimization from the
MiT-B5 initialization; omit resume/load overrides for this comparison.

Each run saves:

- `experiment_spec.py`, `native_config.py` and `launch_manifest.json`.
- Source archive, logs and TensorBoard output.
- `best_target_mDice_iter_N.pth`: student weights selected by target mDice.
- `iter_5000.pth`: full state at the stop cap, usable for native resume.
- `early_stop.json`: update counters and configured horizon.

The default periodic full-checkpoint interval is 50,000 updates. A full
checkpoint is also saved at the requested stop cap. Use
`python tools/train.py --help` for all options. Native MMEngine configs can
also be supplied directly to the entrypoint.

### Evaluate the best checkpoint

Use the same config as training. Set `CT_CHECKPOINT` and `MR_CHECKPOINT` to
the best files produced by your runs; the selected iteration can vary.

```bash
CT_CHECKPOINT=work_dirs/ct2mr_epa_fda_seed0_5k/best_target_mDice_iter_3500.pth
MR_CHECKPOINT=work_dirs/mr2ct_epa_fda_seed0_5k/best_target_mDice_iter_3000.pth

python tools/test.py configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  "$CT_CHECKPOINT" --eval mDice --metrics-out work_dirs/ct2mr_epa_fda_seed0_5k/best_target_metrics.json
python tools/test.py configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  "$MR_CHECKPOINT" --eval mDice --metrics-out work_dirs/mr2ct_epa_fda_seed0_5k/best_target_metrics.json
```

CT → MR evaluates `val_mr` (2,400 slices); MR → CT evaluates `val_ct`
(1,200 slices). The JSON reports raw 0–1 foreground/per-class Dice and ASSD;
console Dice is shown as a percentage. Background is excluded from the mean.
ASSD follows the original equally weighted directional means and empty-mask
rules, rather than the installed MedPy pooled-distance default.
The slice evaluator uses an ASSD fallback of 1.0 when the surface-distance
calculation fails, including empty masks. Historical NPZ volume evaluation
also retains both-empty Dice=0. Preserve these definitions for comparison.

Add `--out FILE.pkl` to retain predictions. For diagnostic source evaluation,
override `data.test.img_list` and `data.test.ann_list` with the source list
pair. For CT → MR, for example:

```bash
python tools/test.py configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  "$CT_CHECKPOINT" --eval mDice --metrics-out work_dirs/ct2mr_epa_fda_seed0_5k/best_source_metrics.json \
  --cfg-options data.test.img_list=data/datalist/val_ct.txt data.test.ann_list=data/datalist/val_ct_gt.txt
```

For MR → CT source evaluation, use the MR validation pair. Native configs
use `test_dataloader.dataset.img_list` and `.ann_list` instead. Evaluation
uses batch size 1 and four workers. Test-time augmentation, slide inference,
FP16 and NPZ volumes use a different protocol from the recorded slice results.

### Recorded migration results

Each stack independently selects its best target checkpoint; source and
final-checkpoint evaluations do not determine acceptance. Dice below is
reported on the 0–1 scale.

| Direction | Legacy best iteration | Modern best iteration | Legacy mDice | Modern mDice | Legacy mASSD | Modern mASSD |
| --- | --- | --- | --- | --- | --- | --- |
| CT → MR | 5,000 | 3,500 | 0.6857276685 | 0.7307150253 | 2.2916152708 | 2.1831863543 |
| MR → CT | 4,000 | 3,000 | 0.8432316865 | 0.8310812836 | 1.4677211894 | 1.5168075542 |

Higher Dice and lower ASSD are better. CT → MR mean Dice improves by
4.499 percentage points and mean ASSD by 4.73%; its AA ASSD increases by
0.29552 pixel. MR → CT mean Dice decreases by 1.215 points, AA Dice by
3.965 points, and mean ASSD increases by 3.34%. The migration review accepted
these measured differences on 2026-10-13. Earlier automatic Dice gates
failed; acceptance records the review decision rather than a claim that
those gates passed. These are 5,000-update, best-target results; the paper's
full training protocol has not been rerun.

Execution is nondeterministic, so exact scores and best iterations can vary.
The recorded campaign predates the cleanup's tiny DropPath rounding change
to ordinary FP32 `torch.linspace`. The cleanup checks include config
equivalence, exact CPU FDA outputs/gradients, pretrained initialization,
strict best-checkpoint loading and one source-training update per direction.
It does not include a new 5,000-update accuracy campaign. FDA keeps its real
inverse `irfft2`; replacing it with `ifft2(...).real` changes the inputs.

Historical code and full reports are preserved at Git commit `a05cff5`.
Local ignored evidence remains under `work_dirs/migration_baseline/` and
`work_dirs/mmcv2/fp32_conv_seed0_5k/`; `comparison_contract_v5.json` in the
former and `accuracy_review_v5.json` in the latter record the accepted policy
and results. These local evidence directories are not part of a Git clone.

## 4. Start a new training run

For a new seed and the full configured 50,000 updates, set the cap explicitly:

```bash
python tools/train.py configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir work_dirs/ct2mr_epa_fda_seed1_50k --seed 1 --stop-after 50000
```

This keeps the original schedule and extends training beyond the validated
5,000-update comparison. Its accuracy has not been measured here. You can
choose any positive stop cap up to `runner.max_iters`.

Use a child config for a new experiment, preserving the original reference:

```python
# Save as configs/medical/ct2mr_epa_fda_custom.py.
_base_ = ['./ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py']
seed = 1
optimizer = dict(lr=6e-5)
```

Launch it with `python tools/train.py CONFIG --work-dir FRESH_WORK_DIR
--stop-after 50000`. `--cfg-options KEY=VALUE` supports overrides, for example
`--cfg-options checkpoint_config.interval=1000` to retain periodic full
checkpoints for recovery. The default `max_keep_ckpts=1` retains one periodic
checkpoint; choose a larger value if needed. Best weights are separate.

For a different dataset, supply paired image/mask lists with the same class
mapping and override `data.train.source`, `data.train.target`, `data.val`
and `data.test` in a child config. The datalist generator is specifically for
the supplied slice names/counts. Changes to data, batch size, crops, loss,
optimizer or horizon define a new experiment and require validation.

`run_experiments.py --config CONFIG` remains available as a wrapper; it writes
child configs under `configs/generated/` and results under `work_dirs/`.
For a two-update CLI smoke run in a separate output directory:

```bash
python run_experiments.py --config configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir work_dirs/ct2mr_startup_smoke --startup-test
```

MIC, dynamic prototypes, AMP, distributed UDA and other architecture/config
variants remain research code outside the qualified single-GPU EPA/FDA scope.
Unsupported runtime policies may be rejected by config translation.

## 5. Resume or initialize from a checkpoint

Resume from a **full native iteration checkpoint**, retaining the original
config, seed, horizon, precision and data/sampling settings:

```bash
python tools/train.py configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py \
  --work-dir work_dirs/ct2mr_epa_fda_seed0_5k \
  --resume-from work_dirs/ct2mr_epa_fda_seed0_5k/iter_5000.pth --seed 0 --stop-after 50000
```

`--resume` without a path discovers the last full checkpoint in the work
directory. Native resume restores optimizer/scheduler, teacher/prototypes,
UDA counters, RNG and data-stream state. Floating continuation can differ
under nondeterministic CUDA execution.

| Checkpoint | Evaluation / `--load-from` | Full native resume |
| --- | --- | --- |
| Legacy, matching student architecture | Strict student load | Unsupported |
| Native best weights | Strict student load | Unsupported: optimizer/scheduler absent |
| Native full iteration checkpoint | Strict student load | Supported with matching runtime/data contract |

Loading strips an anchored `module.` prefix and selects `model.` student
keys from UDA checkpoints; every student key and tensor shape must match.
`--load-from CHECKPOINT` starts fresh optimization, UDA and sampling state.
GPU count, horizon, precision and data/sampling settings must match for
resume. CPU and deterministic CUDA continuation passed exact qualification
checks; the supported nondeterministic CUDA policy can change floating
continuation. A shorter diagnostic stop can save a full resume checkpoint
without changing the schedule horizon.

## Repository layout

| Path | Purpose |
| --- | --- |
| `mmseg/` | Local segmentation, EPA/UDA, data and MMEngine implementation |
| `configs/_base_/` | Models, datasets, optimizer, schedule and UDA bases |
| `configs/medical/` | Direction/method experiment configs |
| `etf_prototypes/` | Fixed prototype tensors and metadata, tracked in Git |
| `tools/` | Training, evaluation, datalist preparation and analysis |
| `data/`, `pretrained/` | Local dataset and downloaded weights |
| `configs/generated/`, `work_dirs/` | Generated configs and run artifacts |

Back up checkpoints, resolved configs, manifests, logs and metrics separately;
run artifacts are ignored by Git. Historical migration reports remain at
commit `a05cff5` (the preserved `mmcv2` branch in the development checkout).
