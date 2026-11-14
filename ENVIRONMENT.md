# Environment

Use the pinned Python 3.11/MMCV 2 stack for this checkout. The local
`mmseg` package supplies the models and MMEngine integration; do not install
upstream `mmsegmentation` over it. Install compiled-op `mmcv`, with
`mmcv-full` and `mmcv-lite` absent.

## Qualified versions

| Component | Version |
| --- | --- |
| Python | 3.11.17 |
| PyTorch / torchvision | 2.1.0+cu118 / 0.16.0+cu118 |
| CUDA runtime / cuDNN | 11.8 / 8.7.0 |
| MMCV / MMEngine | 2.1.0 / 0.10.7 |
| NumPy / SciPy | 1.26.4 / 1.11.4 |
| timm / Kornia | 0.4.12 / 0.5.8 |
| MedPy / SimpleITK | 0.5.2 / 2.3.1 |
| OpenCV | 4.8.1.78 |

`requirements-mmcv2.txt` records direct dependencies;
`requirements-mmcv2.lock.txt` pins all 78 Python distributions, including
transitive dependencies and packaging tools. Keep those versions for
reproduction. The runtime enforces the exact MMCV and MMEngine versions.
The tested GPU is an RTX 4080 Laptop GPU with about 12 GiB VRAM.

## Installation

Run from the repository root. Choose an unused environment name if needed.

```bash
conda create --name epa-mmcv2 --override-channels --channel conda-forge python=3.11.17 pip --no-default-packages --yes
conda activate epa-mmcv2
python -m pip install pip==25.1.1 setuptools==75.1.0 wheel==0.44.0 packaging==24.1
python -m pip install numpy==1.26.4 torch==2.1.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-mmcv2.lock.txt --only-binary=mmcv
python -m pip check
```

The requirements include the official PyTorch and MMCV wheel locations.
A compatible NVIDIA driver and visible CUDA GPU are needed for training.
The Python lock does not pin the host driver or the conda base packages.

## Dependency and CUDA check

This small check needs no datasets, weights, or saved experiment fixtures.
Run it from the repository root with GPU access.

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
assert torch.cuda.is_available()
x = torch.randn(1, 3, 16, 16, device='cuda', requires_grad=True)
conv = torch.nn.Conv2d(3, 4, 3).cuda()
conv(x).square().mean().backward()
assert torch.isfinite(x.grad).all()
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device='cuda')
scores = torch.tensor([.9, .8], device='cuda')
_, keep = nms(boxes, scores, .5)
assert keep.tolist() == [0]
print('CUDA and MMCV ops passed:', torch.cuda.get_device_name(0))
PY
```

This checks package resolution, CUDA autograd and a compiled MMCV operation.
Training and checkpoint scope are documented in [TRAINING.md](TRAINING.md).

## Medical metrics

The local metric adapter keeps the original ASSD definition: the equally
weighted mean of the two directional surface-distance means, with the
original spacing/connectivity defaults and empty-mask fallback. MedPy
0.5.2's pooled-distance ASSD has different semantics. Volume NPZ evaluation
also preserves the original both-empty Dice value of zero. Keep the local
adapter when reproducing the [recorded results](TRAINING_RESULTS.md).

Historical environment instructions and migration reports remain in Git at
`a05cff5`; `requirements.txt` is retained as the legacy dependency record.
Use the modern lock above for the current runtime.
