# Qualified MMCV 2 environment

All migration edits, checks, and progress updates must occur on `mmcv2`.
The `main` branch and the existing `epa` environment must remain unchanged.

Step 3 passed on 2026-10-11. The isolated environment is
`/home/akash/miniconda3/envs/epa-mmcv2`. This qualifies the dependencies and
their CUDA operations. Step 4 has since qualified local model construction,
exact legacy initialization and source-training/checkpoint smoke checks;
see `MIGRATION_STEP4.md`. Step 5 qualified dataset transport and medical
evaluation (`MIGRATION_STEP5.md`). Step 6 qualified native UDA training,
learning-rate curves, checkpoint replay and source/target evaluation smoke
(`MIGRATION_STEP6.md`). Original-config/CLI integration and cross-stack
numerical qualification subsequently passed steps 7-8. Step 9's matched
modern accuracy runs are in progress. The local guard accepts the qualified
pins; the current scope and reproduction guide is `MIGRATION_SUPPORT.md`.

## Installed stack

| Component | Qualified version |
| --- | --- |
| Python | 3.11.17, conda-forge CPython |
| Torch / torchvision | 2.1.0+cu118 / 0.16.0+cu118 |
| CUDA runtime / cuDNN | 11.8 / 8.7.0 |
| MMCV | 2.1.0, compiled operations included |
| MMEngine | 0.10.7 |
| NumPy / SciPy | 1.26.4 / 1.11.4 |
| timm / Kornia | 0.4.12 / 0.5.8 |
| MedPy / SimpleITK | 0.5.2 / 2.3.1 |
| OpenCV | 4.8.1.78 |
| Packaging tools | pip 25.1.1, setuptools 75.1.0, wheel 0.44.0, packaging 24.1 |

The host is an RTX 4080 Laptop GPU, compute capability 8.9, with about
12 GiB VRAM and driver 595.99.02. The selected official MMCV wheel reports
CUDA 11.8 and GCC 9.3. The
[official wheel index](https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html)
provides this CPython 3.11 build, and the
[PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/)
specify the matching Torch/torchvision pair. No GitHub source build was needed.

Use `mmcv` with operations, keeping `mmcv-full` and `mmcv-lite` absent, as
described in the
[MMCV installation documentation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
Keep the repository's local `mmseg` package authoritative; do not install an
upstream `mmsegmentation` distribution over it.

## Reproduction

Run from the repository root on `mmcv2`. These commands create a new
environment; preserve `epa` for the legacy runs.

```bash
git switch mmcv2
conda create --name epa-mmcv2 --override-channels --channel conda-forge python=3.11.17 pip --no-default-packages --yes
conda activate epa-mmcv2
python -m pip install pip==25.1.1 setuptools==75.1.0 wheel==0.44.0 packaging==24.1
python -m pip install numpy==1.26.4 torch==2.1.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-mmcv2.lock.txt --only-binary=mmcv
python -m pip check
python -u tools/verify_mmcv2_environment.py --out-dir work_dirs/migration_baseline/environment_recheck
```

Choose an unused verification output directory. GPU verification needs host
GPU access; restricted sandboxes may hide CUDA. The script also requires the
captured legacy dependency probes under the evidence directory below.

`requirements-mmcv2.txt` records the direct dependency choices.
`requirements-mmcv2.lock.txt` pins all 78 installed Python distributions,
including packaging tools and transitive dependencies. Both specify the
official PyTorch and MMCV download locations. They are version locks, with
download hashes retained separately in installation reports.

The exact conda provisioning base is also saved as
`work_dirs/migration_baseline/step3_environment_20261011/conda_base_explicit.txt`.
For an exact Linux conda base, substitute the following creation command,
then apply the packaging bootstrap and Python package lock above:

```bash
conda create --name epa-mmcv2 --file work_dirs/migration_baseline/step3_environment_20261011/conda_base_explicit.txt --yes
```

The conda base records the original provisioning packages; the Python lock
records the final versions after pip overrides. Keep the ignored evidence
directory with the experiment artifacts when reproducing these checks.

## Dependency audit

Every entry in the original `requirements.txt` is retained or replaced with
an explicit compatible pin. MedPy, SimpleITK, MMEngine, TensorBoard,
protobuf, scikit-learn, and packaging tools are also pinned explicitly.

| Original requirement | Modern pin | Check or required follow-up |
| --- | --- | --- |
| torch 1.7.1+cu110 | 2.1.0+cu118 | CUDA forward/backward and AdamW passed; FDA inverse qualified in step 8 |
| torchvision 0.8.2+cu110 | 0.16.0+cu118 | Imports through timm passed; paired official build |
| timm 0.3.2 | 0.4.12 | Removes `torch._six` dependency; relevant layer ASTs and fixed-seed outputs match |
| kornia 0.5.8 | 0.5.8 | ColorJitter and GaussianBlur forward/backward passed |
| numpy 1.19.2 | 1.26.4 | Tensor/array probes passed; no removed NumPy aliases found in local `mmseg` code |
| scipy 1.6.3 | 1.11.4 | Legacy-compatible surface-distance probes passed |
| opencv-python 4.4.0.46 | 4.8.1.78 | Fixed medical preprocessing, resize and inference passed in step 5 |
| Pillow 8.3.1 | 10.4.0 | Imported with torchvision; config-specific transforms remain to be checked |
| matplotlib 3.4.2 | 3.8.4 | Import passed |
| pandas 1.1.3 | 2.1.4 | Import passed |
| seaborn 0.11.1 | 0.13.2 | Import passed |
| cityscapesscripts 2.2.0 | 2.2.0 | Label utilities import passed; optional for the selected medical configs |
| gdown 4.2.0 | 5.2.0 | Dependency check passed; update old downloader CLI usage in step 7 |
| cycler 0.10.0 | 0.12.1 | Dependency check and plotting import passed |
| humanfriendly 9.2 | 10.0 | Dependency check passed |
| kiwisolver 1.2.0 | 1.4.5 | Dependency check and plotting import passed |
| prettytable 2.1.0 | 3.10.2 | Dependency check passed; used by evaluation formatting |
| pyparsing 2.4.7 | 3.1.2 | Dependency check and plotting import passed |
| pytz 2020.1 | 2025.2 | Dependency check and pandas import passed |
| PyYAML 5.4.1 | 6.0.2 | MMEngine config load/dump passed |
| tqdm 4.48.2 | 4.66.5 | Dependency check passed |
| typing-extensions 3.7.4.3 | 4.12.2 | Dependency check and Torch imports passed |
| wcwidth 0.2.5 | 0.2.13 | Dependency check passed |
| yapf 0.31.0 | 0.40.1 | Both selected configs load and dump through MMEngine |

Retaining Kornia's version and timm's sampling/initialization formulas limits
unnecessary changes to training behavior. This does not establish complete
model or augmentation equivalence; steps 5 and 8 perform those comparisons.

MedPy 0.4.0 uses a removed NumPy boolean alias. MedPy 0.5.2 runs with the
new NumPy version, but its ASSD averages pooled surface distances. Legacy
MedPy averages the two directional means with equal weight. The irregular
saved mask gives `0.7810632058839182` with the legacy formula versus
`0.7890288595373066` with the new `assd` function. Step 5 implemented and qualified:

```python
try:
    value = np.mean((asd(prediction, reference), asd(reference, prediction)))
except Exception:
    value = 1.0
```

Retain the original voxel spacing/connectivity defaults and aggregation.
All five saved examples, including the empty-mask fallback, match the
legacy formula within `1e-10`. The adapter is verified in the environment
probe and the dataset port; see `MIGRATION_STEP5.md`.

## Verification and evidence

Evidence root: `work_dirs/migration_baseline/step3_environment_20261011/`.

- `verification_v2/verification.json`: successful dependency and GPU checks.
- `legacy_dependency_probes.json` and `.npz`: legacy timm and ASSD examples.
- `packages.txt`, `conda_base_explicit.txt`, and `pip_check.txt`: environment
  inventory and successful dependency consistency check.
- `torch_install_report.json` and `dependency_install_report.json`: download
  URLs, hashes, and dependency metadata.
- `legacy_packages_recheck.txt` and `manifest.json`: preservation checks,
  version-lock hashes, and pending source-port requirements.

Checks passed for both selected configs' MMEngine load/dump, local package
resolution, absence of conflicting MMCV/MMSeg distributions, exact timm
layer sampling and initialization probes, legacy-compatible ASSD, Kornia
augmentation gradients, OpenCV resize, plotting/data/medical utility imports,
and TensorBoard writing. On CUDA, convolution/BatchNorm/AdamW, CPU and CUDA
NMS, ROIAlign forward/backward, and FFT roundtrip passed. Peak Torch allocated
memory was 3,801,088 bytes; the probe capped its allocator at 10% of VRAM.

The first probe run had an incorrect keyword argument to the ROIAlign
autograd function. The corrected positional call passed in `verification_v2`;
the original failed report is retained in `verification` for provenance.
Torch emitted an `nvrtc.so` cuDNN workaround warning. Forward/backward passed
using that workaround; test its performance and numerical behavior during
the full-model smoke check in step 8.

At the end of step 3, all original tracked source hashes and all 81 legacy
package versions matched the step-1 audit. Step 4 then froze legacy source
and ported the working tree's foundational APIs and models. Both training
environments and original configs remain unchanged. At the step-4
checkpoint, CT -> MR legacy training was active and MR -> CT queued; both
were capped at 5,000 updates with the 50,000-iteration horizon retained.

The step-4 foundational API and model initialization port passed. The
[MMCV API migration documentation](https://mmcv.readthedocs.io/en/latest/get_started/api_reference.html)
describes the removed runner/parallel APIs and their MMEngine replacements.
The snapshot under `work_dirs/migration_baseline/legacy_source` must remain
frozen until the baseline queue and its checkpoint evaluations finish.
Step 5's data transport and medical evaluation port also passed:
`MIGRATION_STEP5.md` records exact tensors/metadata, shuffle RNG, native
inference, and fixed-prediction Dice/ASSD qualification. Both MedPy ASSD
aggregation and NPZ both-empty Dice now use the verified legacy definitions.
Subsequent training/CLI/numerical gates passed steps 6-8. Step 9's full
accuracy comparison remains pending; consult `PLAN.md` for current status.
