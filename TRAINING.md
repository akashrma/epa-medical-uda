# Training and evaluation

Use the [pinned environment](ENVIRONMENT.md) and run commands from the
repository root. Existing medical experiment configs are translated to
native MMEngine configs by the public training entrypoints. Native configs
can also be supplied directly.

## Tested configurations

- `configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`
- `configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py`

Both use one GPU, MiT-B5/SegFormer, five classes, no-mix EPA/ETF, FDA L=0.01,
256x256 crops, batch size 4 and four workers. Seed 0 and 5,000 updates were
used for the [recorded results](TRAINING_RESULTS.md). MIC, dynamic
prototypes, AMP, distributed UDA, other FDA geometries, other configs and
50,000-update accuracy have not been qualified. The wider config catalogue
is retained for research; translation rejects unsupported runtime settings.

Supply slice arrays under `data/data_np/` and initialization weights at
`pretrained/mit_b5_mmseg.pth`; run `python tools/prepare_medical_datalists.py`
to create the exact ordered train/validation lists under `data/datalist/`.
The fixed ETF assets are tracked in Git and should be preserved. Arrays,
datalists and large weights remain local. See [README.md](README.md) for the
download link, layout and `resources/reproduction_assets.sha256` checksums.

## Train

```bash
conda activate epa-mmcv2
python tools/train.py configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py --work-dir work_dirs/ct2mr_epa_fda --seed 0 --stop-after 5000
python tools/train.py configs/medical/mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py --work-dir work_dirs/mr2ct_epa_fda --seed 0 --stop-after 5000
```

Run sequentially on the single GPU and use fresh work directories for new
runs. The default update cap is 5,000. `--stop-after` changes the cap while
preserving the configured `max_iters=50000` LR/UDA horizon. Validation runs
every 500 updates. The periodic full-checkpoint interval is 50,000, and a
full checkpoint is also saved at the stop cap.

For a fresh full-length experiment, pass `--stop-after 50000` explicitly.
For example, use `--seed 1 --stop-after 50000` with a new work directory.
The 50,000-update path retains the configured schedule; its accuracy has not
been qualified. `--cfg-options checkpoint_config.interval=1000` can save
periodic full checkpoints for recovery. The default `max_keep_ckpts=1`
retains one periodic checkpoint; adjust that setting if needed.

Training/evaluation use FP32 storage, convolution TF32 disabled, matmul TF32
enabled, cuDNN benchmark enabled and nondeterministic CUDA execution. The
sampler, worker seeding, optimizer groups, AdamW execution, polynomial LR
and overlapping linear warmup preserve the tested experiment semantics.
Changing the precision policy, optimizer, data streams or LR timing requires
separate validation. DropPath rates use ordinary FP32 `torch.linspace`.
FDA uses `torch.fft.fft2` and the real inverse `irfft2`; the modified spectrum
is not Hermitian symmetric, so `ifft2(...).real` changes the inputs.

`--dry-run` writes `experiment_spec.py`, `native_config.py` and
`launch_manifest.json` without building a model or starting GPU work:

```bash
python tools/train.py configs/medical/ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py --work-dir work_dirs/ct2mr_dry_run --dry-run
```

Actual runs also retain a source archive, logs, best weights and the full
stopped checkpoint. `early_stop.json` records runner/UDA counters and the
configured horizon. `run_experiments.py --config CONFIG` remains available
as a wrapper; it writes child configs to `configs/generated/` and results
under `work_dirs/`. Use `python tools/train.py --help` for options.

## Evaluate

Use the same experiment config as training, replacing the placeholders:

```bash
python tools/test.py CONFIG CHECKPOINT --eval mDice --metrics-out METRICS.json
```

Evaluation loads the student and reports raw 0-1 foreground/per-class Dice
and ASSD using the local medical adapter. `--out FILE.pkl` retains
predictions. Use separate metric files for each checkpoint and domain.
Target best selection uses strictly greater foreground mDice over scheduled
validations; the saved file is `best_target_mDice_iter_N.pth`. A stopped
run also saves `iter_5000.pth` when the cap is 5,000.

| Direction | Target validation | Diagnostic source validation |
| --- | --- | --- |
| CT -> MR | `val_mr`: 2,400 slices | `val_ct`: 1,200 slices |
| MR -> CT | `val_ct`: 1,200 slices | `val_mr`: 2,400 slices |

For source evaluation, override the original config's `data.test.img_list`
and `data.test.ann_list` with the source image/GT list pair using
`--cfg-options`. Native configs use `test_dataloader.dataset.img_list` and
`.ann_list`. Keep preprocessing, evaluation batch size 1 and four workers.
Volume NPZ evaluation, test-time augmentation, slide inference and FP16 use
different protocols from the recorded validation-slice results.

## Checkpoint loading and resume

| Checkpoint | Evaluation / weight initialization | Full native resume |
| --- | --- | --- |
| Legacy, matching student architecture | Strict load / `--load-from` | Unsupported |
| Native best weights | Strict load / `--load-from` | Unsupported: optimizer/scheduler absent |
| Native full iteration checkpoint | Strict load / `--load-from` | Supported with matching runtime/data contract |

Loading strips an anchored `module.` prefix and selects `model.` student
keys from UDA checkpoints. Every student key and tensor shape must match.
`--load-from` starts fresh optimization, UDA and sampling state.

```bash
python tools/train.py CONFIG --work-dir WORK_DIR --resume-from FULL_ITER_CHECKPOINT --stop-after 5000
```

`--resume` without a path uses the work directory's last full checkpoint.
Resume restores student/teacher/prototypes, optimizer, scheduler, UDA
counters, RNG, data stream and gradient-presence state. GPU count, horizon,
precision and dataset/sampling contracts must match. CPU and deterministic
CUDA continuation passed exact checks during qualification. Under the
supported nondeterministic CUDA policy, restored state is exact but floating
continuation can differ. Best weights cannot resume optimization. A shorter
diagnostic stop can create a full resume checkpoint with the horizon retained.

Back up checkpoints, resolved configs, manifests, logs and metric evidence
separately: `work_dirs/` is ignored by Git. The cleanup preserves existing
local artifacts; no external backup has been made.
