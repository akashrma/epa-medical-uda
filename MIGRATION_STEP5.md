# Step 5: data transport and medical evaluation

All edits, checks, and progress updates belong on **`mmcv2`**. `main`, the
legacy `epa` environment, original experiment configs, and the verified
247-file frozen legacy source remain unchanged.

## Transport contract

The selected pipelines now return ordinary tensors and CPU metadata
dictionaries. `DefaultFormatBundle` also registers as `PackUDAInputs`.
`uda_collate` registers with MMEngine and keeps these training keys:

| Key | Batched representation |
| --- | --- |
| `img`, `target_img` | Float32 NCHW tensors |
| `gt_semantic_seg` | Int64 N1HW tensor |
| `valid_pseudo_mask`, when present | N1HW tensor, original mask dtype |
| `img_metas`, `target_img_metas` | Lists of per-image CPU dictionaries |

Testing preserves the outer augmentation lists around image and metadata
batches. Metadata includes original/resized/padded shapes, filenames, flips,
scale factors, and `img_norm_cfg`. `UDADataPreprocessor` subclasses MMEngine
`BaseDataPreprocessor` and transfers tensor fields without modifying metadata,
normalizing, padding again, or consuming RNG. Its integration into native
model training/validation steps is step 6. The inference/test APIs already
use the standalone adapter; training imports are deferred until that port.

NPY scaling, BGR replication, normalization, crop/flip order, labels and
ignore value 255 stay in the existing pipelines. NPZ slicing, flips, list
resolution and volume reconstruction retain their algorithms. Synchronized
UDA cropping now slices tensors directly. Removed `ToDataContainer` is not
used by the selected or checked-in configs; custom external configs using
that transform require explicit conversion.

Three compatibility details required explicit handling:

1. `LegacyRandomSampler` reproduces Torch 1.7.1's global RNG permutation and
   its unused seed draw. Torch 2's default RandomSampler uses a different
   generator and consumes another permutation at exhaustion. Keep this
   sampler in the step-7 native single-GPU loader config. Six small loader
   shuffle orders and post-epoch Torch RNG states match the legacy oracle.
2. Legacy containers were not pinned by PyTorch despite the config's
   `pin_memory=True`. Native loaders explicitly disable pinning to preserve
   this effective behavior and avoid conversion of metadata tuples to lists.
   Keep `persistent_workers=False`, matching the actual Torch 1.7 loader.
3. Batch collation retains the container's zero padding for variable spatial
   sizes. The pipeline separately applies label-255 padding. Both cases are
   checked against frozen legacy execution.

The sampler and collate function also resolve through an actual native
`Runner.build_dataloader` call. Preserve those settings during config
translation instead of selecting MMEngine's default infinite sampler.

## Medical evaluator

`MedicalMetric` implements native MMEngine `BaseMetric` collection and
delegates metric computation to the medical datasets. Use separate
evaluators with `prefix='target'` and `prefix='source'`; the resulting
checkpoint-selection key will be `target/mDice`. Source reporting remains
diagnostic. Validation must use an ordered, non-shuffling sampler. Standard
strided distributed collection is supported by the native base class;
multi-process distributed execution is still unqualified.

The evaluator preserves these protocol distinctions:

- **NPY:** total class intersections/areas over slices, then foreground
  `nanmean`; absent classes retain NaNs. ASSD is averaged over slices per
  foreground class, then over the four classes. Ignore 255 applies to Dice
  exactly as before.
- **NPZ:** ordered slices reconstruct each original volume; per-volume
  foreground Dice/ASSD is averaged across volumes and classes. The legacy
  binary Dice value for both-empty masks is **0.0**, rather than newer
  MedPy's 1.0. `legacy_dc` preserves this behavior.
- **ASSD:** `legacy_assd` averages the two directional means, preserving
  MedPy 0.4.0. MedPy 0.5.2 pools the surface distances. Existing exception
  handling still returns 1.0 for empty/invalid masks, using default
  connectivity and unit spacing (pixels for NPY, voxels for NPZ).

`MIGRATION_CONTRACT.md` records these observed implementation details without
changing the comparison tolerances or the selected validation splits.

The native extension points follow the official
[BaseDataPreprocessor API](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.model.BaseDataPreprocessor.html),
[BaseMetric API](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.evaluator.BaseMetric.html),
and [package migration guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/migration/package.md).
The implementation was checked against installed MMEngine **0.10.7**.

## Qualification

Evidence is under `work_dirs/migration_baseline/step5_data_20261011/`.
`tools/verify_mmcv2_data.py --legacy` imports the frozen source with the
untouched legacy interpreter and captures synthetic oracles. The modern
verification checks the following:

| Check | Result |
| --- | --- |
| CT -> MR and MR -> CT real four-pair fixed batches | All image/label tensors and metadata exactly equal |
| Source/target CUDA transfer | Values/dtypes unchanged; metadata objects remain on CPU |
| Three crop/flip seeds, pipeline padding, variable batch sizes | Exact frozen-legacy equality |
| Two worker seeds, optional pseudo mask, synchronized crop | Passed |
| Six sampler orders and final RNG states | Exact frozen-legacy equality |
| Six NPZ cases: axes 0/1/2, with/without flips; two volume shapes | Exact TTA tensors/metadata and metric results |
| NPY missing classes, false positives, ignored pixels, all-empty foreground | Exact finite metrics and NaN locations |
| Five complete historical source/target prediction sets | All metric values exactly equal; 8,224 slices total |
| Fresh CT -> MR 5,000-step source/target prediction sets | All metric values exactly equal; 3,600 slices |
| Real historical student: `single_gpu_test`, `inference_segmentor` | Strict student load; exact predictions against direct inference |

The full historical metric report is `verification_cuda.json`; additional
worker/padding checks are in `preprocessing_cuda.json`. CPU inference evidence
is `inference_smoke.json`. Fresh baseline metrics are in
`ct2mr_epa_fda_seed0_5k_final_evaluation_metrics.json`. Finite metric tolerance
is `1e-10`, with matching NaNs; observed deltas are zero.

Reproduction commands, from the repository root on `mmcv2`:

```bash
/home/akash/miniconda3/envs/epa/bin/python -u tools/verify_mmcv2_data.py --legacy
/home/akash/miniconda3/envs/epa-mmcv2/bin/python -u tools/verify_mmcv2_data.py --device cuda
/home/akash/miniconda3/envs/epa-mmcv2/bin/python -u tools/verify_mmcv2_data.py --inference-smoke
/home/akash/miniconda3/envs/epa-mmcv2/bin/python -u tools/verify_mmcv2_data.py --reference-dir work_dirs/migration_baseline/ct2mr_epa_fda_seed0_5k/final_evaluation
```

CUDA and DataLoader multiprocessing checks need host access; sandbox worker
socket communication is restricted. The expected all-empty NPY fixture emits
the same `Mean of empty slice` warning as legacy execution. Full pre-commit
was not run; changed Python files were formatted with project isort/YAPF
settings, compiled, and checked with Flake8 (existing CLI bootstrap E402
exception). Validation artifacts stay out of commits.

## Baseline progress and next gate

Fresh legacy CT -> MR completed exactly **5,000 updates**, preserving both
configured runner/UDA horizons at **50,000**. Final and best checkpoints
(best at 5,000) give target **mDice 0.6857276685**, diagnostic source
**mDice 0.8769931395**, and target **mASSD 2.2916152708**. The isolated
supervisor has started MR -> CT with the same 5,000-update stop.

Step 6 must integrate this data preprocessor and evaluator with native
training/validation, preserve all UDA backward/EMA/optimizer/LR semantics,
and qualify checkpoint resume. Full-size sampled training sequences and
modern retraining accuracy remain pending steps 6-9. Step 5 demonstrates
equivalent preprocessing/evaluation, not trained-model accuracy parity.
