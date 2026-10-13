# MMCV 2 migration comparison contract

Version 1, frozen on 2026-10-11 before modern training results exist.
Versions 2/3/4/5 acceptance revisions authorized on 2026-10-13, after results.
All edits and progress updates must occur on `mmcv2`; leave `main` unchanged.
This contract records step 2 of [PLAN.md](PLAN.md). Legacy training and
checkpoint measurements are complete; original protocol notes are preserved below.

## Current acceptance: version 5

On 2026-10-13 the user accepted the current results and declared the migration
successful. Both independently selected best target checkpoints are accepted.
Improvements are allowed, source/final evaluations are excluded, and lower
ASSD is better. The measured MR -> CT mean/AA Dice and ASSD differences
are accepted deviations, not automatic passes under earlier tolerances.
No further accuracy training or repeated-seed comparisons are required.

Structural and immutable numerical qualification evidence stays unchanged.
Earlier contracts, automatic gate outcomes, manifests and review JSON remain
preserved. Current machine-readable acceptance is in
`work_dirs/migration_baseline/comparison_contract_v5.json` and candidate
`accuracy_review_v5.json`. The portable decision, scope and results are in
[MIGRATION_ACCEPTANCE.md](MIGRATION_ACCEPTANCE.md).

The sections below retain the original protocol and its historical status.
The authorized modern precision candidate disables convolution TF32 while
retaining matmul TF32, benchmark and nondeterminism; see
[MIGRATION_PRECISION_EXPERIMENT.md](MIGRATION_PRECISION_EXPERIMENT.md).

The scope is EPA/FDA in both adaptation directions, seed 0, one GPU, and
exactly 5,000 optimizer updates. Keep the configured runner and UDA horizon
at 50,000 iterations, including the original learning-rate schedule. Apply
the same stop rule to legacy and modern runs. These comparisons establish
accuracy at 5,000 updates.

## Experiments and data

| Direction | Original config under `configs/medical/` | Source training slices | Target training slices | Target validation slices | Source diagnostic slices |
| --- | --- | --- | --- | --- | --- |
| CT -> MR | `ct2mr_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py` | CT: 8,400 | MR: 9,600 | MR: 2,400 | CT: 1,200 |
| MR -> CT | `mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py` | MR: 9,600 | CT: 8,400 | CT: 1,200 | MR: 2,400 |

Use `data/datalist/train_{ct,mr}.txt`, their `_gt.txt` annotation lists, and
`val_{ct,mr}.txt`/`val_{ct,mr}_gt.txt` exactly as audited. The current configs'
`data.val` and `data.test` both identify target validation slices. Target
annotations are loaded by the legacy pipeline but are not supplied to UDA
training as target supervision. Preserve that behavior.

The supplied MR -> CT checkpoint's historical score used CT NPZ test volumes.
Keep that volume score as supplemental evidence. Its 8,000-update checkpoint
also exceeds the new stopping point, so its slice score is an inference
reference, not the fresh 5,000-update training baseline. Fresh runs provide
the accuracy references for this contract.

The machine-readable freeze is
`work_dirs/migration_baseline/comparison_contract_v1.json`. It contains the
resolved configs, config hashes, datalist counts/hashes, initialization and
prototype hashes, runtime rules, and numerical reference paths. The original
dataset-file and source manifests remain under
`work_dirs/migration_baseline/step1_audit_20261011/`.

## Settings to preserve

| Area | Contract |
| --- | --- |
| Model | Local `mit_b5` backbone and local `SegFormerHead`; five classes; decoder embedding 768; BatchNorm; dropout 0.1; bilinear feature/logit resizing with `align_corners=False` |
| Initialization | Original `pretrained/mit_b5_mmseg.pth`; same seeded head initialization and module construction order; compare initialized tensors against saved legacy state |
| Labels | 0=background, 1=Myo, 2=LAC, 3=LVC, 4=AA; ignore 255; `reduce_zero_label=False`; no class remapping |
| Losses | Local DiceCE: CE weight 0.5, Dice weight 0.5, total weight 1; mean reduction; Dice smooth 1, exponent 2, epsilon `1e-6`; preserve weighting and reductions, including background in the training loss |
| EPA/ETF | Source and target loss weights 0.2, temperatures 0.05; load existing K=5/D=768 `.pt` and metadata; preserve class ordering and feature extraction hook |
| FDA | Source-to-target, probability 1, L=0.01, denormalized input; preserve FFT conventions, amplitude replacement, clamping, and renormalization |
| UDA | No class mixing; target-only segmentation and target ETF start when zero-based `local_iter >= 2000`, first active update 2,001; feature distance, MIC, and dynamic prototype losses disabled |
| Teacher | Initialize teacher parameters from the student at update 1; EMA before subsequent forwards with `min(1 - 1/(local_iter+1), 0.99)`; disable teacher dropout/DropPath while preserving legacy BatchNorm behavior |
| Pseudo labels | Threshold 0.968; legacy batch-wide fraction above threshold supplies the segmentation pixel weight; preserve ETF target weighting separately |
| Updates | One `zero_grad(set_to_none=False)` and one AdamW step per batch; retain the original multiple backward passes and their order; preserve UDA `local_iter` and buffers |
| Optimizer | AdamW, LR `6e-5`, betas (0.9, 0.999), epsilon `1e-8`, weight decay 0.01, AMSGrad false; head LR multiplier 10; legacy custom-key precedence and parameter groups preserved |
| Execution | Seed 0; one GPU; source batch 4 plus paired target batch 4; four workers; no gradient accumulation or automatic LR scaling |

The optimizer-group inventories contain 2,134 groups, including teacher
parameters; preserve names, order, options, and which parameters receive
gradients. A broad modern "no decay for all normalization/bias" rule would
change these groups: the legacy rule matches parameter names containing
`norm`, rather than every normalization module. Use the saved inventory.
Select the non-fused, per-parameter AdamW execution for the first port
(`foreach=False`, `fused=False` where supported), and verify its updates
against the saved traces during step 8.

## Preprocessing, augmentation, and sampling

For both domains, load the NPY array, expand a single channel to three, scale
`[-1,1]` with `(image + 1) * 127.5`, reverse channels to BGR, and retain
float32. Random-crop to 256x256 with the legacy default category-ratio rule,
then horizontal-flip with probability 0.5. Source and target transforms run
independently; synchronized cropping and rare-class sampling are disabled.
Normalize using means `[104.00698793, 116.66876762, 122.67891434]`, standard
deviations `[1,1,1]`, and `to_rgb=False`. Pad images with 0 and labels with 255.
Preserve NCHW layout, label dtype, metadata, and collation semantics.

Strong augmentation keeps the actual legacy predicates: color jitter uses
strength 0.2 and applies when a uniform draw is greater than the configured
0.2, giving an 80% outer application probability. Blur applies when its draw
is greater than 0.5, with sigma sampled uniformly from 0.15 to 1.15 and the
legacy image-size-dependent kernel. Keep Kornia parameter generation and
random-number consumption, including augmentation constructed before the
target segmentation start.

`UDADataset` has 80,640,000 source/target pairs. Pair index `i` maps to source
`i // len(target)` and target `i % len(target)`. Single-GPU training uses
PyTorch RandomSampler without replacement through `shuffle=True`, rather
than MMEngine's default infinite sampler. Keep training `drop_last=True`,
`pin_memory=True`, four workers, and effective `persistent_workers=False`
(the legacy builder does not pass that option to Torch 1.7). Worker NumPy
and Python seeds are `num_workers * rank + worker_id + seed`; retain
PyTorch's DataLoader-generated worker seed separately. Preserve loader
construction/RNG order. Matching seed values alone does not prove matching
sample or augmentation sequences; verify them during the data port.

Step-5 implementation clarification (2026-10-11): keep the *effective*
Torch 1.7.1 RandomSampler sequence with `LegacyRandomSampler`. Torch 1.7.1
consumes an unused generator seed and uses global RNG for `randperm`;
Torch 2 changes this behavior. Six small loader orders and post-epoch RNG
states match the frozen legacy oracle. Full-size run sequence/resume checks
remain part of steps 6/8. Although legacy config requested `pin_memory=True`,
its DataContainers were not pinned by PyTorch. Native loaders explicitly use
`pin_memory=False` to retain unpinned tensors and CPU metadata tuple types.
The original experiment configs stay unchanged.

Validation/inference uses one image per GPU, four workers, no shuffling,
no dropping, no flip augmentation, and 256x256 resizing with
`keep_ratio=False`. Preserve the legacy image-resize backend/interpolation,
logit resizing to the original shape, softmax/argmax order, and student
inference. Do not evaluate the EMA teacher as the primary model.

## Schedule, stopping, and checkpoint selection

Use the recorded zero-based LR curve in each selected `final_trace_*_epa/`:
linear polynomial decay over 50,000 iterations, power 1, minimum LR 0,
linear warmup for 1,500 iterations, warmup ratio `1e-6`. Warmup multiplies
the already decaying polynomial LR. Representative backbone values are:

| Zero-based iteration | LR |
| --- | --- |
| 0 | `6.000000000172534e-11` |
| 1 | `4.005915880080067e-8` |
| 1,499 | `5.81623992388008e-5` |
| 1,500 | `5.82e-5` |
| 2,000 | `5.76e-5` |
| 4,999 | `5.40012e-5` |

The corresponding head LR is ten times the backbone value. At the stop,
LR has not decayed to zero. Validation runs after updates 500, 1,000, ...,
5,000. Select best by strictly greater, unrounded target validation mDice;
retain the earlier checkpoint on a tie. Save `iter_5000.pth` after the final
validation and best-checkpoint selection, then close the training hooks.

Compare stopped checkpoint to stopped checkpoint, and each stack's own
validation-selected best to its counterpart. Do not choose a modern
checkpoint using source diagnostics, test volumes, or knowledge of the
legacy result. Record checkpoint hash, iteration, config, seed, and protocol.

## Metrics and acceptance

Use the local `MedNpyListDataset.evaluate` definition. For each foreground
class, sum intersections and prediction/ground-truth areas across all slices,
then compute Dice. Target mDice is `nanmean` of Myo/LAC/LVC/AA Dice, excluding
background; it is not a mean of per-slice Dice. Ignore label 255 during Dice
accumulation and preserve legacy NaN handling.

ASSD uses MedPy on each slice/class, averages slices within each class, then
averages foreground classes. Preserve the legacy exception/empty-mask value
of 1.0, default connectivity and unit voxel spacing; units are pixels. Keep
this implementation's ASSD mask handling unchanged during the first port.

Step-5 metric clarification: MedPy 0.4.0 ASSD averages the two directional
means, whereas newer MedPy pools surface distances. Use `legacy_assd` in
both medical datasets. Historical NPZ evaluation averages binary Dice
per volume/class, with both-empty masks receiving **0.0** under MedPy 0.4.0
(newer MedPy returns 1.0). Use `legacy_dc` for this protocol. This differs
from NPY total-area Dice, which retains NaNs for absent foreground classes.
These definitions were verified against frozen legacy execution; comparison
tolerances and the current validation-slice protocol are unchanged.

Adopt the previously proposed tolerances as version-1 working criteria for
the user's request for "about the same accuracy":

- For each direction, require `abs(modern_mDice - legacy_mDice) <= 0.01`
  separately for stopped and best checkpoints: at most one percentage point.
- Flag every foreground class with absolute Dice difference greater than
  0.02. Resolve such differences before accepting the migration, even if
  aggregate mDice passes.
- Report source mDice, source/target per-class Dice and ASSD, runtime, and
  peak GPU memory for both checkpoint selections. ASSD and source results
  are diagnostic; investigate material regressions. No unmeasured ASSD
  accuracy cutoff is invented for this contract.
- Both directions and both checkpoint selections must pass the primary
  criterion; do not average away a failure. A score improvement outside
  the symmetric tolerance also needs investigation.
- If variability prevents a conclusion, use matched additional seeds and
  report paired differences, means, and standard deviations. Record any
  expanded run plan first. Do not retune the experiment to obtain parity.

For fixed saved predictions, require identical class counts/masking and
Dice/ASSD agreement to absolute `1e-10` on finite values, with identical NaN
locations and empty-mask outcomes. Floating-point metric discrepancies above
that threshold require diagnosis of implementation/dependency differences.

Initial weights, transformed fixed inputs, optimizer groups, LR traces, EMA
equations, and update counts are structural comparison gates. Compare tensor
contents, not hashes of differently serialized modern checkpoint files.
Logit/loss/gradient tolerance calibration is a step-8 gate: repeat the legacy
fixed-input checks, measure max absolute/relative error and prediction
disagreement, and record those tolerances before viewing modern outputs.
Completed 2026-10-11: immutable budgets and passing scoped results are in
`MIGRATION_STEP8.md` and `step8_numerics_20261011/qualification.json`.
Keep FP32, TF32 enabled for matmul and cuDNN, cuDNN benchmarking enabled,
and deterministic algorithms disabled as observed in legacy captures.

Step-8 reference correction (2026-10-11): the original frozen-checkout
MR -> CT run regenerated ETF tensors because asset lookup required ignored
`AGENTS.md`, advancing parent NumPy RNG used by Gaussian blur. Its preserved
final/best scores are superseded for accuracy acceptance. A replacement
legacy run pins the original declared assets without editing frozen source,
uses the same seed/settings and 5,000-update stop with 50,000 horizon, and
records final/best source and target results separately. CT -> MR remains
valid. The acceptance thresholds above are unchanged; await replacement
completion before step-9 modern accuracy training.

## Work that can proceed during baseline training

The contract, documentation, modern dependency pins, and isolated environment
preparation can proceed now. Keep the `epa` environment and original source,
configs, data, initialization, and prototypes available to the running queue.
The queue checks tracked legacy source hashes before launching its next run
and evaluations. Complete those jobs before editing shared legacy framework
files, or first isolate the queued baseline code and update its guarded
execution paths. Do not silently change the running or queued experiments.

Record future contract revisions on `mmcv2`, with a date and reason, before
modern training results are inspected. Do not weaken tolerances in response
to an observed modern discrepancy. Add completed legacy measurements by
checkpoint identity without changing these rules.
