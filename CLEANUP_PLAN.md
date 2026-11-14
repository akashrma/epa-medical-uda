# Repository cleanup

Implemented on `mmcv2-cleanup` after user approval. The accepted migration
and its full historical reports remain on `mmcv2` at `a05cff5`; `main`
remains unchanged.

## Changes

- Removed 15 migration-only utilities, 13 completed migration-history
  documents and the unused evaluation-loop re-export shim.
- Consolidated setup, training/evaluation/resume and accepted results into
  [ENVIRONMENT.md](ENVIRONMENT.md), [TRAINING.md](TRAINING.md) and
  [TRAINING_RESULTS.md](TRAINING_RESULTS.md); updated README and runtime links.
- Replaced custom Torch 1.7 DropPath rounding with FP32 `torch.linspace` in
  backbone construction and reset.
- Removed obsolete FFT/real-imag fallback branches while retaining FDA's
  real inverse transform and frequency mutation.
- Preserved dependency pins, experiment configs, training cap/horizon,
  precision, optimizer/sampler/LR behavior, medical metrics and checkpoint
  semantics. Existing data, weights and ignored run artifacts are retained.

## Verification

- Public training dry runs for both selected EPA/FDA configs: full resolved
  configs match before/after after normalizing output paths. The horizon
  remains 50,000 and the default update cap remains 5,000.
- Train/test/experiment-launcher CLI help and inference imports pass.
- Both MiT-B5 models initialize from pretrained weights and strictly load
  their existing best target checkpoints (CT -> MR iteration 3,500;
  MR -> CT iteration 3,000). Both produce finite 1x5x256x256 CPU logits.
- DropPath construction/reset probabilities match ordinary FP32 linspace.
- Four CPU FDA cases, including odd dimensions and the configured 256x256,
  L=0.01 case, match the pre-cleanup outputs and input gradients exactly.
- One CPU source-segmentation AdamW update per direction passes with seed 0,
  batch size 1 and finite losses/gradients, starting from those best weights.
  This checks the source model; no full UDA or accuracy campaign was rerun.
- All 232 remaining Python files pass AST parsing. Changed Python files
  pass available YAPF formatting checks; whitespace, dependency-pin,
  deletion-count, removed-reference and local documentation-link checks pass.
- Full pre-commit was attempted from an isolated temporary environment.
  Sandbox DNS blocked its first hook fetch; the host retry stalled fetching
  the configured GitLab Flake8 repository and was interrupted. The full
  hook suite has not completed. Hook configuration is unchanged.

Local verification code, JSON and logs are saved under the ignored
`work_dirs/mmcv2_cleanup/runtime_simplifications/` directory. Full source
and target validation metrics remain the historical measurements in
[TRAINING_RESULTS.md](TRAINING_RESULTS.md). They predate the DropPath
rounding simplification; no new accuracy result is claimed for this cleanup.
