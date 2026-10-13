# MMCV 2 migration accuracy results

Current status: migration accepted by the user on 2026-10-13; see
[MIGRATION_ACCEPTANCE.md](MIGRATION_ACCEPTANCE.md). Best target checkpoints
only; source/final results excluded; measured MR -> CT differences accepted.
No further accuracy runs are required. Execution/status notes below are
historical campaign evidence and retain their original automatic outcomes.


All runs, edits and progress belong to `mmcv2`. Seed 0; 5,000 updates; original 50,000-iteration horizon.

| Direction | Selection | Legacy target mDice | Modern target mDice | Delta | Modern source mDice | Modern target ASSD | Accuracy/class gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ct2mr | final | 0.6857276685 | 0.6251673327 | -0.0605603358 | 0.8678568804 | 2.4923604681 | False |
| ct2mr | best | 0.6857276685 | 0.6750950841 | -0.0106325844 | 0.7765594884 | 2.2462948457 | False |
| mr2ct | final | 0.8027866677 | 0.7928191797 | -0.0099674880 | 0.7833636407 | 1.7081418482 | False |
| mr2ct | best | 0.8432316865 | 0.7942362434 | -0.0489954432 | 0.8298503640 | 1.6805179639 | False |

Full per-class/source/target Dice and ASSD, checkpoint hashes and deltas: `work_dirs/mmcv2/accuracy_comparison.json`.

Runtime and peak allocated/reserved GPU memory are in `runtime_summary.json` under each run directory.

The migration remains pending review of source/ASSD diagnostics, any failed accuracy/class gate, and final reproduction/support documentation. No hyperparameters were retuned for this comparison.
