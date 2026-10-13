"""Replace the isolated MR-to-CT reference using the declared ETF assets.

The frozen legacy source, original runs and environments remain untouched.
Only the asset path is pinned; training uses the frozen legacy launcher.
"""

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'work_dirs/migration_baseline'
FROZEN = BASE / 'legacy_source'
WORK = BASE / 'mr2ct_epa_fda_seed0_declared_5k'
STATUS = BASE / 'mr2ct_declared_baseline_status.json'
CONFIG = ('configs/medical/'
          'mr2ct_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py')
PYTHON = '/home/akash/miniconda3/envs/epa/bin/python'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard():
    branch = subprocess.check_output(['git', 'branch', '--show-current'],
                                     cwd=ROOT,
                                     text=True).strip()
    assert branch == 'mmcv2', 'All work and progress must stay on mmcv2'
    manifest = json.loads((BASE / 'legacy_source_manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert digest(FROZEN / name) == expected, name
    for path in (FROZEN / 'etf_prototypes').iterdir():
        assert digest(path) == digest(ROOT / 'etf_prototypes' / path.name)


def save(state):
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATUS)


def train():
    guard()
    sys.path.insert(0, str(FROZEN))
    os.chdir(FROZEN)
    import mmseg.models.uda.dacs as dacs
    import mmseg.models.utils.etf_prototypes as assets
    assert str(FROZEN) in dacs.__file__
    declared = FROZEN / 'etf_prototypes'
    assets.get_default_etf_dir = lambda: str(declared)
    dacs.get_default_etf_dir = assets.get_default_etf_dir

    def forbid_generation(*args, **kwargs):
        raise RuntimeError(
            'Declared prototypes must load; regeneration forbidden')

    dacs.generate_etf_class_prototypes = forbid_generation
    sys.argv = [
        str(FROZEN / 'tools/train_legacy_baseline.py'), '--stop-after', '5000',
        CONFIG, '--work-dir',
        str(WORK)
    ]
    runpy.run_path(sys.argv[0], run_name='__main__')
    guard()


def capture(selection):
    """Keep frozen imports/configs while resolving outputs in workspace."""
    guard()
    os.chdir(FROZEN)
    sys.path.insert(0, str(FROZEN))
    checkpoint = WORK / 'iter_5000.pth'
    if selection == 'best':
        best = list(WORK.glob('best_mDice*.pth'))
        assert len(best) == 1
        checkpoint = best[0]
    namespace = runpy.run_path(
        str(FROZEN / 'tools/capture_legacy_reference.py'))
    # Its output safety check otherwise rejects the sibling run directory.
    # Change only artifact-root resolution; imported model/data stay frozen.
    namespace['main'].__globals__['ROOT'] = ROOT
    import mmseg
    assert str(FROZEN) in mmseg.__file__
    sys.argv = [
        'capture_legacy_reference.py', CONFIG,
        str(checkpoint), '--out-dir',
        str(WORK / (selection + '_evaluation')), '--source-domain', 'mr'
    ]
    namespace['main']()
    guard()


def supervise(evaluation_only=False):
    guard()
    state = json.loads(STATUS.read_text())
    state.update(
        state='evaluating' if evaluation_only else 'training',
        supervisor_pid=os.getpid())
    state.pop('error', None)
    save(state)
    try:
        if not evaluation_only:
            with (WORK / 'console.log').open('w') as stream:
                child = subprocess.Popen(
                    [PYTHON, '-u',
                     str(Path(__file__).resolve()), '--train'],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT)
                state['training_pid'] = child.pid
                save(state)
                code = child.wait()
            assert code == 0, 'Replacement legacy training failed'
        stop = json.loads((WORK / 'early_stop.json').read_text())
        assert stop['completed_iterations'] == stop['uda_local_iter'] == 5000
        assert stop['configured_max_iters'] == 50000
        best = list(WORK.glob('best_mDice*.pth'))
        assert len(best) == 1
        final = WORK / 'iter_5000.pth'
        assert final.is_file()
        state.update(state='evaluating', early_stop=stop, evaluations=[])
        save(state)
        for selection, checkpoint in (('final', final), ('best', best[0])):
            guard()
            output = WORK / (selection + '_evaluation')
            command = [
                PYTHON, '-u',
                str(Path(__file__).resolve()), '--capture', selection
            ]
            with (WORK / (selection + '_evaluation.log')).open('w') as stream:
                child = subprocess.Popen(
                    command,
                    cwd=FROZEN,
                    stdout=stream,
                    stderr=subprocess.STDOUT)
                state['evaluation_pid'] = child.pid
                save(state)
                code = child.wait()
            assert code == 0, 'Replacement legacy evaluation failed'
            state['evaluations'].append(
                dict(
                    selection=selection,
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=digest(checkpoint),
                    summary=str(output / 'summary.json'),
                    metrics=json.loads((output / 'summary.json').read_text())))
            save(state)
        guard()
        state['state'] = 'complete'
    except Exception as error:
        state.update(state='failed', error=repr(error))
        raise
    finally:
        save(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for flag in ('launch', 'run', 'train', 'status'):
        modes.add_argument('--' + flag, action='store_true')
    modes.add_argument('--capture', choices=('final', 'best'))
    modes.add_argument('--evaluate', action='store_true')
    modes.add_argument('--finish-evaluation', action='store_true')
    args = parser.parse_args()
    if args.status:
        state = json.loads(STATUS.read_text())
        for key in ('supervisor_pid', 'training_pid', 'evaluation_pid'):
            pid = state.get(key)
            if pid:
                path = Path('/proc') / str(pid) / 'cmdline'
                state[key +
                      '_alive'] = path.exists() and bool(path.read_bytes())
        print(json.dumps(state, indent=2))
    elif args.train:
        train()
    elif args.capture:
        capture(args.capture)
    elif args.evaluate:
        supervise(evaluation_only=True)
    elif args.finish_evaluation:
        guard()
        state = json.loads(STATUS.read_text())
        assert state['state'] == 'failed'
        stop = json.loads((WORK / 'early_stop.json').read_text())
        assert stop['completed_iterations'] == stop['uda_local_iter'] == 5000
        assert stop['configured_max_iters'] == 50000
        assert not (WORK / 'final_evaluation').exists()
        assert not (WORK / 'best_evaluation').exists()
        with (WORK / 'evaluation_recovery_supervisor.log').open('w') as stream:
            child = subprocess.Popen(
                [PYTHON, '-u',
                 str(Path(__file__).resolve()), '--evaluate'],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True)
        print('Legacy evaluation recovery PID', child.pid, flush=True)
    elif args.run:
        supervise()
    else:
        guard()
        assert not STATUS.exists(
        ), 'Inspect existing replacement; do not launch twice'
        qualification = json.loads(
            (BASE / 'step8_numerics_20261011/qualification.json').read_text())
        assert qualification['passed'], 'Finish numerical checks first'
        WORK.mkdir(exist_ok=False)
        state = dict(
            branch='mmcv2',
            state='prepared',
            direction='mr2ct',
            seed=0,
            configured_max_iters=50000,
            stop_after=5000,
            source_root=str(FROZEN),
            work_dir=str(WORK),
            config=CONFIG,
            prototype_dir=str(FROZEN / 'etf_prototypes'),
            prototype_sha256={
                p.name: digest(p)
                for p in (FROZEN / 'etf_prototypes').iterdir()
            },
            replaces=str(BASE / 'mr2ct_epa_fda_seed0_5k'),
            reason='Ignored AGENTS.md broke frozen-root resolution; '
            'ETF regeneration consumed NumPy RNG used by blur.')
        save(state)
        with (WORK / 'supervisor.log').open('w') as stream:
            child = subprocess.Popen(
                [PYTHON, '-u',
                 str(Path(__file__).resolve()), '--run'],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True)
        print('Replacement legacy supervisor PID', child.pid, flush=True)


if __name__ == '__main__':
    main()
