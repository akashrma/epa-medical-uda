"""Freeze baseline source and hand off its queue while training continues."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'work_dirs/migration_baseline'
AUDIT = BASE / 'step1_audit_20261011'
FROZEN = BASE / 'legacy_source'
STATUS = BASE / 'selected_baselines_status.json'
MANIFEST = BASE / 'legacy_source_manifest.json'
PYTHON = '/home/akash/miniconda3/envs/epa/bin/python'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def branch_guard():
    branch = subprocess.check_output(['git', 'branch', '--show-current'],
                                     cwd=ROOT,
                                     text=True).strip()
    if branch != 'mmcv2':
        raise RuntimeError('All migration work must occur on mmcv2')


def frozen_guard():
    branch_guard()
    for name, expected in json.loads(MANIFEST.read_text())['sha256'].items():
        if sha256(FROZEN / name) != expected:
            raise RuntimeError('Frozen legacy source changed: ' + name)


def save(state):
    state['updated_at'] = now()
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    temporary.replace(STATUS)


def freeze():
    branch_guard()
    original = json.loads((AUDIT / 'source_files.json').read_text())
    tracked = subprocess.check_output(['git', 'ls-files', '-z'],
                                      cwd=ROOT).decode().split('\0')
    names = [name for name in tracked if name]
    for name in names:
        if sha256(ROOT / name) != original[name]:
            raise RuntimeError('Freeze before editing original source: ' +
                               name)
    names += [
        'tools/train_legacy_baseline.py', 'tools/capture_legacy_reference.py',
        'tools/audit_legacy_baseline.py'
    ]
    names += [
        str(p.relative_to(ROOT)) for p in (ROOT / 'etf_prototypes').glob('*')
        if p.is_file()
    ]
    FROZEN.mkdir(exist_ok=False)
    hashes = {}
    for name in sorted(set(names)):
        destination = FROZEN / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(ROOT / name, destination)
        hashes[name] = sha256(destination)
        if hashes[name] != sha256(ROOT / name):
            raise RuntimeError('Copy mismatch: ' + name)
    for name in ('data', 'pretrained', '.git'):
        (FROZEN / name).symlink_to(ROOT / name, target_is_directory=True)
    (FROZEN / 'work_dirs').mkdir()
    MANIFEST.write_text(
        json.dumps(
            dict(
                created_at=now(),
                branch='mmcv2',
                source_root=str(FROZEN),
                source_revision=subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                sha256=hashes,
                shared_paths=['data', 'pretrained', '.git'],
                note=('Current trainer retains loaded modules; '
                      'future jobs use frozen source')),
            indent=2) + '\n')
    frozen_guard()
    print('Frozen and verified {} files at {}'.format(len(hashes), FROZEN))


def process_alive(pid):
    try:
        fields = Path('/proc/{}/stat'.format(pid)).read_text().split(') ')[1]
        return fields.split()[0] != 'Z'
    except FileNotFoundError:
        return False


def handoff():
    frozen_guard()
    state = json.loads(STATUS.read_text())
    if state['state'] != 'running' or state.get('legacy_source_root'):
        raise RuntimeError('Expected original running supervisor')
    active = [r for r in state['runs'] if r['state'] == 'training']
    if len(active) != 1 or any(r['evaluations'] for r in state['runs']):
        raise RuntimeError(
            'Handoff requires one active trainer before evaluation')
    supervisor = state['supervisor_pid']
    cmdline = Path('/proc/{}/cmdline'.format(supervisor)).read_bytes()
    if b'run_selected_baselines.py' not in cmdline:
        raise RuntimeError('Supervisor PID does not match')
    trainer = active[0]['pid']
    trainer_cmd = Path('/proc/{}/cmdline'.format(trainer)).read_bytes()
    if b'train_legacy_baseline.py' not in trainer_cmd:
        raise RuntimeError('Trainer PID does not match')
    (BASE / 'selected_baselines_status_before_isolation.json'
     ).write_text(json.dumps(state, indent=2) + '\n')
    # Stop only the waiting supervisor. Never signal its process group/trainer.
    os.kill(supervisor, signal.SIGTERM)
    for _ in range(100):
        if not process_alive(supervisor):
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Original supervisor did not stop')
    if not process_alive(trainer):
        raise RuntimeError(
            'Trainer exited during handoff; inspect stop evidence')
    state.update(
        legacy_source_root=str(FROZEN),
        previous_supervisor_pid=supervisor,
        isolated_at=now())
    save(state)
    with (BASE / 'isolated_supervisor.log').open('w') as console:
        process = subprocess.Popen(
            [PYTHON, '-u',
             str(Path(__file__).resolve()), '--supervise'],
            cwd=ROOT,
            stdout=console,
            stderr=subprocess.STDOUT,
            start_new_session=True)
    print(
        'Handoff launched PID {}; trainer {} continues'.format(
            process.pid, trainer),
        flush=True)


def supervise():
    frozen_guard()
    state = json.loads(STATUS.read_text())
    state['supervisor_pid'] = os.getpid()
    save(state)
    try:
        for run in state['runs']:
            if run['state'] == 'complete':
                continue
            frozen_guard()
            work = ROOT / run['work_dir']
            if run['state'] == 'training':
                # Preserve the ongoing trajectory, RNG, EMA and loaded source.
                while process_alive(run['pid']):
                    time.sleep(5)
                run['train_returncode'] = None
                run['exit_verified_by'] = (
                    'Completed stop evidence and checkpoints')
            elif run['state'] == 'queued':
                command = [
                    PYTHON, '-u',
                    str(FROZEN / 'tools/train_legacy_baseline.py'),
                    '--stop-after',
                    str(run['stop_after']), run['config'], '--work-dir',
                    str(work)
                ]
                run.update(command=command, state='training', started_at=now())
                with (ROOT / run['console_log']).open('w') as console:
                    process = subprocess.Popen(
                        command,
                        cwd=FROZEN,
                        stdout=console,
                        stderr=subprocess.STDOUT)
                    run['pid'] = process.pid
                    save(state)
                    run['train_returncode'] = process.wait()
                if run['train_returncode']:
                    raise RuntimeError('Legacy training failed: ' +
                                       run['direction'])
            else:
                raise RuntimeError('Unexpected run state: ' + run['state'])
            run['training_finished_at'] = now()
            stop = json.loads((work / 'early_stop.json').read_text())
            if (stop['completed_iterations'] != 5000
                    or stop['configured_max_iters'] != 50000
                    or stop['uda_local_iter'] != 5000):
                raise RuntimeError('Baseline stop protocol mismatch')
            final = work / 'iter_5000.pth'
            best = list(work.glob('best_mDice*.pth'))
            if not final.is_file() or len(best) != 1:
                raise RuntimeError('Missing final/best checkpoint')
            run.update(
                state='evaluating',
                early_stop=stop,
                final_checkpoint=str(final.relative_to(ROOT)),
                best_checkpoint=str(best[0].relative_to(ROOT)))
            save(state)
            for selection, checkpoint in [('final', final), ('best', best[0])]:
                frozen_guard()
                relative = Path('work_dirs') / work.name / (
                    selection + '_evaluation')
                out = FROZEN / relative
                command = [
                    PYTHON, '-u',
                    str(FROZEN / 'tools/capture_legacy_reference.py'),
                    run['config'],
                    str(checkpoint), '--out-dir',
                    str(relative), '--source-domain',
                    'ct' if run['direction'] == 'ct2mr' else 'mr'
                ]
                with (work /
                      (selection + '_evaluation.log')).open('w') as console:
                    process = subprocess.Popen(
                        command,
                        cwd=FROZEN,
                        stdout=console,
                        stderr=subprocess.STDOUT)
                    run['evaluation_pid'] = process.pid
                    save(state)
                    code = process.wait()
                result = dict(
                    selection=selection,
                    command=command,
                    returncode=code,
                    summary=str((out / 'summary.json').relative_to(ROOT)))
                if code == 0:
                    result['metrics'] = json.loads(
                        (out / 'summary.json').read_text())
                    (work / (selection + '_evaluation')).symlink_to(
                        out, target_is_directory=True)
                run['evaluations'].append(result)
                save(state)
                if code:
                    raise RuntimeError('Legacy checkpoint evaluation failed')
            run.update(state='complete', finished_at=now())
            save(state)
        state['state'] = 'complete'
    except Exception as error:
        state.update(state='failed', error=repr(error))
        raise
    finally:
        state['finished_at'] = now()
        save(state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=['freeze', 'handoff'])
    parser.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    if args.supervise:
        supervise()
    elif args.action == 'freeze':
        freeze()
    elif args.action == 'handoff':
        handoff()
    else:
        parser.error('Choose freeze or handoff')
