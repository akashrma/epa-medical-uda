"""Run the qualified seed-0, 5,000-update migration comparison on mmcv2.

Prepare without CUDA; launch a durable, sequential GPU queue after the
corrected legacy MR-to-CT reference has finished its evaluations.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import runpy
import shutil
import subprocess
import sys
import sysconfig
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'work_dirs/migration_baseline'
OUT = ROOT / 'work_dirs/mmcv2'
STATUS = OUT / 'comparison_status.json'
MANIFEST = OUT / 'comparison_manifest.json'
RESULTS = ROOT / 'MIGRATION_RESULTS.md'
CAMPAIGN = 'original'
LEGACY_STATUS = BASE / 'mr2ct_declared_baseline_status.json'
QUALIFICATION = BASE / 'step8_numerics_20261011/qualification.json'
PYTHON = '/home/akash/miniconda3/envs/epa-mmcv2/bin/python'
MAIN_REVISION = '4a589d11e637c4e61442a69fe6ac8e6f57a883e7'
DIRECTIONS = ('ct2mr', 'mr2ct')
CLASSES = ('Myo', 'LAC', 'LVC', 'AA')


def campaign_options():
    return ['--campaign', CAMPAIGN] if CAMPAIGN != 'original' else []


def select_campaign(campaign):
    global CAMPAIGN, OUT, STATUS, MANIFEST, QUALIFICATION, RESULTS
    CAMPAIGN = campaign
    if campaign == 'fp32-conv':
        OUT = ROOT / 'work_dirs/mmcv2/fp32_conv_seed0_5k'
        QUALIFICATION = (
            BASE / 'step9_fp32_conv_qualification_20261012' /
            'qualification.json')
        RESULTS = ROOT / 'MIGRATION_PRECISION_RESULTS.md'
    STATUS = OUT / 'comparison_status.json'
    MANIFEST = OUT / 'comparison_manifest.json'


def config(direction):
    return ('configs/medical/' + direction +
            '_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py')


def work(direction):
    return OUT / (direction + '_epa_fda_seed0_5k')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def packages():
    # setuptools adds its bundled dependencies to sys.path after import.
    # Inventory installed distributions, independent of those runtime paths.
    paths = sorted(
        {sysconfig.get_path('purelib'),
         sysconfig.get_path('platlib')})
    return sorted((d.metadata['Name'].lower(), d.version)
                  for d in importlib.metadata.distributions(path=paths))


def branch_guard():
    branch = subprocess.check_output(['git', 'branch', '--show-current'],
                                     cwd=ROOT,
                                     text=True).strip()
    assert branch == 'mmcv2', 'All edits and progress must stay on mmcv2'
    main = subprocess.check_output(['git', 'rev-parse', 'main'],
                                   cwd=ROOT,
                                   text=True).strip()
    assert main == MAIN_REVISION, 'main changed'


def guard(supervisor_update=False):
    branch_guard()
    manifest = read(MANIFEST)
    for name, expected in manifest['sha256'].items():
        if supervisor_update and name == 'tools/run_mmcv2_comparison.py':
            continue
        assert digest(ROOT / name) == expected, 'Changed input: ' + name
    assert [list(p) for p in packages()] == manifest['packages']
    frozen = BASE / 'legacy_source'
    for name, expected in read(
            BASE / 'legacy_source_manifest.json')['sha256'].items():
        assert digest(frozen / name) == expected, 'Changed legacy: ' + name


def save(state):
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    write(STATUS, state)


def progress(message):
    branch_guard()
    if CAMPAIGN == 'fp32-conv':
        message = 'FP32-convolution candidate: ' + message
    path = ROOT / 'PLAN.md'
    content = path.read_text()
    marker = '| YYYY-MM-DD | N | Add progress here on `mmcv2` | '
    assert marker in content, 'Progress placeholder missing from PLAN.md'
    row = ('| ' + datetime.now().date().isoformat() + ' | 9 | ' + message +
           ' | Inspect `' + str(STATUS.relative_to(ROOT)) + '` |\n')
    path.write_text(content.replace(marker, row + marker, 1))


def training_command(direction, dry_run=False):
    command = [
        PYTHON, '-u', 'tools/train.py',
        config(direction), '--work-dir',
        str(work(direction)), '--stop-after', '5000'
    ]
    return command + (['--dry-run'] if dry_run else [])


def prepare():
    branch_guard()
    assert not STATUS.exists(), 'Existing comparison: inspect, do not replace'
    assert not MANIFEST.exists(), 'Existing preparation: inspect first'
    assert read(QUALIFICATION)['passed'], 'Numerical qualification failed'
    qualified = read(BASE / 'step8_numerics_20261011/manifest.json')
    permitted = set()
    previous_automation_changes = {}
    if CAMPAIGN == 'fp32-conv':
        permitted = {
            'mmseg/engine/hooks.py', 'mmseg/engine/config.py', 'tools/test.py',
            'tools/run_mmcv2_comparison.py',
            'tools/capture_migration_numerics.py',
            'tools/verify_mmcv2_numerics.py'
        }
        previous = read(ROOT / 'work_dirs/mmcv2/comparison_status.json')
        assert previous['state'] == 'awaiting_review'
        assert all(r['state'] == 'complete' for r in previous['runs'].values())
        qualification = read(QUALIFICATION)
        assert qualification['candidate'] == CAMPAIGN
        assert not qualification['backend']['cudnn_allow_tf32']
        assert qualification['backend']['matmul_allow_tf32']
        assert qualification['tolerance_sha256'] == read(
            BASE /
            'step8_numerics_20261011/qualification.json')['tolerance_sha256']
    for name, expected in qualified['source_sha256'].items():
        if name.startswith(('mmseg/', 'tools/')) and name not in permitted:
            current = digest(ROOT / name)
            if CAMPAIGN == 'fp32-conv' and \
                    name == 'tools/repair_legacy_baseline.py':
                # This output-root repair predates the precision experiment
                # and is not invoked by candidate training or evaluation.
                assert current == ('72349bcf7ac85a66c13a73f347529a321'
                                   '6388ee2e800df0ed3f036cde0efcf68')
                previous_automation_changes[name] = dict(
                    step8_sha256=expected,
                    pre_candidate_sha256=current,
                    reason='Previously completed legacy evaluation '
                    'output-root repair; not used by candidate runtime')
            else:
                assert current == expected, name
    if CAMPAIGN == 'fp32-conv':
        original = read(ROOT / 'work_dirs/mmcv2/comparison_manifest.json')
        for name, expected in original['sha256'].items():
            if name not in permitted:
                assert digest(ROOT / name) == expected, name
    for direction in DIRECTIONS:
        assert not work(direction).exists(), 'Do not overwrite training runs'
    OUT.mkdir(parents=True, exist_ok=True)
    paths = set()
    for folder in ('mmseg', 'configs/_base_', 'configs/medical'):
        paths.update(
            p for p in (ROOT / folder).rglob('*.py')
            if '__pycache__' not in p.parts)
    paths.update((ROOT / 'etf_prototypes').iterdir())
    paths.update((ROOT / 'data/datalist').glob('*.txt'))
    for name in ('tools/train.py', 'tools/test.py',
                 'tools/run_mmcv2_comparison.py',
                 'requirements-mmcv2.lock.txt', 'MIGRATION_CONTRACT.md',
                 'pretrained/mit_b5_mmseg.pth'):
        paths.add(ROOT / name)
    paths.add(QUALIFICATION)
    if CAMPAIGN == 'fp32-conv':
        for name in ('MIGRATION_PRECISION_EXPERIMENT.md',
                     'tools/capture_migration_numerics.py',
                     'tools/verify_mmcv2_numerics.py',
                     'tools/repair_legacy_baseline.py',
                     'work_dirs/mmcv2/comparison_manifest.json',
                     'work_dirs/mmcv2/accuracy_comparison.json',
                     'work_dirs/mmcv2/fp32_conv_transition/before_change.json',
                     'work_dirs/migration_baseline/step8_numerics_20261011/'
                     'tolerances.json'):
            paths.add(ROOT / name)
    audit = read(BASE / 'step1_audit_20261011/data_audit.json')
    for item in audit['lists'].values():
        assert digest(ROOT / item['path']) == item['sha256']
    write(
        MANIFEST,
        dict(
            branch='mmcv2',
            campaign=CAMPAIGN,
            precision_policy=read(QUALIFICATION).get('backend'),
            permitted_source_changes=sorted(permitted),
            previous_automation_changes=previous_automation_changes,
            main_revision=MAIN_REVISION,
            created_at=datetime.now(timezone.utc).isoformat(),
            python=sys.version,
            packages=packages(),
            qualification=str(QUALIFICATION),
            tolerance_sha256=read(QUALIFICATION)['tolerance_sha256'],
            data_audit=str(BASE / 'step1_audit_20261011/data_audit.json'),
            sha256={
                str(p.relative_to(ROOT)): digest(p)
                for p in sorted(paths)
            },
            commands={d: training_command(d)
                      for d in DIRECTIONS},
            target_mDice_absolute_limit=0.01,
            class_dice_investigation_limit=0.02))
    guard()
    for direction in DIRECTIONS:
        subprocess.run(
            training_command(direction, dry_run=True), cwd=ROOT, check=True)
        launch = read(work(direction) / 'launch_manifest.json')
        assert launch['seed'] == 0 and launch['gpu_ids'] == [0]
        assert launch['max_iters'] == 50000 and launch['stop_after'] == 5000
        assert not launch['resume'] and launch['load_from'] is None
    state = dict(
        branch='mmcv2',
        campaign=CAMPAIGN,
        state='prepared',
        seed=0,
        configured_max_iters=50000,
        stop_after=5000,
        manifest=str(MANIFEST),
        runs={
            d:
            dict(
                state='queued',
                config=config(d),
                work_dir=str(work(d)),
                evaluations={})
            for d in DIRECTIONS
        })
    save(state)
    print('Prepared both matched runs; no CUDA training started.', flush=True)


def train(direction):
    guard()
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import torch
    from mmseg.engine.hooks import backend_flags
    assert torch.cuda.is_available(), 'Host CUDA required'
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    sys.argv = training_command(direction)[2:]
    runpy.run_path(str(ROOT / 'tools/train.py'), run_name='__main__')
    if CAMPAIGN == 'fp32-conv':
        assert backend_flags() == read(QUALIFICATION)['backend']
    torch.cuda.synchronize()
    console = work(direction) / 'console.log'
    logged_memory = re.findall(r'\bmemory:\s*(\d+)', console.read_text()) \
        if console.exists() else []
    write(
        work(direction) / 'runtime_summary.json',
        dict(
            elapsed_seconds=time.monotonic() - started,
            includes='Initialization, training, scheduled validation '
            'and saves',
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_byte_counter_scope='Since last MMEngine logger reset',
            whole_run_peak_gpu_allocated_mib_logged=max(
                (int(value) for value in logged_memory), default=None),
            logged_memory_precision='MiB truncated by MMEngine logger',
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            cuda_runtime=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            precision='FP32',
            backend=backend_flags(),
            seed=0,
            configured_max_iters=50000,
            stop_after=5000))
    guard()


def run_child(command, logfile, state, run, phase):
    with Path(logfile).open('w') as stream:
        child = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        run.update(
            state=phase,
            child_pid=child.pid,
            command=command,
            console=str(logfile))
        save(state)
        while child.poll() is None:
            time.sleep(10)
            if phase == 'training':
                text = Path(logfile).read_text(errors='replace')[-100000:]
                matches = re.findall(r'Iter\(train\)\s*\[\s*(\d+)/\s*50000\]',
                                     text)
                if matches:
                    run['last_logged_iteration'] = int(matches[-1])
            save(state)
        assert child.returncode == 0, phase + ' failed; see ' + str(logfile)


def legacy_reference(direction, selection):
    name = ('ct2mr_epa_fda_seed0_5k'
            if direction == 'ct2mr' else 'mr2ct_epa_fda_seed0_declared_5k')
    path = BASE / name / (selection + '_evaluation/summary.json')
    summary = read(path)
    assert digest(summary['checkpoint']) == summary['checkpoint_sha256']
    return summary


def evaluate(direction, checkpoint, selection, state, run):
    guard()
    output = work(direction) / (selection + '_evaluation')
    output.mkdir(exist_ok=False)
    summary = dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint),
        selection=selection,
        config=config(direction),
        seed=0)
    for domain in ('target', 'source'):
        command = [
            PYTHON, '-u', 'tools/test.py',
            config(direction),
            str(checkpoint), '--eval', 'mDice', '--metrics-out',
            str(output / (domain + '_metrics.json')), '--out',
            str(output / (domain + '_predictions.pkl'))
        ]
        if domain == 'source':
            source = 'ct' if direction == 'ct2mr' else 'mr'
            command += [
                '--options',
                'data.test.img_list=data/datalist/val_' + source + '.txt',
                'data.test.ann_list=data/datalist/val_' + source + '_gt.txt'
            ]
        run.update(selection=selection, evaluation_domain=domain)
        run_child(command, output / (domain + '.log'), state, run,
                  'evaluating')
        metrics = read(output / (domain + '_metrics.json'))
        assert set(metrics) == {'mDice', 'mASSD'} | {
            prefix + '.' + name
            for name in CLASSES
            for prefix in ('Dice', 'ASSD')
        }
        assert all(math.isfinite(v) for v in metrics.values())
        summary[domain] = metrics
    write(output / 'summary.json', summary)
    run['evaluations'][selection] = summary
    save(state)


def compare(state):
    guard()
    manifest = read(MANIFEST)
    rows = []
    for direction in DIRECTIONS:
        for selection in ('final', 'best'):
            legacy = legacy_reference(direction, selection)
            modern = state['runs'][direction]['evaluations'][selection]
            target = modern['target']
            reference = legacy['reproduced_target']
            deltas = {key: target[key] - reference[key] for key in target}
            class_flags = [
                name for name in CLASSES if abs(deltas['Dice.' + name]) >
                manifest['class_dice_investigation_limit']
            ]
            accepted = abs(deltas['mDice']) <= \
                manifest['target_mDice_absolute_limit'] and not class_flags
            rows.append(
                dict(
                    direction=direction,
                    selection=selection,
                    legacy=legacy,
                    modern=modern,
                    target_deltas=deltas,
                    source_deltas={
                        key:
                        modern['source'][key] -
                        legacy['diagnostic_source'][key]
                        for key in modern['source']
                    },
                    target_mDice_within_limit=abs(deltas['mDice'])
                    <= manifest['target_mDice_absolute_limit'],
                    class_investigations=class_flags,
                    numerical_accuracy_gate_passed=accepted))
    result = dict(
        manifest=str(MANIFEST),
        campaign=CAMPAIGN,
        precision_policy=manifest.get('precision_policy'),
        rows=rows,
        target_and_class_gates_passed=all(row['numerical_accuracy_gate_passed']
                                          for row in rows),
        source_assd_review_required=True,
        migration_complete=False,
        note='Review source/ASSD diagnostics and documentation; '
        'investigate any failed gate before accepting migration.')
    write(OUT / 'accuracy_comparison.json', result)
    lines = [
        '# MMCV 2 ' + ('FP32-convolution candidate' if CAMPAIGN == 'fp32-conv'
                       else 'migration') + ' accuracy results', '',
        'All runs, edits and progress belong to `mmcv2`. Seed 0; '
        '5,000 updates; original 50,000-iteration horizon.', '',
        '| Direction | Selection | Legacy target mDice | '
        'Modern target mDice | Delta | Modern source mDice | '
        'Modern target ASSD | Accuracy/class gate |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |'
    ]
    for row in rows:
        modern = row['modern']
        lines.append(f'| {row["direction"]} | {row["selection"]} | '
                     f'{row["legacy"]["reproduced_target"]["mDice"]:.10f} | '
                     f'{modern["target"]["mDice"]:.10f} | '
                     f'{row["target_deltas"]["mDice"]:+.10f} | '
                     f'{modern["source"]["mDice"]:.10f} | '
                     f'{modern["target"]["mASSD"]:.10f} | '
                     f'{row["numerical_accuracy_gate_passed"]} |')
    if CAMPAIGN == 'fp32-conv':
        original = read(ROOT / 'work_dirs/mmcv2/accuracy_comparison.json')
        previous = {
            (row['direction'], row['selection']): row
            for row in original['rows']
        }
        lines += [
            '', 'Convolution TF32 disabled for candidate training and '
            'evaluation; matmul TF32 and benchmark/nondeterminism retained.',
            '', '| Direction | Selection | Original modern target mDice | '
            'Candidate target mDice | Candidate minus original modern |',
            '| --- | --- | --- | --- | --- |'
        ]
        for row in rows:
            old = previous[(row['direction'], row['selection'])]['modern']
            row['original_modern'] = old
            delta = row['modern']['target']['mDice'] - old['target']['mDice']
            row['target_mDice_delta_from_original_modern'] = delta
            lines.append(
                f'| {row["direction"]} | {row["selection"]} | '
                f'{old["target"]["mDice"]:.10f} | '
                f'{row["modern"]["target"]["mDice"]:.10f} | {delta:+.10f} |')
        write(OUT / 'accuracy_comparison.json', result)
    lines += [
        '', 'Full per-class/source/target Dice and ASSD, checkpoint '
        'hashes and deltas: `' + str(
            (OUT / 'accuracy_comparison.json').relative_to(ROOT)) + '`.', '',
        'Runtime and peak allocated/reserved GPU memory are in '
        '`runtime_summary.json` under each run directory.', '',
        'The migration remains pending review of source/ASSD '
        'diagnostics, any failed accuracy/class gate, and final '
        'reproduction/support documentation. No hyperparameters were '
        'retuned for this comparison.', ''
    ]
    branch_guard()
    RESULTS.write_text('\n'.join(lines))
    return result


def supervise():
    guard()
    state = read(STATUS)
    assert state['state'] == 'prepared', 'Queue already started'
    state.update(
        state='waiting_for_legacy_reference', supervisor_pid=os.getpid())
    save(state)
    try:
        while True:
            legacy = read(LEGACY_STATUS)
            assert legacy['state'] != 'failed', legacy.get('error')
            if legacy['state'] == 'complete':
                break
            pid = legacy['supervisor_pid']
            command = Path('/proc') / str(pid) / 'cmdline'
            assert command.exists() and command.read_bytes(), \
                'Legacy supervisor stopped before completion; inspect it'
            state['legacy_state'] = legacy['state']
            save(state)
            time.sleep(10)
        for direction in DIRECTIONS:
            for selection in ('final', 'best'):
                legacy_reference(direction, selection)
        progress('Corrected legacy MR -> CT final/best source/target '
                 'evaluations completed; modern sequential queue released')
        for direction in DIRECTIONS:
            guard()
            state['active_direction'] = direction
            run = state['runs'][direction]
            if run['state'] == 'training_complete':
                assert digest(work(direction) / 'iter_5000.pth') == \
                    run['recovered_final_checkpoint_sha256']
                progress('Recovered completed ' + direction +
                         ' at 5,000; evaluate saved checkpoints without '
                         'retraining or resuming')
            else:
                assert run['state'] == 'queued'
                assert not list(work(direction).glob('*.pth')), \
                    'Existing checkpoint: do not overwrite or silently resume'
                state['state'] = 'training'
                progress('Started modern ' + direction +
                         ' EPA/FDA from seed 0, original initialization; '
                         'stop 5,000, horizon 50,000')
                command = [
                    PYTHON, '-u',
                    str(Path(__file__).resolve()), '--train', direction
                ] + campaign_options()
                run_child(command,
                          work(direction) / 'console.log', state, run,
                          'training')
            guard()
            stop = read(work(direction) / 'early_stop.json')
            assert stop['completed_iterations'] == 5000
            assert stop['uda_local_iter'] == 5000
            assert stop['configured_max_iters'] == 50000
            run['early_stop'] = stop
            run['runtime'] = read(work(direction) / 'runtime_summary.json')
            best = list(work(direction).glob('best_target_mDice*.pth'))
            assert len(best) == 1, 'Exactly one best checkpoint required'
            state['state'] = 'evaluating'
            for selection, checkpoint in (('final',
                                           work(direction) / 'iter_5000.pth'),
                                          ('best', best[0])):
                evaluate(direction, checkpoint, selection, state, run)
            run['state'] = 'complete'
            save(state)
            progress('Modern ' + direction + ' stopped exactly at 5,000; '
                     'full final/best source/target evaluations completed')
        result = compare(state)
        state.update(
            state='awaiting_review',
            target_and_class_gates_passed=result[
                'target_and_class_gates_passed'],
            comparison=str(OUT / 'accuracy_comparison.json'))
        progress('Both modern runs/evaluations finished; accuracy comparison '
                 'written; target/class gates passed=' +
                 str(result['target_and_class_gates_passed']) +
                 '; source/ASSD and final documentation review pending')
    except Exception as error:
        state.update(
            state='failed',
            error=repr(error),
            traceback=traceback.format_exc())
        progress('Comparison queue failed: ' + type(error).__name__ +
                 '; see supervisor.log and comparison_status.json')
        raise
    finally:
        save(state)


def launch():
    guard()
    assert read(STATUS)['state'] == 'prepared', 'Do not launch twice'
    with (OUT / 'launch.lock').open('x') as lock:
        lock.write(str(os.getpid()) + '\n')
    with (OUT / 'supervisor.log').open('w') as stream:
        child = subprocess.Popen(
            [PYTHON, '-u',
             str(Path(__file__).resolve()), '--run'] + campaign_options(),
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True)
    print('Modern comparison supervisor PID', child.pid, flush=True)


def recover():
    """Recover the inventory failure without repeating training updates."""
    guard(supervisor_update=True)
    state = read(STATUS)
    assert state['state'] == 'failed'
    assert state['runs']['ct2mr']['last_logged_iteration'] == 5000
    assert state['runs']['mr2ct']['state'] == 'queued'
    assert not list(work('mr2ct').glob('*.pth'))
    assert not state['runs']['ct2mr']['evaluations']
    assert not (work('ct2mr') / 'final_evaluation').exists()
    assert not (work('ct2mr') / 'best_evaluation').exists()
    supervisor = Path('/proc') / str(state['supervisor_pid']) / 'cmdline'
    assert not supervisor.exists() or not supervisor.read_bytes()
    stop = read(work('ct2mr') / 'early_stop.json')
    assert stop['completed_iterations'] == stop['uda_local_iter'] == 5000
    assert stop['configured_max_iters'] == 50000
    runtime = read(work('ct2mr') / 'runtime_summary.json')
    assert runtime['seed'] == 0 and runtime['stop_after'] == 5000
    assert runtime['configured_max_iters'] == 50000
    best = list(work('ct2mr').glob('best_target_mDice*.pth'))
    assert len(best) == 1
    final = work('ct2mr') / 'iter_5000.pth'
    assert final.stat().st_size > 0
    archive = OUT / 'package_guard_recovery'
    archive.mkdir(exist_ok=False)
    for path in (MANIFEST, STATUS, OUT / 'supervisor.log',
                 OUT / 'launch.lock'):
        shutil.copyfile(path, archive / path.name)
    manifest = read(MANIFEST)
    manifest['automation_recovery'] = dict(
        archive=str(archive),
        reason='setuptools vendored metadata was '
        'counted as installed packages after training',
        previous_supervisor_sha256=manifest['sha256']
        ['tools/run_mmcv2_comparison.py'],
        current_supervisor_sha256=digest(Path(__file__).resolve()),
        final_checkpoint_sha256=digest(final),
        best_checkpoint_sha256=digest(best[0]),
        runtime_and_training_inputs_unchanged=True)
    manifest['sha256']['tools/run_mmcv2_comparison.py'] = \
        digest(Path(__file__).resolve())
    write(MANIFEST, manifest)
    guard()
    state['runs']['ct2mr'].update(
        state='training_complete',
        recovered_final_checkpoint_sha256=digest(final),
        early_stop=stop,
        runtime=runtime)
    for key in ('error', 'traceback', 'supervisor_pid'):
        state.pop(key, None)
    state.update(
        state='prepared',
        legacy_state='complete',
        package_guard_recovery=str(archive))
    save(state)
    (OUT / 'supervisor.log').rename(archive / 'supervisor_original.log')
    (OUT / 'launch.lock').rename(archive / 'launch_original.lock')
    progress('Fixed post-training installed-package inventory: 78 pins '
             'unchanged; archived original manifest/status/log; CT -> MR '
             '5,000 checkpoint retained for evaluation; MR -> CT queued')
    launch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--campaign', choices=('original', 'fp32-conv'), default='original')
    modes = parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare', 'launch', 'run', 'status', 'compare', 'recover'):
        modes.add_argument('--' + flag, action='store_true')
    modes.add_argument('--train', choices=DIRECTIONS)
    args = parser.parse_args()
    select_campaign(args.campaign)
    assert not args.recover or CAMPAIGN == 'original', (
        'Package-guard recovery applies only to the original campaign')
    if args.prepare:
        prepare()
    elif args.train:
        train(args.train)
    elif args.run:
        supervise()
    elif args.recover:
        recover()
    elif args.compare:
        print(json.dumps(compare(read(STATUS)), indent=2))
    elif args.status:
        state = read(STATUS)
        for pid in ('supervisor_pid', ):
            if pid in state:
                path = Path('/proc') / str(state[pid]) / 'cmdline'
                state[pid +
                      '_alive'] = path.exists() and bool(path.read_bytes())
        print(json.dumps(state, indent=2))
    else:
        launch()


if __name__ == '__main__':
    main()
