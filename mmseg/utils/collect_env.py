# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add code archive generation

import os
import tarfile

from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

import mmseg


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMSegmentation'] = f'{mmseg.__version__}+{get_git_hash()[:7]}'

    return env_info


def is_source_file(x):
    excluded = {
        'data', 'pretrained', 'work_dirs', 'configs/generated', '.git',
        '.codex', '.agents', '__pycache__', '.mim', 'jobs'
    }
    parts = x.name.removeprefix('./').split('/')
    generated = '/'.join(parts[:2]) == 'configs/generated'
    if any(part in excluded for part in parts) or generated or x.issym():
        return None
    suffixes = ('.py', '.sh', '.yml', '.json', '.txt', '.md', '.cfg')
    if x.isdir() or x.name.endswith(suffixes):
        return x
    else:
        return None


def gen_code_archive(out_dir, file='code.tar.gz'):
    archive = os.path.join(out_dir, file)
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    with tarfile.open(archive, mode='w:gz') as tar:
        tar.add('.', filter=is_source_file)
    return archive


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
