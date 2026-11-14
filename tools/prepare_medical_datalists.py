"""Create the ordered slice and optional test-volume lists for EPA."""

import argparse
import os
from pathlib import Path


def prepare_datalists(data_root, output_dir, check=False, include_test=False):
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    counts = {
        ('ct', 'train'): 8400,
        ('ct', 'val'): 1200,
        ('mr', 'train'): 9600,
        ('mr', 'val'): 2400
    }
    contents = {}
    missing = []
    for (modality, split), count in counts.items():
        for labels in (False, True):
            suffix = '_gt' if labels else ''
            folder = f'gt_{split}_{modality}' if labels else \
                f'{split}_{modality}'
            paths = [
                data_root / folder / f'{modality}_{split}_slice{i}{suffix}.npy'
                for i in range(count)
            ]
            missing.extend(path for path in paths if not path.is_file())
            filename = output_dir / f'{split}_{modality}{suffix}.txt'
            contents[filename] = ''.join(
                os.path.relpath(path, output_dir) + '\n' for path in paths)
    test_count = 0
    if include_test:
        test_ids = {
            'ct': (1003, 1008, 1014, 1019),
            'mr': (1007, 1009, 1018, 1019)
        }
        for modality, ids in test_ids.items():
            paths = [
                data_root / f'test_{modality}' /
                f'image_{modality}_{case}.nii.gz.npz' for case in ids
            ]
            missing.extend(path for path in paths if not path.is_file())
            contents[output_dir / f'test_{modality}.txt'] = ''.join(
                os.path.relpath(path, output_dir) + '\n' for path in paths)
            test_count += len(paths)
    if missing:
        examples = '\n'.join(str(path) for path in missing[:5])
        raise ValueError(f'Missing {len(missing)} data files under '
                         f'{data_root}. Examples:\n{examples}')

    # Validate every destination before writing any lists. Preserve custom lists.
    for filename, content in contents.items():
        if filename.exists():
            if filename.read_text() != content:
                raise ValueError(
                    f'{filename} differs from the recorded split; '
                    'choose a new --output-dir to preserve it')
        elif check:
            raise ValueError(f'Missing datalist: {filename}')
    if not check:
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in contents.items():
            if not filename.exists():
                filename.write_text(content)
    print(f'Checked {len(contents)} lists and '
          f'{sum(counts.values())} image/label pairs and '
          f'{test_count} test volumes: {output_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/data_np')
    parser.add_argument('--output-dir', default='data/datalist')
    parser.add_argument(
        '--include-test',
        action='store_true',
        help='Also prepare the supplied CT/MR test-volume lists')
    parser.add_argument(
        '--check',
        action='store_true',
        help='Verify existing lists and file paths without writing')
    args = parser.parse_args()
    try:
        prepare_datalists(args.data_root, args.output_dir, args.check,
                          args.include_test)
    except ValueError as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
