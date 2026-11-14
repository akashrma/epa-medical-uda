# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Also save config as json file

import argparse
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# isort:skip_file

from mmengine.config import Config, DictAction

from mmseg.engine.config import student_config, build_runtime_config
from mmseg.models import build_segmentor


def main(argv=None):
    parser = argparse.ArgumentParser(description='Print the inherited config')
    parser.add_argument('config')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--out', help='Save config as a Python file')
    parser.add_argument(
        '--options', '--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args(argv)
    cfg = Config.fromfile(args.config)
    if args.options:
        cfg.merge_from_dict(args.options)
    if args.native:
        cfg = build_runtime_config(cfg)
    print(cfg.pretty_text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cfg.dump(args.out)
    if args.graph:
        model_cfg = student_config(cfg)
        model_cfg['pretrained'] = None
        model_cfg['train_cfg'] = None
        print(build_segmentor(model_cfg))
    return cfg


if __name__ == '__main__':
    main()
