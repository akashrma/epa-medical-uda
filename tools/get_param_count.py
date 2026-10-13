# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# isort:skip_file

from mmengine.config import Config

from experiments import generate_experiment_cfgs
from mmseg.engine.config import student_config
from mmseg.models import build_segmentor


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Trainable Params: {human_format(count)}')
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description='Count student parameters')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config')
    group.add_argument('--exp', type=int, default=None)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix='epa-params-') as temporary:
        if args.config:
            configs = [Config.fromfile(args.config)]
        else:
            configs = []
            for i, spec in enumerate(
                    generate_experiment_cfgs(args.exp or 100)):
                spec['_base_'] = [
                    str(ROOT / 'configs' / base) for base in spec['_base_']
                ]
                filename = Path(temporary) / f'{i}.py'
                Config(spec).dump(str(filename))
                configs.append(Config.fromfile(str(filename)))
        for cfg in configs:
            model_cfg = student_config(cfg)
            model_cfg['pretrained'] = None
            model_cfg['train_cfg'] = None
            model = build_segmentor(model_cfg)
            print('Student:')
            count_parameters(model)
            print('Encoder:')
            count_parameters(model.backbone)
            print('Decoder:')
            count_parameters(model.decode_head)


if __name__ == '__main__':
    main()
