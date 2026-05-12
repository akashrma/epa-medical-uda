#!/usr/bin/env python3
"""Convert a HuggingFace SegFormer checkpoint to mmseg MiT backbone format.

This script maps HF-style keys (segformer.encoder.*) to the mmseg MiT
backbone keys expected by mix_transformer.py (patch_embed*/block*/norm*).
The output checkpoint can be used as model.pretrained for SegFormer backbones.
"""

import argparse
import os
import re

import torch


def _load_state_dict(path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            return ckpt['state_dict']
        if 'model' in ckpt:
            return ckpt['model']
    return ckpt


def _map_hf_to_mmseg(state):
    patch_re = re.compile(
        r'^segformer\.encoder\.patch_embeddings\.(\d+)\.'
        r'(proj|layer_norm)\.(weight|bias)$')
    layer_norm_re = re.compile(
        r'^segformer\.encoder\.layer_norm\.(\d+)\.(weight|bias)$')
    block_re = re.compile(
        r'^segformer\.encoder\.block\.(\d+)\.(\d+)\.(.+)$')

    new_state = {}
    used_keys = set()
    kv_store = {}

    def _set_param(key, value):
        if key in new_state:
            raise KeyError(f'duplicate key generated: {key}')
        new_state[key] = value

    for key, value in state.items():
        m = patch_re.match(key)
        if m:
            stage = int(m.group(1)) + 1
            part = m.group(2)
            param = m.group(3)
            if part == 'proj':
                new_key = f'patch_embed{stage}.proj.{param}'
            else:
                new_key = f'patch_embed{stage}.norm.{param}'
            _set_param(new_key, value)
            used_keys.add(key)
            continue

        m = layer_norm_re.match(key)
        if m:
            stage = int(m.group(1)) + 1
            param = m.group(2)
            new_key = f'norm{stage}.{param}'
            _set_param(new_key, value)
            used_keys.add(key)
            continue

        m = block_re.match(key)
        if m:
            stage = int(m.group(1)) + 1
            block = int(m.group(2))
            rest = m.group(3)

            if rest.startswith('layer_norm_1.'):
                param = rest.split('.', 1)[1]
                _set_param(f'block{stage}.{block}.norm1.{param}', value)
                used_keys.add(key)
                continue
            if rest.startswith('layer_norm_2.'):
                param = rest.split('.', 1)[1]
                _set_param(f'block{stage}.{block}.norm2.{param}', value)
                used_keys.add(key)
                continue

            if rest.startswith('attention.self.query.'):
                param = rest.split('.', 3)[3]
                _set_param(f'block{stage}.{block}.attn.q.{param}', value)
                used_keys.add(key)
                continue
            if rest.startswith('attention.self.key.'):
                param = rest.split('.', 3)[3]
                kv_store.setdefault((stage, block, param), {})['k'] = value
                used_keys.add(key)
                continue
            if rest.startswith('attention.self.value.'):
                param = rest.split('.', 3)[3]
                kv_store.setdefault((stage, block, param), {})['v'] = value
                used_keys.add(key)
                continue
            if rest.startswith('attention.output.dense.'):
                param = rest.split('.', 3)[3]
                _set_param(f'block{stage}.{block}.attn.proj.{param}', value)
                used_keys.add(key)
                continue
            if rest.startswith('attention.self.sr.'):
                param = rest.split('.', 3)[3]
                _set_param(f'block{stage}.{block}.attn.sr.{param}', value)
                used_keys.add(key)
                continue
            if rest.startswith('attention.self.layer_norm.'):
                param = rest.split('.', 3)[3]
                _set_param(f'block{stage}.{block}.attn.norm.{param}', value)
                used_keys.add(key)
                continue

            if rest.startswith('mlp.dense1.'):
                param = rest.split('.', 2)[2]
                _set_param(f'block{stage}.{block}.mlp.fc1.{param}', value)
                used_keys.add(key)
                continue
            if rest.startswith('mlp.dense2.'):
                param = rest.split('.', 2)[2]
                _set_param(f'block{stage}.{block}.mlp.fc2.{param}', value)
                used_keys.add(key)
                continue
            if rest.startswith('mlp.dwconv.dwconv.'):
                param = rest.split('.', 3)[3]
                _set_param(f'block{stage}.{block}.mlp.dwconv.dwconv.{param}',
                           value)
                used_keys.add(key)
                continue

    # Combine K and V into KV
    for (stage, block, param), kv in kv_store.items():
        if 'k' not in kv or 'v' not in kv:
            raise KeyError(
                f'missing K/V for block{stage}.{block} param {param}')
        new_key = f'block{stage}.{block}.attn.kv.{param}'
        new_val = torch.cat([kv['k'], kv['v']], dim=0)
        _set_param(new_key, new_val)

    return new_state, used_keys


def _verify(new_state, arch):
    from mmseg.models import build_backbone

    model = build_backbone(dict(type=arch, style='pytorch'))
    # Strict load to ensure all keys match.
    model.load_state_dict(new_state, strict=True)


def main():
    parser = argparse.ArgumentParser(
        description='Convert HF SegFormer checkpoint to mmseg MiT backbone.')
    parser.add_argument('--checkpoint', required=True, help='HF checkpoint.')
    parser.add_argument('--output', required=True, help='Output .pth path.')
    parser.add_argument('--arch', default='mit_b5', help='MiT backbone arch.')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify by loading into mmseg backbone (requires mmseg).')
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    state = _load_state_dict(args.checkpoint)
    new_state, used_keys = _map_hf_to_mmseg(state)

    # Report unused keys (ignoring decoder/classifier heads)
    unused = [
        k for k in state.keys()
        if k not in used_keys and not k.startswith('classifier.')
        and not k.startswith('segformer.decode_head.')
    ]
    print(f'Converted keys: {len(new_state)}')
    print(f'Unused keys (excluding heads): {len(unused)}')
    if unused:
        print('Sample unused keys:', unused[:20])

    if args.verify:
        _verify(new_state, args.arch)
        print('Verification passed.')

    torch.save(
        {
            'state_dict': new_state,
            'meta': {
                'source': args.checkpoint,
                'arch': args.arch,
            }
        },
        args.output,
    )
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
