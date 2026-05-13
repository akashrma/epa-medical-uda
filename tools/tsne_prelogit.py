#!/usr/bin/env python
"""t-SNE/PCA analysis for SegFormer pre-classifier pixel features.

The captured representation is the input to ``decode_head.linear_pred``.
For MR->CT configs, ``cfg.data.test`` is the target CT split, so target
ground truth is used only for post-hoc coloring and validation metrics.
"""

import argparse
import csv
import json
import os
import os.path as osp
import re
import sys
from collections import OrderedDict, defaultdict

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_state_dict, wrap_fp16_model
from mmcv.runner.checkpoint import _load_checkpoint
from scipy.spatial.distance import cdist

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from tools.test import _fix_img_scale_tuple, _iter_dataset_cfgs, update_legacy_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze target pre-logit representations with t-SNE.')
    parser.add_argument('config', help='MMSeg config path')
    parser.add_argument(
        'checkpoints',
        nargs='*',
        help='Checkpoint paths. If omitted, use --checkpoint-dir discovery.')
    parser.add_argument(
        '--checkpoint-dir',
        action='append',
        default=[],
        help='Directory to scan for iter_*.pth/best_*_iter_*.pth checkpoints.')
    parser.add_argument(
        '--checkpoint-stride',
        type=int,
        default=1000,
        help='When discovering checkpoints, keep iterations divisible by this.'
        ' Use 0 to keep all discovered checkpoints.')
    parser.add_argument(
        '--out-dir',
        default='work_dirs/tsne/prelogit',
        help='Directory for plots, sampled arrays, and metrics.')
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=1000,
        help='Maximum sampled target pixels per foreground class.')
    parser.add_argument(
        '--include-background',
        action='store_true',
        help='Also sample class 0 background pixels.')
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='Limit target images/slices used for sampling. 0 means all.')
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Data loader workers per GPU.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for fixed pixel sampling and t-SNE.')
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='t-SNE perplexity.')
    parser.add_argument(
        '--pca-dim',
        type=int,
        default=50,
        help='PCA dimensions before t-SNE/metrics.')
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable L2 normalization before PCA/t-SNE/metrics.')
    parser.add_argument(
        '--save-features',
        action='store_true',
        help='Save sampled feature arrays for each checkpoint.')
    return parser.parse_args()


def checkpoint_iter(path):
    matches = re.findall(r'(?:^|[_/])iter_(\d+)', path)
    if matches:
        return int(matches[-1])
    return -1


def discover_checkpoints(dirs, stride):
    paths = []
    for ckpt_dir in dirs:
        for root, _, files in os.walk(ckpt_dir):
            for name in files:
                if not name.endswith('.pth'):
                    continue
                path = osp.join(root, name)
                iteration = checkpoint_iter(path)
                if iteration < 0:
                    continue
                if stride and iteration % stride != 0:
                    continue
                paths.append(path)
    return sorted(set(paths), key=lambda p: (checkpoint_iter(p), p))


def build_target_dataset(cfg):
    cfg = cfg.copy()
    for dataset_cfg in _iter_dataset_cfgs(cfg.data.test):
        _fix_img_scale_tuple(dataset_cfg)
        dataset_cfg['test_mode'] = True
    return build_dataset(cfg.data.test)


def build_model(cfg, checkpoint_path, dataset, device='cuda'):
    cfg = cfg.copy()
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = _load_checkpoint(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    revised_state_dict = OrderedDict()
    for key, value in state_dict.items():
        key = re.sub(r'^module\.', '', key)
        key = re.sub(r'^model\.', '', key)
        if key.startswith('ema_'):
            continue
        revised_state_dict[key] = value
    load_state_dict(model, revised_state_dict, strict=False, logger=None)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        model.PALETTE = dataset.PALETTE

    model.eval()
    if device == 'cuda':
        model = MMDataParallel(model.cuda(), device_ids=[0])
    return model


def register_prelogit_hook(model):
    base = model.module if hasattr(model, 'module') else model
    holder = {}

    def hook(_, inputs, __):
        holder['features'] = inputs[0].detach()

    handle = base.decode_head.linear_pred.register_forward_hook(hook)
    return holder, handle


def forward_model(model, data, device='cuda'):
    with torch.no_grad():
        if device == 'cuda':
            _ = model(return_loss=False, rescale=True, **data)
        else:
            raise RuntimeError('CPU inference path is not implemented.')


def load_gt(dataset, idx):
    ann = dataset.img_infos[idx].get('ann')
    if ann is None or 'seg_map' not in ann:
        raise KeyError('Target dataset must include ann_list/seg_map for GT.')
    return np.load(ann['seg_map'])


def resize_label_nearest(label, size_hw):
    tensor = torch.from_numpy(label.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=size_hw, mode='nearest')
    return resized[0, 0].numpy().astype(np.int64)


def choose_image_indices(num_images, max_images, rng):
    indices = np.arange(num_images)
    if max_images and max_images < num_images:
        indices = rng.choice(indices, size=max_images, replace=False)
        indices = np.sort(indices)
    return indices.tolist()


def make_sample_plan(dataset, feature_hw, classes, samples_per_class,
                     max_images, seed):
    rng = np.random.RandomState(seed)
    indices = choose_image_indices(len(dataset), max_images, rng)
    candidates = {c: [] for c in classes}

    for idx in indices:
        label = resize_label_nearest(load_gt(dataset, idx), feature_hw)
        for cls_id in classes:
            coords = np.argwhere(label == cls_id)
            if coords.size == 0:
                continue
            for y, x in coords:
                candidates[cls_id].append((idx, int(y), int(x)))

    plan = defaultdict(list)
    counts = {}
    for cls_id, cls_candidates in candidates.items():
        if len(cls_candidates) == 0:
            counts[cls_id] = 0
            continue
        keep = min(samples_per_class, len(cls_candidates))
        selected = rng.choice(len(cls_candidates), size=keep, replace=False)
        counts[cls_id] = keep
        for candidate_idx in selected:
            idx, y, x = cls_candidates[int(candidate_idx)]
            plan[idx].append((y, x, cls_id))

    return dict(plan), counts, indices


def l2_normalize(features, eps=1e-12):
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, eps)


def pca_reduce(features, n_components):
    if n_components <= 0 or n_components >= features.shape[1]:
        return features
    centered = features - features.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    dim = min(n_components, vt.shape[0])
    return np.dot(centered, vt[:dim].T)


def compute_silhouette(features, labels):
    unique = np.array(sorted(set(labels.tolist())))
    if len(unique) < 2:
        return {'overall': float('nan'), 'per_class': {}}

    per_class = {}
    sample_scores = []
    for cls_id in unique:
        cls_mask = labels == cls_id
        cls_features = features[cls_mask]
        if len(cls_features) < 2:
            per_class[int(cls_id)] = float('nan')
            continue

        same_dist = cdist(cls_features, cls_features, metric='euclidean')
        a = same_dist.sum(axis=1) / np.maximum(len(cls_features) - 1, 1)

        b = None
        for other_id in unique:
            if other_id == cls_id:
                continue
            other_features = features[labels == other_id]
            if len(other_features) == 0:
                continue
            mean_dist = cdist(
                cls_features, other_features, metric='euclidean').mean(axis=1)
            b = mean_dist if b is None else np.minimum(b, mean_dist)

        denom = np.maximum(a, b)
        scores = np.where(denom > 0, (b - a) / denom, 0.0)
        per_class[int(cls_id)] = float(np.mean(scores))
        sample_scores.append(scores)

    if sample_scores:
        overall = float(np.mean(np.concatenate(sample_scores)))
    else:
        overall = float('nan')
    return {'overall': overall, 'per_class': per_class}


def compute_metrics(features, labels):
    unique = np.array(sorted(set(labels.tolist())))
    centroids = {}
    intra = {}
    for cls_id in unique:
        cls_features = features[labels == cls_id]
        centroid = cls_features.mean(axis=0)
        centroids[int(cls_id)] = centroid
        intra[int(cls_id)] = float(
            np.mean(np.sum((cls_features - centroid)**2, axis=1)))

    centroid_matrix = np.stack([centroids[int(c)] for c in unique], axis=0)
    centroid_dists = cdist(centroid_matrix, centroid_matrix)
    non_diag = centroid_dists[~np.eye(len(unique), dtype=bool)]

    dists_to_centroids = cdist(features, centroid_matrix)
    pred = unique[np.argmin(dists_to_centroids, axis=1)]
    centroid_acc = float(np.mean(pred == labels))
    centroid_acc_per_class = {}
    for cls_id in unique:
        mask = labels == cls_id
        centroid_acc_per_class[int(cls_id)] = float(np.mean(pred[mask] == cls_id))

    silhouette = compute_silhouette(features, labels)
    return {
        'silhouette': silhouette['overall'],
        'silhouette_per_class': silhouette['per_class'],
        'intra_variance_per_class': intra,
        'intra_variance_mean': float(np.mean(list(intra.values()))),
        'inter_centroid_distance_mean': float(np.mean(non_diag)),
        'inter_centroid_distance_min': float(np.min(non_diag)),
        'nearest_centroid_accuracy': centroid_acc,
        'nearest_centroid_accuracy_per_class': centroid_acc_per_class,
        'centroid_distance_matrix': centroid_dists.tolist(),
    }


def run_tsne(features, seed, perplexity):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return None

    usable_perplexity = min(float(perplexity), max(2.0, (len(features) - 1) / 3.0))
    return TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        perplexity=usable_perplexity,
        random_state=seed).fit_transform(features)


def plot_embedding(embedding, labels, class_names, title, out_file):
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=180)
    cmap = {
        0: '#9e9e9e',
        1: '#d62728',
        2: '#1f77b4',
        3: '#2ca02c',
        4: '#ff7f0e',
    }
    for cls_id in sorted(set(labels.tolist())):
        mask = labels == cls_id
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=5,
            c=cmap.get(int(cls_id), None),
            label=name,
            alpha=0.65,
            linewidths=0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=3, frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def metrics_flat_row(checkpoint, metrics, class_names):
    row = {
        'checkpoint': checkpoint,
        'iter': checkpoint_iter(checkpoint),
        'silhouette': metrics['silhouette'],
        'intra_variance_mean': metrics['intra_variance_mean'],
        'inter_centroid_distance_mean': metrics['inter_centroid_distance_mean'],
        'inter_centroid_distance_min': metrics['inter_centroid_distance_min'],
        'nearest_centroid_accuracy': metrics['nearest_centroid_accuracy'],
    }
    for field in [
            'silhouette_per_class', 'intra_variance_per_class',
            'nearest_centroid_accuracy_per_class'
    ]:
        for cls_id, value in metrics[field].items():
            cls_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(cls_id)
            row[f'{field}.{cls_name}'] = value
    return row


def write_metrics_csv(rows, out_file):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_feature_hw(cfg, checkpoint, dataset, data_loader, device):
    model = build_model(cfg, checkpoint, dataset, device=device)
    holder, handle = register_prelogit_hook(model)
    data = next(iter(data_loader))
    forward_model(model, data, device=device)
    features = holder.get('features')
    handle.remove()
    del model
    torch.cuda.empty_cache()
    if features is None:
        raise RuntimeError('Failed to capture decode_head.linear_pred input.')
    return tuple(features.shape[-2:])


def extract_checkpoint_features(cfg, checkpoint, dataset, data_loader,
                                sample_plan, device):
    model = build_model(cfg, checkpoint, dataset, device=device)
    holder, handle = register_prelogit_hook(model)
    feature_chunks = []
    label_chunks = []
    source_indices = []

    for idx, data in enumerate(data_loader):
        if idx not in sample_plan:
            continue
        forward_model(model, data, device=device)
        features = holder.get('features')
        if features is None:
            raise RuntimeError('Hook did not capture pre-logit features.')
        fmap = features[0].detach().float().cpu().numpy().transpose(1, 2, 0)
        coords = sample_plan[idx]
        for y, x, cls_id in coords:
            feature_chunks.append(fmap[y, x])
            label_chunks.append(cls_id)
            source_indices.append(idx)

    handle.remove()
    del model
    torch.cuda.empty_cache()
    return (
        np.stack(feature_chunks, axis=0),
        np.array(label_chunks, dtype=np.int64),
        np.array(source_indices, dtype=np.int64),
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this repository inference path.')

    cfg = mmcv.Config.fromfile(args.config)
    cfg = update_legacy_cfg(cfg)
    checkpoints = list(args.checkpoints)
    if not checkpoints:
        checkpoints = discover_checkpoints(args.checkpoint_dir, args.checkpoint_stride)
    checkpoints = sorted(checkpoints, key=lambda p: (checkpoint_iter(p), p))
    if not checkpoints:
        raise FileNotFoundError('No checkpoints found.')

    mmcv.mkdir_or_exist(args.out_dir)
    dataset = build_target_dataset(cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False)

    class_names = tuple(getattr(dataset, 'CLASSES', tuple(map(str, range(5)))))
    classes = list(range(0 if args.include_background else 1, len(class_names)))

    print(f'Config: {args.config}')
    print(f'Target samples: {len(dataset)}')
    print(f'Checkpoints: {len(checkpoints)}')
    for path in checkpoints:
        print(f'  iter {checkpoint_iter(path)}: {path}')

    feature_hw = get_feature_hw(
        cfg, checkpoints[0], dataset, data_loader, device='cuda')
    print(f'Captured pre-logit feature map size: {feature_hw}')

    sample_plan, counts, image_indices = make_sample_plan(
        dataset,
        feature_hw,
        classes,
        args.samples_per_class,
        args.max_images,
        args.seed)
    print('Sample counts per class:')
    for cls_id in classes:
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        print(f'  {cls_id} {name}: {counts.get(cls_id, 0)}')

    manifest = {
        'config': osp.abspath(args.config),
        'checkpoints': [osp.abspath(p) for p in checkpoints],
        'feature_hw': feature_hw,
        'classes': list(class_names),
        'sample_counts': {str(k): int(v) for k, v in counts.items()},
        'image_indices': image_indices,
        'normalized': not args.no_normalize,
        'seed': args.seed,
    }
    mmcv.dump(manifest, osp.join(args.out_dir, 'manifest.json'), indent=2)

    metric_rows = []
    metric_records = []
    for checkpoint in checkpoints:
        iteration = checkpoint_iter(checkpoint)
        tag = f'iter_{iteration:06d}' if iteration >= 0 else osp.splitext(
            osp.basename(checkpoint))[0]
        print(f'Processing {tag}: {checkpoint}')
        features, labels, source_indices = extract_checkpoint_features(
            cfg, checkpoint, dataset, data_loader, sample_plan, device='cuda')
        if np.isnan(features).any() or np.isinf(features).any():
            raise ValueError(f'NaN/Inf detected in features for {checkpoint}')

        analysis_features = l2_normalize(features) if not args.no_normalize else features
        pca_features = pca_reduce(analysis_features, args.pca_dim)
        metrics = compute_metrics(pca_features, labels)

        pca_2d = pca_reduce(analysis_features, 2)
        plot_embedding(
            pca_2d,
            labels,
            class_names,
            f'Target pre-logit PCA - {tag}',
            osp.join(args.out_dir, f'{tag}_pca_by_class.png'))

        tsne = run_tsne(pca_features, args.seed, args.perplexity)
        if tsne is not None:
            plot_embedding(
                tsne,
                labels,
                class_names,
                f'Target pre-logit t-SNE - {tag}',
                osp.join(args.out_dir, f'{tag}_tsne_by_class.png'))
            np.save(osp.join(args.out_dir, f'{tag}_tsne.npy'), tsne)
        else:
            print('scikit-learn is not installed; skipped t-SNE plot.')

        np.save(osp.join(args.out_dir, f'{tag}_pca.npy'), pca_2d)
        if args.save_features:
            np.savez_compressed(
                osp.join(args.out_dir, f'{tag}_features.npz'),
                features=features,
                labels=labels,
                source_indices=source_indices)

        record = {
            'checkpoint': osp.abspath(checkpoint),
            'iter': iteration,
            'metrics': metrics,
        }
        metric_records.append(record)
        row = metrics_flat_row(checkpoint, metrics, class_names)
        metric_rows.append(row)
        print(
            f"  silhouette={metrics['silhouette']:.4f}, "
            f"inter_mean={metrics['inter_centroid_distance_mean']:.4f}, "
            f"intra_mean={metrics['intra_variance_mean']:.4f}, "
            f"nearest_centroid_acc={metrics['nearest_centroid_accuracy']:.4f}")

    mmcv.dump(metric_records, osp.join(args.out_dir, 'metrics.json'), indent=2)
    write_metrics_csv(metric_rows, osp.join(args.out_dir, 'metrics.csv'))
    print(f'Wrote outputs to {args.out_dir}')


if __name__ == '__main__':
    main()
