#!/usr/bin/env python
"""Combine no-FDA evaluation and representation metrics."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path('work_dirs')
METHODS = [('etf', 'Fixed ETF'), ('srcproto', 'Dynamic SrcProto')]
TSNE_DIRS = {
    ('mr2ct', 'etf'): ROOT / 'tsne' / 'mr2ct_etf_tgt_nofda_10k',
    ('mr2ct', 'srcproto'): ROOT / 'tsne' /
    'mr2ct_srcproto_tgtalign_nofda_10k',
    ('ct2mr', 'etf'): ROOT / 'tsne' / 'ct2mr_etf_tgt_nofda_10k',
    ('ct2mr', 'srcproto'): ROOT / 'tsne' /
    'ct2mr_srcproto_tgtalign_nofda_10k',
}


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value in (None, ''):
        return float('nan')
    return float(value)


def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys)
             if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return float('nan')
    xs, ys = zip(*pairs)
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    den_x = math.sqrt(sum((x - x_mean)**2 for x in xs))
    den_y = math.sqrt(sum((y - y_mean)**2 for y in ys))
    if den_x == 0 or den_y == 0:
        return float('nan')
    return num / (den_x * den_y)


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_combined_rows():
    combined = []
    for direction in ['mr2ct', 'ct2mr']:
        eval_rows = read_csv(
            ROOT / f'eval_{direction}_nofda_10k_dice_asd' /
            'dice_asd_summary.csv')
        eval_by_key = {
            (row['method'], int(row['iter'])): row
            for row in eval_rows
        }
        for method, _ in METHODS:
            rep_rows = read_csv(TSNE_DIRS[(direction, method)] / 'metrics.csv')
            for rep in rep_rows:
                iteration = int(rep['iter'])
                ev = eval_by_key[(method, iteration)]
                row = {
                    'direction': direction,
                    'method': method,
                    'iter': iteration,
                    'mDice_percent': ev['mDice_percent'],
                    'mASSD': ev['mASSD'],
                    'silhouette': rep['silhouette'],
                    'nearest_centroid_accuracy':
                    rep['nearest_centroid_accuracy'],
                    'intra_variance_mean': rep['intra_variance_mean'],
                    'inter_centroid_distance_mean':
                    rep['inter_centroid_distance_mean'],
                    'inter_centroid_distance_min':
                    rep['inter_centroid_distance_min'],
                }
                for cls_name in ['Myo', 'LAC', 'LVC', 'AA']:
                    row[f'Dice_{cls_name}_percent'] = ev[
                        f'Dice_{cls_name}_percent']
                    row[f'ASSD_{cls_name}'] = ev[f'ASSD_{cls_name}']
                    for prefix in [
                            'silhouette_per_class',
                            'nearest_centroid_accuracy_per_class',
                            'intra_variance_per_class'
                    ]:
                        row[f'{prefix}.{cls_name}'] = rep.get(
                            f'{prefix}.{cls_name}', '')
                combined.append(row)
    combined.sort(key=lambda row: (row['direction'], row['method'],
                                   row['iter']))
    return combined


def plot_representation(direction, rows):
    out_dir = ROOT / f'eval_{direction}_nofda_10k_dice_asd'
    metrics = [
        ('silhouette', 'Silhouette'),
        ('nearest_centroid_accuracy', 'Nearest-centroid accuracy'),
        ('inter_centroid_distance_mean', 'Inter-centroid distance mean'),
        ('intra_variance_mean', 'Intra-class variance mean'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=180, sharex=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for method, method_label in METHODS:
            method_rows = [
                row for row in rows
                if row['direction'] == direction and row['method'] == method
            ]
            method_rows.sort(key=lambda row: row['iter'])
            ax.plot([row['iter'] for row in method_rows],
                    [to_float(row[metric]) for row in method_rows],
                    marker='o',
                    label=method_label)
        ax.set_title(label)
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel('Iteration')
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f'{direction.upper()} no-FDA representation metrics')
    fig.tight_layout()
    fig.savefig(out_dir / f'{direction}_representation_metrics_comparison.png')
    plt.close(fig)


def markdown_summary(rows):
    lines = ['# No-FDA Evaluation and Representation Summary', '']
    for direction in ['mr2ct', 'ct2mr']:
        lines += [
            f'## {direction.upper()} Correlation',
            '| Method | corr(silhouette,mDice) | corr(NCA,mDice) | corr(intra,mDice) | corr(silhouette,mASSD) |',
            '|---|---:|---:|---:|---:|',
        ]
        for method, label in METHODS:
            method_rows = [
                row for row in rows
                if row['direction'] == direction and row['method'] == method
            ]
            mdice = [to_float(row['mDice_percent']) for row in method_rows]
            massd = [to_float(row['mASSD']) for row in method_rows]
            sil = [to_float(row['silhouette']) for row in method_rows]
            nca = [
                to_float(row['nearest_centroid_accuracy'])
                for row in method_rows
            ]
            intra = [to_float(row['intra_variance_mean']) for row in method_rows]
            lines.append(
                f'| {label} | {corr(sil, mdice):.3f} | '
                f'{corr(nca, mdice):.3f} | {corr(intra, mdice):.3f} | '
                f'{corr(sil, massd):.3f} |')
        lines += [
            '',
            f'## {direction.upper()} Final Checkpoint Representation',
            '| Method | mDice % | mASSD | silhouette | NCA | inter mean | intra mean |',
            '|---|---:|---:|---:|---:|---:|---:|',
        ]
        for method, label in METHODS:
            final = [
                row for row in rows
                if row['direction'] == direction and row['method'] == method
                and row['iter'] == 10000
            ][0]
            lines.append(
                f"| {label} | {to_float(final['mDice_percent']):.2f} | "
                f"{to_float(final['mASSD']):.4f} | "
                f"{to_float(final['silhouette']):.4f} | "
                f"{to_float(final['nearest_centroid_accuracy']):.4f} | "
                f"{to_float(final['inter_centroid_distance_mean']):.4f} | "
                f"{to_float(final['intra_variance_mean']):.4f} |")
        lines.append('')
    return '\n'.join(lines)


def main():
    rows = load_combined_rows()
    write_csv(ROOT / 'nofda_eval_representation_summary.csv', rows)
    for direction in ['mr2ct', 'ct2mr']:
        plot_representation(direction, rows)
    summary = markdown_summary(rows)
    (ROOT / 'nofda_eval_representation_summary.md').write_text(summary + '\n')
    print(summary)


if __name__ == '__main__':
    main()
