#!/usr/bin/env python
"""Summarize no-FDA medical Dice/ASSD evaluation logs."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path('work_dirs')
CLASSES = ['Myo', 'LAC', 'LVC', 'AA']
ITERS = list(range(1000, 10001, 1000))
METHODS = [('etf', 'Fixed ETF'), ('srcproto', 'Dynamic SrcProto')]


def grab_class(section, cls_name):
    match = re.search(
        r'\|\s*' + re.escape(cls_name) + r'\s*\|\s*([0-9.]+)\s*\|',
        section)
    if not match:
        raise ValueError(f'missing {cls_name}')
    return float(match.group(1))


def parse_log(path):
    text = path.read_text(errors='replace')
    if 'Summary:' not in text:
        raise ValueError(f'incomplete log {path}')
    dice_section, rest = text.split('Per-class ASSD', 1)
    assd_section, summary_section = rest.split('Summary:', 1)
    row = {}
    for cls_name in CLASSES:
        row[f'Dice_{cls_name}_percent'] = grab_class(dice_section, cls_name)
        row[f'ASSD_{cls_name}'] = grab_class(assd_section, cls_name)
    nums = re.findall(
        r'\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|',
        summary_section)
    if not nums:
        raise ValueError(f'missing summary {path}')
    row['mDice_percent'] = float(nums[-1][0])
    row['mASSD'] = float(nums[-1][1])
    return row


def write_csv(path, rows):
    fields = ['direction', 'method', 'iter', 'mDice_percent', 'mASSD']
    fields += [f'Dice_{cls_name}_percent' for cls_name in CLASSES]
    fields += [f'ASSD_{cls_name}' for cls_name in CLASSES]
    fields += ['log']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_metric(out_dir, direction, rows, metric, ylabel, title):
    plt.figure(figsize=(7.0, 4.2), dpi=180)
    for method, label in METHODS:
        method_rows = [row for row in rows if row['method'] == method]
        method_rows.sort(key=lambda row: row['iter'])
        plt.plot([row['iter'] for row in method_rows],
                 [row[metric] for row in method_rows],
                 marker='o',
                 label=label)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / f'{direction}_{metric}_comparison.png')
    plt.close()


def plot_classwise(out_dir, direction, rows, prefix, ylabel, title):
    fig, axes = plt.subplots(
        2, 2, figsize=(9.0, 6.5), dpi=180, sharex=True)
    axes = axes.ravel()
    for ax, cls_name in zip(axes, CLASSES):
        if prefix == 'ASSD':
            key = f'ASSD_{cls_name}'
        else:
            key = f'Dice_{cls_name}_percent'
        for method, label in METHODS:
            method_rows = [row for row in rows if row['method'] == method]
            method_rows.sort(key=lambda row: row['iter'])
            ax.plot([row['iter'] for row in method_rows],
                    [row[key] for row in method_rows],
                    marker='o',
                    label=label)
        ax.set_title(cls_name)
        ax.grid(alpha=0.25)
    for ax in axes[2:]:
        ax.set_xlabel('Iteration')
    for ax in axes[::2]:
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / f'{direction}_classwise_{prefix}_comparison.png')
    plt.close(fig)


def markdown_summary(direction, rows):
    lines = [
        f'# {direction.upper()} no-FDA ETF vs Dynamic Source Prototype Summary',
        '',
        '## Best Checkpoints',
        '| Method | Best mDice iter | Best mDice % | mASSD there | Best mASSD iter | Best mASSD | mDice there |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for method, label in METHODS:
        method_rows = [row for row in rows if row['method'] == method]
        best_dice = max(method_rows, key=lambda row: row['mDice_percent'])
        best_assd = min(method_rows, key=lambda row: row['mASSD'])
        lines.append(
            f"| {label} | {best_dice['iter']} | "
            f"{best_dice['mDice_percent']:.2f} | {best_dice['mASSD']:.4f} | "
            f"{best_assd['iter']} | {best_assd['mASSD']:.4f} | "
            f"{best_assd['mDice_percent']:.2f} |")

    lines += [
        '',
        '## Best Class-Wise Dice',
        '| Method | Myo | LAC | LVC | AA |',
        '|---|---:|---:|---:|---:|',
    ]
    for method, label in METHODS:
        method_rows = [row for row in rows if row['method'] == method]
        vals = []
        for cls_name in CLASSES:
            key = f'Dice_{cls_name}_percent'
            best = max(method_rows, key=lambda row: row[key])
            vals.append(f"{best[key]:.2f} @ {best['iter']}")
        lines.append(f"| {label} | " + ' | '.join(vals) + ' |')

    lines += [
        '',
        '## Final Checkpoint',
        '| Method | mDice % | mASSD | Myo Dice | LAC Dice | LVC Dice | AA Dice |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for method, label in METHODS:
        final = [
            row for row in rows
            if row['method'] == method and row['iter'] == 10000
        ][0]
        lines.append(
            f"| {label} | {final['mDice_percent']:.2f} | "
            f"{final['mASSD']:.4f} | {final['Dice_Myo_percent']:.2f} | "
            f"{final['Dice_LAC_percent']:.2f} | "
            f"{final['Dice_LVC_percent']:.2f} | "
            f"{final['Dice_AA_percent']:.2f} |")
    return '\n'.join(lines) + '\n'


def main():
    all_rows = []
    for direction in ['mr2ct', 'ct2mr']:
        out_dir = ROOT / f'eval_{direction}_nofda_10k_dice_asd'
        rows = []
        for method, _ in METHODS:
            for iteration in ITERS:
                log = out_dir / 'logs' / f'{method}_iter_{iteration}.log'
                parsed = parse_log(log)
                parsed.update({
                    'direction': direction,
                    'method': method,
                    'iter': iteration,
                    'log': str(log),
                })
                rows.append(parsed)
                all_rows.append(parsed)
        rows.sort(key=lambda row: (row['method'], row['iter']))
        write_csv(out_dir / 'dice_asd_summary.csv', rows)
        (out_dir / f'{direction}_analysis_summary.md').write_text(
            markdown_summary(direction, rows))
        plot_metric(out_dir, direction, rows, 'mDice_percent', 'mDice (%)',
                    f'{direction.upper()} no-FDA mDice')
        plot_metric(out_dir, direction, rows, 'mASSD', 'mASSD',
                    f'{direction.upper()} no-FDA mASSD')
        plot_classwise(out_dir, direction, rows, 'Dice', 'Dice (%)',
                       f'{direction.upper()} no-FDA class-wise Dice')
        plot_classwise(out_dir, direction, rows, 'ASSD', 'ASSD',
                       f'{direction.upper()} no-FDA class-wise ASSD')

    write_csv(ROOT / 'eval_nofda_10k_dice_asd_summary.csv', all_rows)
    for direction in ['mr2ct', 'ct2mr']:
        out_dir = ROOT / f'eval_{direction}_nofda_10k_dice_asd'
        print(out_dir / 'dice_asd_summary.csv')
        print((out_dir / f'{direction}_analysis_summary.md').read_text())


if __name__ == '__main__':
    main()
