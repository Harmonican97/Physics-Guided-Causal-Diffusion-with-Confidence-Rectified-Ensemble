"""Aggregate completed strict-protocol seeds without imputing missing runs."""

import csv
import glob
import json
import os

import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, 'results_aggregated_strict')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load(pattern):
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
        with open(path, encoding='utf-8') as handle:
            payload = json.load(handle)
        payload['_path'] = os.path.relpath(path, ROOT)
        rows.append(payload)
    return rows


def _summary(rows, class_names):
    result = {'n_completed_seeds': len(rows), 'source_files': [row['_path'] for row in rows]}
    for metric in ('seen_accuracy', 'unseen_accuracy', 'h_score'):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = {
            'mean': float(values.mean()) if len(values) else None,
            'std': float(values.std(ddof=1)) if len(values) > 1 else None,
            'values': values.tolist(),
        }
    per_class = {}
    for name in class_names:
        per_class[name] = {}
        for metric in ('precision', 'recall', 'f1-score', 'support'):
            values = np.asarray([row['per_class'][name][metric] for row in rows], dtype=float)
            per_class[name][metric] = {
                'mean': float(values.mean()) if len(values) else None,
                'std': float(values.std(ddof=1)) if len(values) > 1 else None,
                'values': values.tolist(),
            }
    result['per_class'] = per_class
    return result


def main():
    xjtu = _load('results_xjtu_strict/seed_*/metrics.json')
    hust = _load('results_hust_confidence_strict/seed_*/metrics.json')
    # Smoke-test seed 999 is never a reportable experimental seed.
    xjtu = [row for row in xjtu if '/seed_999/' not in row['_path'].replace('\\', '/')]
    hust = [row for row in hust if '/seed_999/' not in row['_path'].replace('\\', '/')]
    payload = {
        'xjtu': _summary(xjtu, ['Ball', 'Inner', 'Outer', 'Mix']),
        'hust': _summary(hust, ['N', 'B', 'I', 'O', 'IB', 'OB']),
        'std_definition': 'sample standard deviation with ddof=1; null when only one seed is complete',
        'runtime_note': (
            'runtime_seconds is intentionally excluded because entry points may '
            'either train from scratch or reload cached checkpoints; use a '
            'separate controlled benchmark for latency claims'
        ),
    }
    json_path = os.path.join(OUTPUT_DIR, 'retraining_summary.json')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)

    csv_path = os.path.join(OUTPUT_DIR, 'retraining_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'metric', 'mean', 'std', 'n_completed_seeds'])
        for dataset in ('xjtu', 'hust'):
            block = payload[dataset]
            for metric in ('seen_accuracy', 'unseen_accuracy', 'h_score'):
                writer.writerow([
                    dataset, metric, block[metric]['mean'], block[metric]['std'],
                    block['n_completed_seeds'],
                ])
    print(json_path)
    print(csv_path)


if __name__ == '__main__':
    main()
