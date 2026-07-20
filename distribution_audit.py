"""Evaluation-only synthetic/real alignment and temporal-drift diagnostics.

Real compound-fault labels are used only after training to compute reportable
distances. Nothing in this script writes model checkpoints or selects training
hyperparameters.
"""

import json
import os

import numpy as np
import torch
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import get_temporal_dataloaders
from hust_data import HUSTTemporalDataset, compute_seen_train_stats


ROOT = os.path.dirname(os.path.abspath(__file__))
SEED = int(os.environ.get('PGCD_SEED', '0'))
MAX_SAMPLES = int(os.environ.get('PGCD_AUDIT_MAX_SAMPLES', '500'))


def _select(array, max_samples, seed):
    array = np.asarray(array)
    if len(array) <= max_samples:
        return array
    rng = np.random.default_rng(seed)
    return array[rng.choice(len(array), max_samples, replace=False)]


def envelope_features(signals, fs, max_samples=MAX_SAMPLES, seed=0):
    """Return normalized log envelope-power spectra under the paper convention."""
    if torch.is_tensor(signals):
        signals = signals.detach().cpu().numpy()
    signals = np.asarray(signals, dtype=np.float64).reshape(len(signals), -1)
    signals = _select(signals, max_samples, seed)
    signals = signals - signals.mean(axis=1, keepdims=True)
    envelope = np.abs(hilbert(signals, axis=1))
    envelope = envelope - envelope.mean(axis=1, keepdims=True)
    envelope *= np.hanning(envelope.shape[1])[None, :]
    power = np.abs(np.fft.rfft(envelope, axis=1, norm='ortho')) ** 2
    frequencies = np.fft.rfftfreq(envelope.shape[1], d=1.0 / fs)
    power = power[:, frequencies >= 5.0]
    power /= np.maximum(power.sum(axis=1, keepdims=True), 1e-12)
    return np.log(power + 1e-12)


def rbf_mmd_unbiased(x, y, seed=0):
    """Unbiased squared MMD with a joint median-distance bandwidth."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pooled = _select(np.vstack([x, y]), min(500, len(x) + len(y)), seed)
    distances = pairwise_distances(pooled, metric='euclidean')
    positive = distances[distances > 0]
    median = float(np.median(positive)) if len(positive) else 1.0
    gamma = 1.0 / max(2.0 * median * median, 1e-12)
    k_xx = np.exp(-gamma * pairwise_distances(x, metric='sqeuclidean'))
    k_yy = np.exp(-gamma * pairwise_distances(y, metric='sqeuclidean'))
    k_xy = np.exp(-gamma * pairwise_distances(x, y, metric='sqeuclidean'))
    n, m = len(x), len(y)
    term_x = (k_xx.sum() - np.trace(k_xx)) / max(n * (n - 1), 1)
    term_y = (k_yy.sum() - np.trace(k_yy)) / max(m * (m - 1), 1)
    value = term_x + term_y - 2.0 * k_xy.mean()
    return {'mmd2_unbiased': float(value), 'rbf_gamma': float(gamma), 'n_x': n, 'n_y': m}


def temporal_domain_auc(train_features, test_features, seed=0):
    n = min(len(train_features), len(test_features), MAX_SAMPLES)
    x = np.vstack([train_features[:n], test_features[:n]])
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    components = min(20, x.shape[1], len(x) - 1)
    model = make_pipeline(
        StandardScaler(), PCA(n_components=components, random_state=seed),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(model, x, y, cv=cv, method='predict_proba')[:, 1]
    return float(roc_auc_score(y, probabilities))


def _class_features(dataset, label, fs, seed):
    mask = dataset.labels == label
    return envelope_features(dataset.data[mask], fs, seed=seed)


def audit_xjtu():
    train, _, seen_test, unseen_test = get_temporal_dataloaders(
        './dataset/xjtu', batch_size=64, signal_len=1024, num_workers=0
    )
    drift = {}
    for label, name in enumerate(['Ball', 'Inner', 'Outer']):
        train_features = _class_features(train.dataset, label, 20480.0, SEED + label)
        test_features = _class_features(seen_test.dataset, label, 20480.0, SEED + 10 + label)
        drift[name] = {
            'mmd': rbf_mmd_unbiased(train_features, test_features, SEED + label),
            'domain_auc': temporal_domain_auc(train_features, test_features, SEED + label),
        }
    synthetic_path = os.path.join(
        ROOT, 'results_xjtu_strict', f'seed_{SEED}', 'synthetic_data.pt'
    )
    alignment = None
    if os.path.exists(synthetic_path):
        synthetic = torch.load(synthetic_path, map_location='cpu').tensors[0]
        syn_features = envelope_features(synthetic, 20480.0, seed=SEED)
        real_features = envelope_features(unseen_test.dataset.data, 20480.0, seed=SEED + 100)
        alignment = rbf_mmd_unbiased(syn_features, real_features, SEED)
    return {'temporal_drift': drift, 'synthetic_real_alignment': {'Mix': alignment}}


def audit_hust():
    root = './dataset/hust'
    stats = compute_seen_train_stats(root, ['N', 'B', 'I', 'O'])
    common = dict(
        root_dir=root, class_codes=['N', 'B', 'I', 'O'], labels=[0, 1, 2, 3],
        normalization_stats=stats, signal_len=1024, stride=1024,
    )
    train = HUSTTemporalDataset(split='train', **common)
    test = HUSTTemporalDataset(split='test', **common)
    drift = {}
    for label, name in enumerate(['N', 'B', 'I', 'O']):
        train_features = _class_features(train, label, 51200.0, SEED + label)
        test_features = _class_features(test, label, 51200.0, SEED + 10 + label)
        drift[name] = {
            'mmd': rbf_mmd_unbiased(train_features, test_features, SEED + label),
            'domain_auc': temporal_domain_auc(train_features, test_features, SEED + label),
        }
    alignment = {}
    for offset, task in enumerate(['IB', 'OB']):
        synthetic_path = os.path.join(
            ROOT, 'results_hust_generation_strict', f'seed_{SEED}', f'Exp_{task}',
            'synthetic_data.pt',
        )
        if not os.path.exists(synthetic_path):
            alignment[task] = None
            continue
        synthetic = torch.load(synthetic_path, map_location='cpu').tensors[0]
        real = HUSTTemporalDataset(
            root, [task], [4 + offset], 'test', stats, 1024, 1024, unseen=True
        )
        syn_features = envelope_features(synthetic, 51200.0, seed=SEED + offset)
        real_features = envelope_features(real.data, 51200.0, seed=SEED + 100 + offset)
        alignment[task] = rbf_mmd_unbiased(syn_features, real_features, SEED + offset)
    return {'temporal_drift': drift, 'synthetic_real_alignment': alignment}


def main():
    output_dir = os.path.join(ROOT, 'results_distribution_audit', f'seed_{SEED}')
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        'seed': SEED,
        'scope': 'post-training evaluation only; real unseen samples are not used for fitting',
        'feature': 'normalized log Hilbert-envelope power spectrum, 1024-point Hann-windowed rFFT',
        'xjtu': audit_xjtu(),
        'hust': audit_hust(),
    }
    path = os.path.join(output_dir, 'distribution_audit.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(path)


if __name__ == '__main__':
    main()
