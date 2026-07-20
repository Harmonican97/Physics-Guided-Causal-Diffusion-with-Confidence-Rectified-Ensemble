import csv
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, random_split

try:
    from sklearn.cluster import HDBSCAN
except ImportError as exc:
    raise ImportError('HDBSCAN requires scikit-learn>=1.3.') from exc

from hust_data import HUSTTemporalDataset, compute_seen_train_stats
from causal_diffusion_model import PhysicsLoss


CONFIG = {
    'batch_size': int(os.environ.get('PGCD_BATCH_SIZE', '64')),
    'signal_len': 1024,
    'stride': 1024,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'root_dir': './dataset/hust',
    'save_dir': f"./results_hust_confidence_strict/seed_{os.environ.get('PGCD_SEED', '0')}",
    'model_ib_path': f"./results_hust_ensemble_strict/seed_{os.environ.get('PGCD_SEED', '0')}/Specialist_IB.pth",
    'model_ob_path': f"./results_hust_ensemble_strict/seed_{os.environ.get('PGCD_SEED', '0')}/Specialist_OB.pth",
    'syn_ib_path': f"./results_hust_generation_strict/seed_{os.environ.get('PGCD_SEED', '0')}/Exp_IB/synthetic_data.pt",
    'syn_ob_path': f"./results_hust_generation_strict/seed_{os.environ.get('PGCD_SEED', '0')}/Exp_OB/synthetic_data.pt",
    'synthetic_val_fraction': 0.20,
    'seed': int(os.environ.get('PGCD_SEED', '0')),
    'min_cluster_size': int(os.environ.get('PGCD_MIN_CLUSTER_SIZE', '20')),
    'min_samples': int(os.environ.get('PGCD_MIN_SAMPLES', '10')),
    'metric': 'euclidean',
    'cluster_selection_method': 'eom',
}
os.makedirs(CONFIG['save_dir'], exist_ok=True)

CLASSES = ['N', 'B', 'I', 'O', 'IB', 'OB']
UNDEFINED_LABEL = -1


def distinctive_band_signature(x, fs=51200.0, bandwidth_hz=50.0):
    """Log BPFI/BPFO envelope-energy ratio used for IB/OB semantics."""
    envelope = PhysicsLoss.analytic_envelope(x)
    envelope = envelope - envelope.mean(dim=-1, keepdim=True)
    window = torch.hann_window(
        x.shape[-1], periodic=True, device=x.device, dtype=x.dtype
    ).view(1, 1, -1)
    power = torch.abs(torch.fft.rfft(envelope * window, dim=-1, norm='ortho')).pow(2)
    frequencies = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs).to(x.device)
    inner = power[:, 0, torch.abs(frequencies - 148.2) <= bandwidth_hz].sum(dim=1)
    outer = power[:, 0, torch.abs(frequencies - 91.5) <= bandwidth_hz].sum(dim=1)
    return torch.log((inner + 1e-12) / (outer + 1e-12))


class RobustClassifier(nn.Module):
    """Four-layer 1D CNN specialist (128-dimensional pooled feature)."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 64, 2, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 32, 2, 16), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 16, 2, 8), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 8, 1, 4), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.features(x).squeeze(-1)
        return self.classifier(feat), feat


class ConfidenceRectifier:
    def __init__(
        self,
        specialists,
        target_global_labels,
        device,
        tau_seen,
        tau_unseen,
        signature_prototypes,
        signature_thresholds,
        hdbscan_params,
    ):
        self.specialists = [model.eval().to(device) for model in specialists]
        self.target_global_labels = np.asarray(target_global_labels, dtype=int)
        self.device = device
        self.tau_seen = float(tau_seen)
        self.tau_unseen = np.asarray(tau_unseen, dtype=float)
        self.signature_prototypes = np.asarray(signature_prototypes, dtype=float)
        self.signature_thresholds = np.asarray(signature_thresholds, dtype=float)
        self.hdbscan_params = dict(hdbscan_params)

    def _forward(self, loader):
        features, targets, signatures = [], [], []
        probability_blocks = [[] for _ in self.specialists]
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                normalized_features = []
                for model_index, model in enumerate(self.specialists):
                    logits, feature = model(x)
                    probability_blocks[model_index].append(
                        F.softmax(logits, dim=1).cpu().numpy()
                    )
                    normalized_features.append(F.normalize(feature, p=2, dim=1))
                features.append(torch.cat(normalized_features, dim=1).cpu().numpy())
                signatures.append(distinctive_band_signature(x).cpu().numpy())
                targets.append(y.numpy())
        return (
            np.concatenate(features),
            np.concatenate(targets),
            [np.concatenate(blocks) for blocks in probability_blocks],
            np.concatenate(signatures),
        )

    def predict(self, loader):
        fused_features, true_labels, probabilities, signatures = self._forward(loader)
        seen_probabilities = np.mean([p[:, :4] for p in probabilities], axis=0)
        seen_confidence = seen_probabilities.max(axis=1)
        seen_prediction = seen_probabilities.argmax(axis=1)

        final_prediction = np.full(len(true_labels), UNDEFINED_LABEL, dtype=int)
        seen_mask = seen_confidence >= self.tau_seen
        final_prediction[seen_mask] = seen_prediction[seen_mask]
        candidate_indices = np.flatnonzero(~seen_mask)

        details = {
            'tau_seen': self.tau_seen,
            'tau_unseen': self.tau_unseen.tolist(),
            'signature_prototypes': self.signature_prototypes.tolist(),
            'signature_thresholds': self.signature_thresholds.tolist(),
            'candidate_count': int(len(candidate_indices)),
            'clusters': [],
            'hdbscan_noise_count': 0,
            'noise_unseen_fallback_count': 0,
            'noise_seen_fallback_count': 0,
        }
        if len(candidate_indices) < self.hdbscan_params['min_cluster_size']:
            return final_prediction, true_labels, fused_features, details

        candidate_features = fused_features[candidate_indices]
        clusterer = HDBSCAN(**self.hdbscan_params)
        cluster_labels = clusterer.fit_predict(candidate_features)

        for cluster_id in sorted(set(cluster_labels) - {-1}):
            local_mask = cluster_labels == cluster_id
            global_indices = candidate_indices[local_mask]
            confidence_vector = np.asarray(
                [p[global_indices, 4].mean() for p in probabilities]
            )
            cluster_signature = float(signatures[global_indices].mean())
            signature_distances = np.abs(
                cluster_signature - self.signature_prototypes
            )
            best_specialist = int(signature_distances.argmin())
            best_confidence = float(confidence_vector[best_specialist])
            signature_supported = bool(
                signature_distances[best_specialist]
                <= self.signature_thresholds[best_specialist]
            )
            confidence_supported = bool(
                best_confidence >= self.tau_unseen[best_specialist]
            )
            assigned_label = UNDEFINED_LABEL
            if confidence_supported or signature_supported:
                assigned_label = int(self.target_global_labels[best_specialist])
                final_prediction[global_indices] = assigned_label
            details['clusters'].append(
                {
                    'cluster_id': int(cluster_id),
                    'size': int(local_mask.sum()),
                    'confidence': confidence_vector.tolist(),
                    'confidence_argmax': int(confidence_vector.argmax()),
                    'cluster_signature': cluster_signature,
                    'signature_distances': signature_distances.tolist(),
                    'confidence_supported': confidence_supported,
                    'signature_supported': signature_supported,
                    'assigned_label': assigned_label,
                }
            )

        noise_indices = candidate_indices[cluster_labels == -1]
        details['hdbscan_noise_count'] = int(len(noise_indices))
        if len(noise_indices):
            # Density noise is not automatically an unknown fault. Select the
            # closest candidate semantic by the validation-derived physical
            # signature, accept it only when its point confidence passes the
            # corresponding validation threshold, and otherwise return to the
            # strongest averaged seen prediction. This avoids inventing an
            # undefined label solely because HDBSCAN leaves a point unassigned.
            noise_signature_distances = np.abs(
                signatures[noise_indices, None]
                - self.signature_prototypes[None, :]
            )
            noise_best_specialist = noise_signature_distances.argmin(axis=1)
            noise_confidences = np.stack(
                [p[noise_indices, 4] for p in probabilities], axis=1
            )
            row_indices = np.arange(len(noise_indices))
            noise_is_unseen = (
                noise_confidences[row_indices, noise_best_specialist]
                >= self.tau_unseen[noise_best_specialist]
            )
            final_prediction[noise_indices] = seen_prediction[noise_indices]
            final_prediction[noise_indices[noise_is_unseen]] = (
                self.target_global_labels[noise_best_specialist[noise_is_unseen]]
            )
            details['noise_unseen_fallback_count'] = int(noise_is_unseen.sum())
            details['noise_seen_fallback_count'] = int((~noise_is_unseen).sum())
        return final_prediction, true_labels, fused_features, details


def _load_synthetic_validation(path):
    synthetic = torch.load(path, map_location='cpu')
    if not isinstance(synthetic, TensorDataset):
        raise TypeError(f'Expected TensorDataset in {path}')
    local_labels = torch.full((len(synthetic),), 4, dtype=torch.long)
    dataset = TensorDataset(synthetic.tensors[0], local_labels)
    val_size = max(1, int(CONFIG['synthetic_val_fraction'] * len(dataset)))
    train_size = len(dataset) - val_size
    _, validation = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['seed']),
    )
    return validation


def _balanced_threshold(negative, positive):
    """Threshold maximizing balanced calibration accuracy."""
    candidates = np.unique(np.concatenate([negative, positive]))
    scores = [
        (0.5 * (np.mean(negative < value) + np.mean(positive >= value)), value)
        for value in candidates
    ]
    return float(max(scores, key=lambda item: item[0])[1])


def calibrate_thresholds(specialists, seen_validation, synthetic_validations, device):
    """Calibrate thresholds without real unseen test samples or labels."""
    seen_loader = DataLoader(seen_validation, batch_size=CONFIG['batch_size'], shuffle=False)
    seen_confidences = []
    seen_candidate_confidences = [[] for _ in specialists]
    with torch.no_grad():
        for x, _ in seen_loader:
            x = x.to(device)
            seen_probs = []
            for model_index, model in enumerate(specialists):
                logits, _ = model(x)
                probabilities = F.softmax(logits, dim=1)
                seen_probs.append(probabilities[:, :4])
                seen_candidate_confidences[model_index].append(
                    probabilities[:, 4].cpu().numpy()
                )
            averaged = torch.stack(seen_probs).mean(dim=0)
            seen_confidences.append(averaged.max(dim=1).values.cpu().numpy())
    seen_confidences = np.concatenate(seen_confidences)
    tau_seen = float(
        np.clip(np.quantile(seen_confidences, 0.05), 0.50, 0.95)
    )

    unseen_thresholds = []
    signature_prototypes = []
    signature_thresholds = []
    rng = np.random.default_rng(CONFIG['seed'])
    with torch.no_grad():
        for model_index, (model, validation) in enumerate(
            zip(specialists, synthetic_validations)
        ):
            loader = DataLoader(validation, batch_size=CONFIG['batch_size'], shuffle=False)
            own_confidences, signature_values = [], []
            for x, _ in loader:
                x = x.to(device)
                logits, _ = model(x)
                own_confidences.append(F.softmax(logits, dim=1)[:, 4].cpu().numpy())
                signature_values.append(distinctive_band_signature(x).cpu().numpy())
            own_confidences = np.concatenate(own_confidences)
            seen_negative = np.concatenate(seen_candidate_confidences[model_index])
            unseen_thresholds.append(_balanced_threshold(seen_negative, own_confidences))

            signature_values = np.concatenate(signature_values)
            prototype = float(signature_values.mean())
            signature_prototypes.append(prototype)
            bootstrap_distances = []
            bootstrap_size = min(CONFIG['min_cluster_size'], len(signature_values))
            for _ in range(2000):
                sample = rng.choice(signature_values, bootstrap_size, replace=True)
                bootstrap_distances.append(abs(float(sample.mean()) - prototype))
            signature_thresholds.append(float(np.quantile(bootstrap_distances, 0.99)))
    return tau_seen, unseen_thresholds, signature_prototypes, signature_thresholds


def save_metrics(targets, predictions, details, runtime_seconds):
    seen_mask = targets < 4
    unseen_mask = targets >= 4
    acc_seen = accuracy_score(targets[seen_mask], predictions[seen_mask])
    acc_unseen = accuracy_score(targets[unseen_mask], predictions[unseen_mask])
    h_score = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen + 1e-8)

    report = classification_report(
        targets,
        predictions,
        labels=list(range(6)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    payload = {
        'seen_accuracy': acc_seen,
        'unseen_accuracy': acc_unseen,
        'h_score': h_score,
        'undefined_prediction_count': int(np.sum(predictions == UNDEFINED_LABEL)),
        'per_class': report,
        'rectification': details,
        'runtime_seconds': runtime_seconds,
        'config': CONFIG,
    }
    with open(os.path.join(CONFIG['save_dir'], 'metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)

    with open(
        os.path.join(CONFIG['save_dir'], 'per_class_metrics.csv'),
        'w', newline='', encoding='utf-8',
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(['class', 'precision', 'recall', 'f1-score', 'support'])
        for name in CLASSES:
            row = report[name]
            writer.writerow([name, row['precision'], row['recall'], row['f1-score'], row['support']])
    return acc_seen, acc_unseen, h_score


def main():
    started_at = time.perf_counter()
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = CONFIG['device']
    for path in (CONFIG['model_ib_path'], CONFIG['model_ob_path']):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    models = [RobustClassifier(num_classes=5), RobustClassifier(num_classes=5)]
    for model, path in zip(models, (CONFIG['model_ib_path'], CONFIG['model_ob_path'])):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval().to(device)

    root = CONFIG['root_dir']
    seen_codes = ['N', 'B', 'I', 'O']
    stats = compute_seen_train_stats(root, seen_codes)
    common = dict(
        root_dir=root,
        class_codes=seen_codes,
        labels=[0, 1, 2, 3],
        normalization_stats=stats,
        signal_len=CONFIG['signal_len'],
        stride=CONFIG['stride'],
    )
    seen_validation = HUSTTemporalDataset(split='val', **common)
    seen_test = HUSTTemporalDataset(split='test', **common)
    real_ib = HUSTTemporalDataset(
        root, ['IB'], [4], 'test', stats, CONFIG['signal_len'], CONFIG['stride'], unseen=True
    )
    real_ob = HUSTTemporalDataset(
        root, ['OB'], [5], 'test', stats, CONFIG['signal_len'], CONFIG['stride'], unseen=True
    )
    synthetic_validations = [
        _load_synthetic_validation(CONFIG['syn_ib_path']),
        _load_synthetic_validation(CONFIG['syn_ob_path']),
    ]
    tau_seen, tau_unseen, signature_prototypes, signature_thresholds = calibrate_thresholds(
        models, seen_validation, synthetic_validations, device
    )

    hdbscan_params = {
        'min_cluster_size': CONFIG['min_cluster_size'],
        'min_samples': CONFIG['min_samples'],
        'metric': CONFIG['metric'],
        'cluster_selection_method': CONFIG['cluster_selection_method'],
        'allow_single_cluster': True,
    }
    test_dataset = ConcatDataset([seen_test, real_ib, real_ob])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    rectifier = ConfidenceRectifier(
        models, [4, 5], device, tau_seen, tau_unseen,
        signature_prototypes, signature_thresholds, hdbscan_params
    )
    predictions, targets, features, details = rectifier.predict(test_loader)
    acc_seen, acc_unseen, h_score = save_metrics(
        targets, predictions, details, time.perf_counter() - started_at
    )
    print(
        f'tau_seen={tau_seen:.4f}, tau_unseen={tau_unseen}, '
        f'signature_prototypes={signature_prototypes}'
    )
    print(f'Seen Acc: {acc_seen * 100:.2f}%')
    print(f'Unseen Acc: {acc_unseen * 100:.2f}%')
    print(f'H-score: {h_score * 100:.2f}%')

    labels = list(range(6)) + [UNDEFINED_LABEL]
    cm = confusion_matrix(targets, predictions, labels=labels)
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES + ['Undefined'], yticklabels=CLASSES + ['Undefined'],
    )
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['save_dir'], 'conf_cm.png'), dpi=300)
    plt.close()

    if len(features) > 3000:
        rng = np.random.default_rng(CONFIG['seed'])
        indices = rng.choice(len(features), 3000, replace=False)
        features, targets, predictions = features[indices], targets[indices], predictions[indices]
    embedding = TSNE(n_components=2, perplexity=30, random_state=CONFIG['seed']).fit_transform(features)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=[CLASSES[t] for t in targets], ax=axes[0])
    prediction_names = [CLASSES[p] if p >= 0 else 'Undefined' for p in predictions]
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=prediction_names, ax=axes[1])
    axes[0].set_title('True labels')
    axes[1].set_title('HDBSCAN-CRE predictions')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['save_dir'], 'conf_tsne.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
