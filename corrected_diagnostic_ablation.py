"""Corrected-protocol HUST diagnostic ablations and AWGN robustness.

This script is evaluation-only except for the unified-classifier baseline.  It
reuses the strict temporal split, the three completed PGCD generators, and the
two completed specialist checkpoints.  Every threshold is calibrated on seen
validation and held-out synthetic data; real IB/OB labels are used only for
the final report.
"""

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import accuracy_score
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, random_split

from hust_data import HUSTTemporalDataset, compute_seen_train_stats
from hust_gzsl_sp_model_confidence import (
    ConfidenceRectifier,
    RobustClassifier,
    calibrate_thresholds,
    distinctive_band_signature,
)


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results_corrected_ablation" / "hust_diagnostic"
SEEDS = (0, 1, 2)
SNRS_DB = tuple(range(10, -11, -2))
BATCH_SIZE = 64
MIN_CLUSTER_SIZE = 20
MIN_SAMPLES = 10
UNDEFINED = -1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_synthetic(path, global_label, seed):
    dataset = torch.load(path, map_location="cpu")
    if not isinstance(dataset, TensorDataset):
        raise TypeError(f"Expected TensorDataset: {path}")
    relabeled = TensorDataset(
        dataset.tensors[0],
        torch.full((len(dataset),), global_label, dtype=torch.long),
    )
    val_size = max(1, int(0.20 * len(relabeled)))
    train_size = len(relabeled) - val_size
    return random_split(
        relabeled,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )


def make_datasets():
    root = ROOT / "dataset" / "hust"
    seen_codes = ["N", "B", "I", "O"]
    stats = compute_seen_train_stats(str(root), seen_codes)
    common = dict(
        root_dir=str(root),
        class_codes=seen_codes,
        labels=[0, 1, 2, 3],
        normalization_stats=stats,
        signal_len=1024,
        stride=1024,
    )
    return {
        "seen_train": HUSTTemporalDataset(split="train", **common),
        "seen_val": HUSTTemporalDataset(split="val", **common),
        "seen_test": HUSTTemporalDataset(split="test", **common),
        "real_ib": HUSTTemporalDataset(
            str(root), ["IB"], [4], "test", stats, 1024, 1024, unseen=True
        ),
        "real_ob": HUSTTemporalDataset(
            str(root), ["OB"], [5], "test", stats, 1024, 1024, unseen=True
        ),
    }


def load_specialists(seed, device):
    models = [RobustClassifier(num_classes=5), RobustClassifier(num_classes=5)]
    paths = [
        ROOT / "results_hust_ensemble_strict" / f"seed_{seed}" / "Specialist_IB.pth",
        ROOT / "results_hust_ensemble_strict" / f"seed_{seed}" / "Specialist_OB.pth",
    ]
    for model, path in zip(models, paths):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval().to(device)
    return models


def collect(models, loader, device):
    features, targets, signatures = [], [], []
    probability_blocks = [[] for _ in models]
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            normalized_features = []
            for index, model in enumerate(models):
                logits, feature = model(x)
                probability_blocks[index].append(F.softmax(logits, dim=1).cpu().numpy())
                normalized_features.append(F.normalize(feature, p=2, dim=1))
            features.append(torch.cat(normalized_features, dim=1).cpu().numpy())
            signatures.append(distinctive_band_signature(x).cpu().numpy())
            targets.append(y.numpy())
    return (
        np.concatenate(features),
        np.concatenate(targets),
        [np.concatenate(block) for block in probability_blocks],
        np.concatenate(signatures),
    )


def metrics(targets, predictions):
    seen_mask = targets < 4
    unseen_mask = targets >= 4
    seen = accuracy_score(targets[seen_mask], predictions[seen_mask])
    unseen = accuracy_score(targets[unseen_mask], predictions[unseen_mask])
    h_score = 2 * seen * unseen / (seen + unseen + 1e-8)
    return {
        "seen_accuracy": float(seen),
        "unseen_accuracy": float(unseen),
        "h_score": float(h_score),
        "undefined_count": int(np.sum(predictions == UNDEFINED)),
        "undefined_rate": float(np.mean(predictions == UNDEFINED)),
    }


def map_clusters(
    algorithm,
    features,
    probabilities,
    signatures,
    tau_seen,
    tau_unseen,
    signature_prototypes,
    signature_thresholds,
    seed,
    use_signature=True,
    use_confidence=True,
    noise_fallback=True,
):
    seen_probabilities = np.mean([p[:, :4] for p in probabilities], axis=0)
    seen_confidence = seen_probabilities.max(axis=1)
    seen_prediction = seen_probabilities.argmax(axis=1)
    predictions = seen_prediction.copy()
    candidate_indices = np.flatnonzero(seen_confidence < tau_seen)
    predictions[candidate_indices] = UNDEFINED
    if len(candidate_indices) < MIN_CLUSTER_SIZE:
        return predictions

    if algorithm == "kmeans":
        cluster_labels = KMeans(n_clusters=2, n_init=20, random_state=seed).fit_predict(
            features[candidate_indices]
        )
    elif algorithm == "hdbscan":
        cluster_labels = HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=True,
        ).fit_predict(features[candidate_indices])
    else:
        raise ValueError(algorithm)

    tau_unseen = np.asarray(tau_unseen)
    signature_prototypes = np.asarray(signature_prototypes)
    signature_thresholds = np.asarray(signature_thresholds)
    for cluster_id in sorted(set(cluster_labels) - {UNDEFINED}):
        global_indices = candidate_indices[cluster_labels == cluster_id]
        confidence_vector = np.asarray([p[global_indices, 4].mean() for p in probabilities])
        cluster_signature = float(signatures[global_indices].mean())
        signature_distances = np.abs(cluster_signature - signature_prototypes)
        if use_signature:
            best_specialist = int(signature_distances.argmin())
        else:
            best_specialist = int(confidence_vector.argmax())
        supported = False
        if use_confidence:
            supported = supported or bool(confidence_vector[best_specialist] >= tau_unseen[best_specialist])
        if use_signature:
            supported = supported or bool(
                signature_distances[best_specialist] <= signature_thresholds[best_specialist]
            )
        if supported:
            predictions[global_indices] = 4 + best_specialist

    noise_indices = candidate_indices[cluster_labels == UNDEFINED]
    if len(noise_indices) and noise_fallback:
        signature_distances = np.abs(
            signatures[noise_indices, None] - signature_prototypes[None, :]
        )
        if use_signature:
            best_specialist = signature_distances.argmin(axis=1)
        else:
            point_confidences = np.stack([p[noise_indices, 4] for p in probabilities], axis=1)
            best_specialist = point_confidences.argmax(axis=1)
        point_confidences = np.stack([p[noise_indices, 4] for p in probabilities], axis=1)
        rows = np.arange(len(noise_indices))
        accepted = point_confidences[rows, best_specialist] >= tau_unseen[best_specialist]
        predictions[noise_indices] = seen_prediction[noise_indices]
        predictions[noise_indices[accepted]] = 4 + best_specialist[accepted]
    return predictions


def train_unified(seed, datasets, synthetic_trains, device, output_dir):
    model_path = output_dir / "unified_classifier.pth"
    model = RobustClassifier(num_classes=6).to(device)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model.eval()

    # Stage-specific reset makes the baseline reproducible independently of
    # how much random sampling a preceding ablation consumes.
    set_seed(seed + 10000)
    model = RobustClassifier(num_classes=6).to(device)
    training = ConcatDataset([datasets["seen_train"], *synthetic_trains])
    loader = DataLoader(
        training,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 20000),
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.train()
    for epoch in range(50):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"seed {seed} unified epoch {epoch + 1}/50")
    torch.save(model.state_dict(), model_path)
    return model.eval()


def evaluate_unified(model, loader, device):
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits, _ = model(x.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def add_awgn(dataset, snr_db, seed):
    x = dataset.data.clone()
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)
    signal_power = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    noise_power = noise.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noisy = x + noise * torch.sqrt(target_noise_power / noise_power)
    return TensorDataset(noisy, dataset.labels.clone())


def calibrate(seed, models, datasets, synthetic_validations, device):
    # The imported calibration helper reads these two module globals.
    import hust_gzsl_sp_model_confidence as confidence_module

    confidence_module.CONFIG["seed"] = seed
    confidence_module.CONFIG["batch_size"] = BATCH_SIZE
    confidence_module.CONFIG["min_cluster_size"] = MIN_CLUSTER_SIZE
    return calibrate_thresholds(
        models, datasets["seen_val"], synthetic_validations, device
    )


def run_seed(seed, datasets, device):
    print(f"\n=== corrected HUST diagnostic ablation: seed {seed} ===")
    set_seed(seed)
    output_dir = RESULT_ROOT / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_specialists(seed, device)

    synthetic_trains, synthetic_validations = [], []
    for global_label, task in ((4, "IB"), (5, "OB")):
        path = (
            ROOT / "results_hust_generation_strict" / f"seed_{seed}"
            / f"Exp_{task}" / "synthetic_data.pt"
        )
        train, validation = split_synthetic(path, global_label, seed)
        synthetic_trains.append(train)
        # Calibration expects the local specialist target index 4.
        synthetic_validations.append(
            TensorDataset(
                validation.dataset.tensors[0][validation.indices],
                torch.full((len(validation),), 4, dtype=torch.long),
            )
        )

    tau_seen, tau_unseen, prototypes, radii = calibrate(
        seed, models, datasets, synthetic_validations, device
    )
    test_dataset = ConcatDataset(
        [datasets["seen_test"], datasets["real_ib"], datasets["real_ob"]]
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    features, targets, probabilities, signatures = collect(models, test_loader, device)

    averaged_seen = np.mean([p[:, :4] for p in probabilities], axis=0)
    simple_scores = np.column_stack(
        [averaged_seen, probabilities[0][:, 4], probabilities[1][:, 4]]
    )
    variants = {
        "Probability averaging": metrics(targets, simple_scores.argmax(axis=1)),
        "K-Means + CRE": metrics(
            targets,
            map_clusters(
                "kmeans", features, probabilities, signatures, tau_seen,
                tau_unseen, prototypes, radii, seed,
                use_signature=True, use_confidence=True, noise_fallback=False,
            ),
        ),
        "HDBSCAN + confidence": metrics(
            targets,
            map_clusters(
                "hdbscan", features, probabilities, signatures, tau_seen,
                tau_unseen, prototypes, radii, seed,
                use_signature=False, use_confidence=True, noise_fallback=True,
            ),
        ),
        "CRE without noise fallback": metrics(
            targets,
            map_clusters(
                "hdbscan", features, probabilities, signatures, tau_seen,
                tau_unseen, prototypes, radii, seed,
                use_signature=True, use_confidence=True, noise_fallback=False,
            ),
        ),
        "Full PGCD-CRE": metrics(
            targets,
            map_clusters(
                "hdbscan", features, probabilities, signatures, tau_seen,
                tau_unseen, prototypes, radii, seed,
                use_signature=True, use_confidence=True, noise_fallback=True,
            ),
        ),
    }

    unified = train_unified(seed, datasets, synthetic_trains, device, output_dir)
    unified_targets, unified_predictions = evaluate_unified(unified, test_loader, device)
    variants = {"Unified classifier": metrics(unified_targets, unified_predictions), **variants}

    noise_rows = []
    for snr_db in SNRS_DB:
        noisy_parts = [
            add_awgn(datasets["seen_test"], snr_db, seed * 1000 + snr_db + 100),
            add_awgn(datasets["real_ib"], snr_db, seed * 1000 + snr_db + 200),
            add_awgn(datasets["real_ob"], snr_db, seed * 1000 + snr_db + 300),
        ]
        loader = DataLoader(ConcatDataset(noisy_parts), batch_size=BATCH_SIZE, shuffle=False)
        noisy_features, noisy_targets, noisy_probabilities, noisy_signatures = collect(
            models, loader, device
        )
        for method, algorithm in (("K-Means + CRE", "kmeans"), ("HDBSCAN-CRE", "hdbscan")):
            prediction = map_clusters(
                algorithm, noisy_features, noisy_probabilities, noisy_signatures,
                tau_seen, tau_unseen, prototypes, radii, seed,
                use_signature=True, use_confidence=True,
                noise_fallback=(algorithm == "hdbscan"),
            )
            row = metrics(noisy_targets, prediction)
            row.update({"seed": seed, "snr_db": snr_db, "method": method})
            noise_rows.append(row)
        print(f"seed {seed}: completed SNR {snr_db:+d} dB")

    payload = {
        "seed": seed,
        "protocol": "strict-temporal-60-20-20-before-windowing-v2",
        "calibration_scope": "seen validation plus held-out synthetic validation only",
        "tau_seen": tau_seen,
        "tau_unseen": list(map(float, tau_unseen)),
        "signature_prototypes": list(map(float, prototypes)),
        "signature_radii": list(map(float, radii)),
        "variants": variants,
        "noise_robustness": noise_rows,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def aggregate(payloads):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    variant_names = list(payloads[0]["variants"])
    summary = {"diagnostic_ablation": {}, "noise_robustness": []}
    for name in variant_names:
        summary["diagnostic_ablation"][name] = {}
        for metric_name in ("seen_accuracy", "unseen_accuracy", "h_score", "undefined_rate"):
            values = np.asarray([p["variants"][name][metric_name] for p in payloads])
            summary["diagnostic_ablation"][name][metric_name] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
                "values": values.tolist(),
            }

    for snr_db in SNRS_DB:
        for method in ("K-Means + CRE", "HDBSCAN-CRE"):
            rows = [
                row for payload in payloads for row in payload["noise_robustness"]
                if row["snr_db"] == snr_db and row["method"] == method
            ]
            aggregate_row = {"snr_db": snr_db, "method": method}
            for metric_name in ("seen_accuracy", "unseen_accuracy", "h_score", "undefined_rate"):
                values = np.asarray([row[metric_name] for row in rows])
                aggregate_row[f"{metric_name}_mean"] = float(values.mean())
                aggregate_row[f"{metric_name}_sample_std"] = float(values.std(ddof=1))
            summary["noise_robustness"].append(aggregate_row)

    with open(RESULT_ROOT / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(RESULT_ROOT / "diagnostic_ablation.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method", "seen_mean", "seen_std", "unseen_mean", "unseen_std",
            "h_mean", "h_std", "undefined_rate_mean", "undefined_rate_std",
        ])
        for name, row in summary["diagnostic_ablation"].items():
            writer.writerow([
                name,
                row["seen_accuracy"]["mean"], row["seen_accuracy"]["sample_std"],
                row["unseen_accuracy"]["mean"], row["unseen_accuracy"]["sample_std"],
                row["h_score"]["mean"], row["h_score"]["sample_std"],
                row["undefined_rate"]["mean"], row["undefined_rate"]["sample_std"],
            ])
    with open(RESULT_ROOT / "noise_robustness.csv", "w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary["noise_robustness"][0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["noise_robustness"])
    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = make_datasets()
    payloads = [run_seed(seed, datasets, device) for seed in SEEDS]
    summary = aggregate(payloads)
    print(json.dumps(summary["diagnostic_ablation"], indent=2))


if __name__ == "__main__":
    main()
