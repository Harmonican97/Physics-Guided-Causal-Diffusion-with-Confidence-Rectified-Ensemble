"""Corrected-protocol XJTU generation ablation and guidance-scale sweep.

The completed strict-temporal denoisers are reused for every variant.  The
downstream CNN, sample count, epoch schedule, validation calibration, temporal
split, and seed IDs are held fixed.  Real Mix windows are accessed only after
training for evaluation and post-training MMD reporting.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, random_split

from causal_diffusion_model import CompositionalUNet1D, PhysicsGuidedDiffusion
from data_loader import get_temporal_dataloaders
from distribution_audit import envelope_features, rbf_mmd_unbiased
from main_gzsl_xjtu import (
    FAULT_FREQUENCIES,
    FS,
    RobustClassifier,
    evaluate_gzsl,
    evaluate_transductive_gzsl,
)


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results_corrected_ablation" / "xjtu_generation"
SEEDS = (0, 1, 2)
GUIDANCE_SCALES = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0)
BATCH_SIZE = 32
CLASSIFIER_EPOCHS = 80
REFINE_STEPS = 120
N_SYNTHETIC = 1200
N_SENSITIVITY = 300
TRANSFER_WEIGHTS = [2.0, 1.5, 1.0]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_diffusion(seed, device):
    unet = CompositionalUNet1D(num_fault_components=3)
    diffusion = PhysicsGuidedDiffusion(
        unet,
        device=device,
        fs=FS,
        physics_bandwidth_hz=FS / 1024,
        physics_harmonics=3,
    )
    path = ROOT / "results_xjtu_strict" / f"seed_{seed}" / "compositional_diffusion.pth"
    diffusion.load_state_dict(torch.load(path, map_location=device))
    return diffusion.eval()


def class_pools(seen_loader):
    pools = {0: [], 1: [], 2: []}
    for x, y in seen_loader:
        for row, label in zip(x, y):
            pools[int(label)].append(row.clone())
    return {label: torch.stack(rows) for label, rows in pools.items()}


def build_coarse_components(pools, n_samples, seed):
    """Return atomic draws, weights, and the fixed raw-mixup baseline."""
    set_seed(seed)
    atomics = [[], [], []]
    weights = []
    n_balanced = int(n_samples * 0.3)
    n_dominant = int(n_samples * 0.3)
    for index in range(n_samples):
        for label in range(3):
            atomics[label].append(pools[label][np.random.randint(len(pools[label]))])
        if index < n_balanced:
            weight = np.random.dirichlet((5.0, 5.0, 5.0))
        elif index < n_balanced + n_dominant:
            weight = np.random.dirichlet((0.5, 0.5, 0.5))
            while np.max(weight) > 0.85 or np.max(weight) < 0.6:
                weight = np.random.dirichlet((0.5, 0.5, 0.5))
        else:
            main_index = np.random.randint(3)
            weight = np.array([0.03, 0.03, 0.03])
            weight[main_index] = 0.94
            weight += np.random.normal(0, 0.005, 3)
            weight = np.abs(weight) / np.sum(np.abs(weight))
        weights.append(weight)

    atomics = torch.stack([torch.stack(rows) for rows in atomics], dim=1)
    weights = torch.tensor(np.asarray(weights), dtype=atomics.dtype)
    raw = (atomics * weights[:, :, None, None]).sum(dim=1)
    # The legacy implementation adds a small acquisition-noise perturbation.
    levels = torch.empty(n_samples, 1, 1).uniform_(0.01, 0.05)
    raw = raw + torch.randn_like(raw) * levels
    raw = raw / (raw.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8)
    return atomics, weights, raw


def q_sample(diffusion, x0, timestep):
    t = torch.full((len(x0),), timestep, device=x0.device, dtype=torch.long)
    alpha_bar = diffusion.alphas_cumprod[t][:, None, None]
    return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * torch.randn_like(x0)


def refine_composition(diffusion, coarse, scale, component_weights, device):
    outputs = []
    for start in range(0, len(coarse), BATCH_SIZE):
        batch = coarse[start:start + BATCH_SIZE].to(device)
        target_hot = torch.ones(len(batch), 3, device=device)
        x_t = q_sample(diffusion, batch, REFINE_STEPS)
        if scale == 0:
            refined = diffusion.sample_from_t(x_t, REFINE_STEPS, target_hot)
        else:
            refined = diffusion.physics_guided_sample_from_t(
                x_t,
                REFINE_STEPS,
                target_multihot=target_hot,
                target_freqs_list=[FAULT_FREQUENCIES[3] for _ in range(len(batch))],
                target_weights_list=[component_weights for _ in range(len(batch))],
                guidance_scale=float(scale),
            )
        outputs.append(refined.detach().cpu())
    return torch.cat(outputs)


def refine_onehot_then_mix(diffusion, atomics, weights, device):
    """One-hot DDPM refinement followed by time-domain signal mixing.

    This variant has neither multi-hot conditioning nor a physical loss.  It
    uses the same learned denoiser and the same 120-step mix-and-refine budget,
    so the comparison does not introduce a second backbone or extra epochs.
    """
    outputs = []
    # A combined batch contains three atomic conditions.  Ten base samples
    # (30 model inputs) are faster than a memory-heavier batch of 60 inputs on
    # the 4-GB GTX 1050 Ti used for the corrected reruns.
    base_batch = int(os.environ.get("PGCD_ONEHOT_BASE_BATCH", "10"))
    onehot_template = torch.eye(3, device=device)
    for start in range(0, len(atomics), base_batch):
        atomic_batch = atomics[start:start + base_batch].to(device)
        count = len(atomic_batch)
        flattened = atomic_batch.transpose(0, 1).reshape(3 * count, 1, 1024)
        condition = onehot_template[:, None, :].expand(3, count, 3).reshape(3 * count, 3)
        x_t = q_sample(diffusion, flattened, REFINE_STEPS)
        refined = diffusion.sample_from_t(x_t, REFINE_STEPS, condition)
        refined = refined.reshape(3, count, 1, 1024).transpose(0, 1)
        mixed = (
            refined * weights[start:start + count].to(device)[:, :, None, None]
        ).sum(dim=1)
        mixed = mixed / (mixed.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8)
        outputs.append(mixed.detach().cpu())
    return torch.cat(outputs)


def as_dataset(signals):
    return TensorDataset(signals, torch.full((len(signals),), 3, dtype=torch.long))


def train_and_evaluate(seed, name, synthetic, loaders, device, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    classifier_path = output_dir / "classifier.pth"
    synthetic_path = output_dir / "synthetic_data.pt"
    if not synthetic_path.exists():
        torch.save(synthetic, synthetic_path)

    val_size = max(1, int(0.20 * len(synthetic)))
    train_size = len(synthetic) - val_size
    synthetic_train, synthetic_validation = random_split(
        synthetic,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    seen_train_loader, seen_val_loader, seen_test_loader, unseen_test_loader = loaders

    set_seed(seed + 10000)
    model = RobustClassifier(num_classes=4).to(device)
    if classifier_path.exists():
        model.load_state_dict(torch.load(classifier_path, map_location=device))
    else:
        training = ConcatDataset([seen_train_loader.dataset, synthetic_train])
        loader = DataLoader(
            training,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + 20000),
        )
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CLASSIFIER_EPOCHS)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        model.train()
        for epoch in range(CLASSIFIER_EPOCHS):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"seed {seed} {name}: classifier epoch {epoch + 1}/{CLASSIFIER_EPOCHS}")
        torch.save(model.state_dict(), classifier_path)

    model.eval()
    transductive = evaluate_transductive_gzsl(
        model, seen_val_loader, seen_test_loader, unseen_test_loader, device
    )
    pointwise = evaluate_gzsl(model, seen_test_loader, unseen_test_loader, device)
    payload = {
        "seed": seed,
        "variant": name,
        "synthetic_sample_count": len(synthetic),
        "transductive": {
            "seen_accuracy": float(transductive[0]),
            "unseen_accuracy": float(transductive[1]),
            "h_score": float(transductive[2]),
            "rectification": transductive[6],
        },
        "pointwise": {
            "seen_accuracy": float(pointwise[0]),
            "unseen_accuracy": float(pointwise[1]),
            "h_score": float(pointwise[2]),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def add_mmd(payload, synthetic, real_unseen, seed):
    synthetic_features = envelope_features(synthetic.tensors[0], FS, seed=seed)
    real_features = envelope_features(real_unseen.dataset.data, FS, seed=seed + 100)
    payload["synthetic_real_mmd"] = rbf_mmd_unbiased(
        synthetic_features, real_features, seed
    )
    return payload


def official_full_payload(seed, loaders, device):
    base = ROOT / "results_xjtu_strict" / f"seed_{seed}"
    with open(base / "metrics.json", encoding="utf-8") as handle:
        official = json.load(handle)
    synthetic = torch.load(base / "synthetic_data.pt", map_location="cpu")
    model = RobustClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load(base / "classifier.pth", map_location=device))
    model.eval()
    pointwise = evaluate_gzsl(model, loaders[2], loaders[3], device)
    payload = {
        "seed": seed,
        "variant": "Transfer-weighted PGCD",
        "synthetic_sample_count": len(synthetic),
        "transductive": {
            "seen_accuracy": float(official["seen_accuracy"]),
            "unseen_accuracy": float(official["unseen_accuracy"]),
            "h_score": float(official["h_score"]),
            "rectification": official.get("rectification", {}),
        },
        "pointwise": {
            "seen_accuracy": float(pointwise[0]),
            "unseen_accuracy": float(pointwise[1]),
            "h_score": float(pointwise[2]),
        },
    }
    return payload, synthetic


def run_ablation_seed(seed, loaders, pools, device):
    print(f"\n=== corrected XJTU generation ablation: seed {seed} ===")
    seed_dir = RESULT_ROOT / "ablation" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    diffusion = load_diffusion(seed, device)
    components_path = seed_dir / "coarse_components.pt"
    if components_path.exists():
        bundle = torch.load(components_path, map_location="cpu")
        atomics, weights, raw = bundle["atomics"], bundle["weights"], bundle["raw"]
    else:
        atomics, weights, raw = build_coarse_components(pools, N_SYNTHETIC, seed + 30000)
        torch.save({"atomics": atomics, "weights": weights, "raw": raw}, components_path)

    variants = []
    raw_dataset = as_dataset(raw)
    variants.append(
        add_mmd(
            train_and_evaluate(
                seed, "Raw signal mixup", raw_dataset, loaders, device,
                seed_dir / "raw_signal_mixup",
            ),
            raw_dataset, loaders[3], seed,
        )
    )

    generation_specs = [
        ("One-hot DDPM + mix", "onehot_ddpm_mix"),
        ("Compositional DDPM (s=0)", "compositional_ddpm"),
        ("Unweighted-physics PGCD", "unweighted_pgcd"),
    ]
    for name, slug in generation_specs:
        output_dir = seed_dir / slug
        synthetic_path = output_dir / "synthetic_data.pt"
        if synthetic_path.exists():
            synthetic = torch.load(synthetic_path, map_location="cpu")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            set_seed(seed + 40000)
            if slug == "onehot_ddpm_mix":
                signals = refine_onehot_then_mix(diffusion, atomics, weights, device)
            elif slug == "compositional_ddpm":
                signals = refine_composition(diffusion, raw, 0.0, TRANSFER_WEIGHTS, device)
            else:
                signals = refine_composition(diffusion, raw, 10.0, [1.0, 1.0, 1.0], device)
            synthetic = as_dataset(signals)
            torch.save(synthetic, synthetic_path)
        payload = train_and_evaluate(seed, name, synthetic, loaders, device, output_dir)
        variants.append(add_mmd(payload, synthetic, loaders[3], seed))

    full_payload, full_synthetic = official_full_payload(seed, loaders, device)
    variants.append(add_mmd(full_payload, full_synthetic, loaders[3], seed))
    with open(seed_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(variants, handle, indent=2)
    return variants


def scale_slug(scale):
    return f"s_{scale:g}".replace(".", "p")


def run_sensitivity_seed(seed, loaders, pools, device):
    print(f"\n=== corrected guidance-scale sensitivity: seed {seed} ===")
    seed_dir = RESULT_ROOT / "sensitivity" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    diffusion = load_diffusion(seed, device)
    coarse_path = seed_dir / "paired_coarse.pt"
    if coarse_path.exists():
        raw = torch.load(coarse_path, map_location="cpu")
    else:
        _, _, raw = build_coarse_components(pools, N_SENSITIVITY, seed + 50000)
        torch.save(raw, coarse_path)

    rows = []
    for scale in GUIDANCE_SCALES:
        output_dir = seed_dir / scale_slug(scale)
        synthetic_path = output_dir / "synthetic_data.pt"
        if synthetic_path.exists():
            synthetic = torch.load(synthetic_path, map_location="cpu")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            # All scales share the same initial coarse samples and reverse-noise
            # stream, giving a paired sensitivity comparison.
            set_seed(seed + 60000)
            signals = refine_composition(
                diffusion, raw, scale, TRANSFER_WEIGHTS, device
            )
            synthetic = as_dataset(signals)
            torch.save(synthetic, synthetic_path)
        payload = train_and_evaluate(
            seed, f"guidance scale {scale:g}", synthetic, loaders, device, output_dir
        )
        payload["guidance_scale"] = scale
        rows.append(add_mmd(payload, synthetic, loaders[3], seed))
        print(f"seed {seed}: completed guidance scale {scale:g}")
    with open(seed_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    return rows


def aggregate(records, group_key, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = []
    seen = []
    for seed_rows in records:
        for row in seed_rows:
            key = row[group_key]
            if key not in seen:
                groups.append(key)
                seen.append(key)
    summary = []
    for key in groups:
        rows = [row for seed_rows in records for row in seed_rows if row[group_key] == key]
        aggregate_row = {group_key: key, "n_seeds": len(rows)}
        for branch in ("transductive", "pointwise"):
            for metric_name in ("seen_accuracy", "unseen_accuracy", "h_score"):
                values = np.asarray([row[branch][metric_name] for row in rows])
                aggregate_row[f"{branch}_{metric_name}_mean"] = float(values.mean())
                aggregate_row[f"{branch}_{metric_name}_sample_std"] = float(values.std(ddof=1))
        mmd_values = np.asarray([row["synthetic_real_mmd"]["mmd2_unbiased"] for row in rows])
        aggregate_row["mmd2_mean"] = float(mmd_values.mean())
        aggregate_row["mmd2_sample_std"] = float(mmd_values.std(ddof=1))
        summary.append(aggregate_row)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ablation", "sensitivity", "all"), default="all")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_temporal_dataloaders(
        str(ROOT / "dataset" / "xjtu"),
        batch_size=BATCH_SIZE,
        signal_len=1024,
        num_workers=0,
    )
    pools = class_pools(loaders[0])
    if args.mode in ("ablation", "all"):
        ablation = [run_ablation_seed(seed, loaders, pools, device) for seed in args.seeds]
        aggregate(ablation, "variant", RESULT_ROOT / "ablation")
    if args.mode in ("sensitivity", "all"):
        sensitivity = [run_sensitivity_seed(seed, loaders, pools, device) for seed in args.seeds]
        aggregate(sensitivity, "guidance_scale", RESULT_ROOT / "sensitivity")


if __name__ == "__main__":
    main()
