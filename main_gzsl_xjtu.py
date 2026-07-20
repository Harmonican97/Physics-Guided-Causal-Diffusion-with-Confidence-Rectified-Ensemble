import csv
import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import HDBSCAN

from causal_diffusion_model import CompositionalUNet1D, PhysicsGuidedDiffusion
from data_loader import get_temporal_dataloaders

FS = 20480 
INPUT_SPEED_RPM = 1800
INPUT_FREQ = 30.0

Z_SUN, Z_RING, Z_PLANET = 28, 100, 36
FREQ_CARRIER = INPUT_FREQ * (Z_SUN / (Z_SUN + Z_RING)) 
FREQ_PLANET_REL = (INPUT_FREQ - FREQ_CARRIER) * (Z_SUN / Z_PLANET) 

COEFFS = {'Ball': 2.3, 'Inner': 5.4, 'Outer': 3.6}

FAULT_FREQUENCIES = {
    0: [COEFFS['Ball'] * FREQ_PLANET_REL], 
    1: [COEFFS['Inner'] * FREQ_PLANET_REL],
    2: [COEFFS['Outer'] * FREQ_PLANET_REL],
    3: [COEFFS['Ball'] * FREQ_PLANET_REL, COEFFS['Inner'] * FREQ_PLANET_REL, COEFFS['Outer'] * FREQ_PLANET_REL]
}

# Fixed, pre-training transfer-path coefficients in [Ball, Inner, Outer] order.
# They are normalized inside PhysicsLoss for each active composition.
TRANSFER_PATH_WEIGHTS = {
    0: [2.0],
    1: [1.5],
    2: [1.0],
    3: [2.0, 1.5, 1.0],
}

def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


CONFIG = {
    'epoch_diffusion': _env_int('PGCD_DIFFUSION_EPOCHS', 200),
    'epoch_classifier': _env_int('PGCD_CLASSIFIER_EPOCHS', 80),
    'batch_size': _env_int('PGCD_BATCH_SIZE', 32),
    'lr_diffusion': 1e-4,
    'lr_classifier': 5e-4,
    'signal_len': 1024,
    'num_fault_components': 3,
    'num_classes_output': 4,
    'unseen_class': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': int(os.environ.get('PGCD_SEED', '0')),
    'save_dir': f"./results_xjtu_strict/seed_{os.environ.get('PGCD_SEED', '0')}",
    
    'guidance_scale': _env_float('PGCD_GUIDANCE_SCALE', 10.0),
    'n_synthetic_samples': _env_int('PGCD_SYNTHETIC_SAMPLES', 1200),
    'refine_timestep': _env_int('PGCD_REFINE_STEPS', 120),
    'checkpoint_interval': _env_int('PGCD_CHECKPOINT_INTERVAL', 10),
    'synthetic_val_fraction': _env_float('PGCD_SYNTHETIC_VAL_FRACTION', 0.20),
}
os.makedirs(CONFIG['save_dir'], exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RobustClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 64, 2, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 32, 2, 16), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 16, 2, 8), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 8, 1, 4), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)

    def extract_features(self, x):
        return self.features(x).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        else: return focal_loss.sum()

def create_stratified_mix_dataset(seen_loader, n_samples=1000):
    print("\n>>> Coarse compositional mixing with fixed protocol...")
    data_pool = {0: [], 1: [], 2: []}
    for x, y in seen_loader:
        for i in range(len(y)):
            label = y[i].item()
            if label in data_pool:
                data_pool[label].append(x[i])
                
    if any(len(v)==0 for v in data_pool.values()): raise ValueError("Data error")

    mixed_data = []
    max_weights = [] 
    
    n_balanced = int(n_samples * 0.3)
    n_dominant = int(n_samples * 0.3)
    
    for i in range(n_samples):
        s0 = data_pool[0][np.random.randint(len(data_pool[0]))]
        s1 = data_pool[1][np.random.randint(len(data_pool[1]))]
        s2 = data_pool[2][np.random.randint(len(data_pool[2]))]
        
        if i < n_balanced:
            w = np.random.dirichlet((5.0, 5.0, 5.0)) # 均衡
        elif i < n_balanced + n_dominant:
            w = np.random.dirichlet((0.5, 0.5, 0.5))
            while np.max(w) > 0.85 or np.max(w) < 0.6: w = np.random.dirichlet((0.5, 0.5, 0.5))
        else:
            main_idx = np.random.randint(3)
            w = np.array([0.03, 0.03, 0.03]) 
            w[main_idx] = 0.94
            w += np.random.normal(0, 0.005, 3)
            w = np.abs(w) / np.sum(np.abs(w))
            
        max_w = np.max(w)
        max_weights.append(max_w)
        
        mixed = w[0]*s0 + w[1]*s1 + w[2]*s2
        
        noise_level = np.random.uniform(0.01, 0.05)
        mixed = mixed + torch.randn_like(mixed) * noise_level
        
        rms = torch.sqrt(torch.mean(mixed**2))
        mixed_norm = mixed / (rms + 1e-8)
        
        mixed_data.append(mixed_norm) 
        
    return torch.stack(mixed_data), torch.tensor(max_weights)


def generate_refined_mix_data(diffusion, seen_loader, n_samples, target_label, device):
    diffusion.eval()
    print(f"\n>>> Mix-and-Refine (fixed physics guidance)...")
    
    coarse_data, _ = create_stratified_mix_dataset(seen_loader, n_samples)
    
    refined_data = []
    refined_labels = []
    
    batch_size = 32
    target_freqs = FAULT_FREQUENCIES[target_label]
    target_weights = TRANSFER_PATH_WEIGHTS[target_label]
    t_refine = CONFIG['refine_timestep']
    
    for i in tqdm(range(0, n_samples, batch_size)):
        x0_coarse = coarse_data[i : i+batch_size].to(device)
        current_bs = x0_coarse.size(0)
        
        # The guidance scale is fixed before test-time evaluation and selected
        # using the seen/synthetic validation protocol.
        scales = torch.full(
            (current_bs, 1, 1), CONFIG['guidance_scale'], device=device
        )
        
        target_hot = torch.ones(current_bs, 3, device=device) 
        freqs_batch = [target_freqs for _ in range(current_bs)]
        weights_batch = [target_weights for _ in range(current_bs)]
        
        t = torch.full((current_bs,), t_refine, device=device, dtype=torch.long)
        noise = torch.randn_like(x0_coarse)
        alpha_bar = diffusion.alphas_cumprod[t][:, None, None]
        x_t = torch.sqrt(alpha_bar) * x0_coarse + torch.sqrt(1 - alpha_bar) * noise
        
        with torch.enable_grad():
            x_refined = diffusion.physics_guided_sample_from_t(
                x_t, t_refine,
                target_multihot=target_hot,
                target_freqs_list=freqs_batch,
                target_weights_list=weights_batch,
                guidance_scale=scales
            )
        
        refined_data.append(x_refined.detach().cpu())
        refined_labels.append(torch.full((current_bs,), target_label).long())
        
    return TensorDataset(torch.cat(refined_data), torch.cat(refined_labels))

def train_classifier_robust(model, train_loader, optimizer, epochs, device):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Cls Epoch {epoch+1}", unit="batch")
        total_acc = 0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            pred = out.argmax(dim=1)
            acc = (pred == y).float().sum().item()
            total_acc += acc
            pbar.set_postfix({'L': f"{loss.item():.4f}", 'A': f"{acc/x.size(0):.2f}"})
        scheduler.step()

def evaluate_gzsl(model, seen_test_loader, unseen_test_loader, device, unseen_logit_bias=0.0):
    model.eval()
    
    def get_preds(loader):
        preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                logits[:, CONFIG['unseen_class']] += unseen_logit_bias
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        return preds, targets

    seen_p, seen_t = get_preds(seen_test_loader)
    unseen_p, unseen_t = get_preds(unseen_test_loader)
    
    acc_seen = accuracy_score(seen_t, seen_p)
    acc_unseen = accuracy_score(unseen_t, unseen_p)
    h_score = (2 * acc_seen * acc_unseen) / (acc_seen + acc_unseen + 1e-8)
    cm = confusion_matrix(seen_t + unseen_t, seen_p + unseen_p)
    return acc_seen, acc_unseen, h_score, cm, np.asarray(seen_t + unseen_t), np.asarray(seen_p + unseen_p)


def calibrate_unseen_logit_bias(model, seen_validation_loader, synthetic_validation, device):
    """Calibrate GZSL stacking bias without real unseen test samples."""
    model.eval()

    def collect(loader):
        logits, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                logits.append(model(x.to(device)).cpu().numpy())
                targets.append(y.numpy())
        return np.concatenate(logits), np.concatenate(targets)

    seen_logits, seen_targets = collect(seen_validation_loader)
    synthetic_loader = DataLoader(
        synthetic_validation, batch_size=CONFIG['batch_size'], shuffle=False
    )
    synthetic_logits, synthetic_targets = collect(synthetic_loader)
    best = None
    for bias in np.linspace(-3.0, 3.0, 1201):
        seen_adjusted = seen_logits.copy()
        synthetic_adjusted = synthetic_logits.copy()
        seen_adjusted[:, CONFIG['unseen_class']] += bias
        synthetic_adjusted[:, CONFIG['unseen_class']] += bias
        seen_accuracy = accuracy_score(seen_targets, seen_adjusted.argmax(axis=1))
        synthetic_accuracy = accuracy_score(
            synthetic_targets, synthetic_adjusted.argmax(axis=1)
        )
        h_score = (
            2 * seen_accuracy * synthetic_accuracy
            / (seen_accuracy + synthetic_accuracy + 1e-8)
        )
        candidate = (h_score, -abs(bias), bias, seen_accuracy, synthetic_accuracy)
        if best is None or candidate > best:
            best = candidate
    return {
        'unseen_logit_bias': float(best[2]),
        'seen_validation_accuracy': float(best[3]),
        'synthetic_validation_accuracy': float(best[4]),
        'validation_h_score': float(best[0]),
    }


def evaluate_transductive_gzsl(
    model, seen_validation_loader, seen_test_loader, unseen_test_loader, device
):
    """Single-candidate HDBSCAN-CRE evaluation with validation-only thresholds."""
    model.eval()

    def collect(loader):
        probabilities, features, targets = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                feature = model.extract_features(x)
                logits = model.classifier(feature)
                probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
                features.append(F.normalize(feature, p=2, dim=1).cpu().numpy())
                targets.append(y.numpy())
        return np.concatenate(probabilities), np.concatenate(features), np.concatenate(targets)

    validation_probabilities, _, _ = collect(seen_validation_loader)
    seen_validation_confidence = validation_probabilities[:, :3].max(axis=1)
    tau_seen = float(
        np.clip(np.quantile(seen_validation_confidence, 0.05), 0.50, 0.95)
    )
    # A 1% validation false-positive operating point supports a candidate
    # cluster despite synthetic/real softmax shift.
    tau_unseen = float(np.quantile(validation_probabilities[:, 3], 0.99))

    test_dataset = ConcatDataset(
        [seen_test_loader.dataset, unseen_test_loader.dataset]
    )
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    probabilities, features, targets = collect(test_loader)
    seen_confidence = probabilities[:, :3].max(axis=1)
    predictions = probabilities[:, :3].argmax(axis=1)
    candidate_indices = np.flatnonzero(seen_confidence < tau_seen)
    predictions[candidate_indices] = -1

    details = {
        'tau_seen': tau_seen,
        'tau_unseen_cluster': tau_unseen,
        'candidate_count': int(len(candidate_indices)),
        'clusters': [],
        'hdbscan_noise_count': 0,
        'noise_unseen_fallback_count': 0,
        'noise_seen_fallback_count': 0,
    }
    if len(candidate_indices) >= 20:
        cluster_labels = HDBSCAN(
            min_cluster_size=20,
            min_samples=10,
            metric='euclidean',
            cluster_selection_method='eom',
            allow_single_cluster=True,
        ).fit_predict(features[candidate_indices])
        for cluster_id in sorted(set(cluster_labels) - {-1}):
            global_indices = candidate_indices[cluster_labels == cluster_id]
            candidate_confidence = float(probabilities[global_indices, 3].mean())
            if candidate_confidence >= tau_unseen:
                assigned_label = CONFIG['unseen_class']
            else:
                # Unsupported clusters are rectified back to a coherent seen
                # class when their aggregate seen evidence is stronger.
                assigned_label = int(
                    probabilities[global_indices, :3].mean(axis=0).argmax()
                )
            predictions[global_indices] = assigned_label
            details['clusters'].append({
                'cluster_id': int(cluster_id),
                'size': int(len(global_indices)),
                'candidate_confidence': candidate_confidence,
                'assigned_label': assigned_label,
            })
        noise_indices = candidate_indices[cluster_labels == -1]
        details['hdbscan_noise_count'] = int(len(noise_indices))
        if len(noise_indices):
            # HDBSCAN deliberately leaves low-density candidates unassigned.
            # Resolve them with the same validation-only 1% false-positive
            # operating point used for cluster support, so the evaluator never
            # converts density noise into an artificial "undefined" class.
            noise_is_unseen = probabilities[noise_indices, 3] >= tau_unseen
            predictions[noise_indices] = np.where(
                noise_is_unseen,
                CONFIG['unseen_class'],
                probabilities[noise_indices, :3].argmax(axis=1),
            )
            details['noise_unseen_fallback_count'] = int(noise_is_unseen.sum())
            details['noise_seen_fallback_count'] = int((~noise_is_unseen).sum())

    seen_mask = targets < CONFIG['unseen_class']
    unseen_mask = targets == CONFIG['unseen_class']
    acc_seen = accuracy_score(targets[seen_mask], predictions[seen_mask])
    acc_unseen = accuracy_score(targets[unseen_mask], predictions[unseen_mask])
    h_score = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen + 1e-8)
    cm = confusion_matrix(targets, predictions, labels=[0, 1, 2, 3])
    return acc_seen, acc_unseen, h_score, cm, targets, predictions, details

def label_to_multihot(labels, device):
    B = labels.size(0)
    multihot = torch.zeros(B, 3, device=device)
    mask_ball = (labels == 0) | (labels == 3)
    multihot[mask_ball, 0] = 1.0
    mask_inner = (labels == 1) | (labels == 3)
    multihot[mask_inner, 1] = 1.0
    mask_outer = (labels == 2) | (labels == 3)
    multihot[mask_outer, 2] = 1.0
    return multihot

def train_diffusion_model(diffusion, train_loader, optimizer, epochs, device):
    diffusion.train()
    print(f"\n>>> Diffusion training...")
    state_path = os.path.join(CONFIG['save_dir'], 'diffusion_training_state.pt')
    start_epoch = 0
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        diffusion.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = int(state['epoch'])
        print(f">>> Resuming diffusion at epoch {start_epoch + 1}/{epochs}")
    for epoch in range(start_epoch, epochs):
        pbar = tqdm(train_loader, desc=f"Diff Epoch {epoch+1}", unit="batch")
        total_loss = 0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            y_hot = label_to_multihot(y, device)
            
            optimizer.zero_grad()
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device).long()
            loss = diffusion.forward_loss(x, t, y_hot)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'L': f"{loss.item():.4f}"})
        if (epoch + 1) % CONFIG['checkpoint_interval'] == 0 or epoch + 1 == epochs:
            torch.save(
                {'epoch': epoch + 1, 'model': diffusion.state_dict(), 'optimizer': optimizer.state_dict()},
                state_path,
            )
    
    torch.save(diffusion.state_dict(), os.path.join(CONFIG['save_dir'], 'compositional_diffusion.pth'))

def main():
    started_at = time.perf_counter()
    set_seed(CONFIG['seed'])
    device = CONFIG['device']
    print(f"Device: {device}")
    
    # 1. Load Data
    seen_train_loader, seen_val_loader, seen_test_loader, unseen_test_loader = get_temporal_dataloaders(
        root_dir='./dataset/xjtu', batch_size=CONFIG['batch_size'], signal_len=CONFIG['signal_len'], num_workers=0
    )
    seen_train = seen_train_loader.dataset
    print(
        'Strict temporal counts (train/val/seen-test/unseen-test):',
        len(seen_train_loader.dataset), len(seen_val_loader.dataset),
        len(seen_test_loader.dataset), len(unseen_test_loader.dataset),
    )
    
    # 2. Train Diffusion
    unet = CompositionalUNet1D(num_fault_components=CONFIG['num_fault_components'])
    diffusion = PhysicsGuidedDiffusion(
        unet,
        device=device,
        fs=FS,
        physics_bandwidth_hz=FS / CONFIG['signal_len'],
        physics_harmonics=3,
    )
    
    checkpoint = os.path.join(CONFIG['save_dir'], 'compositional_diffusion.pth')
    legacy_checkpoint = os.path.join(CONFIG['save_dir'], 'causal_diffusion.pth')
    training_state_path = os.path.join(CONFIG['save_dir'], 'diffusion_training_state.pt')
    completed_epochs = 0
    if os.path.exists(training_state_path):
        completed_epochs = int(torch.load(training_state_path, map_location='cpu')['epoch'])
    if (os.path.exists(checkpoint) or os.path.exists(legacy_checkpoint)) and completed_epochs >= CONFIG['epoch_diffusion']:
        print("Load diffusion model...")
        diffusion.load_state_dict(torch.load(
            checkpoint if os.path.exists(checkpoint) else legacy_checkpoint, map_location=device
        ))
    else:
        train_diffusion_model(diffusion, seen_train_loader, optim.Adam(diffusion.parameters(), lr=CONFIG['lr_diffusion']), 
                            CONFIG['epoch_diffusion'], device)    
    
    synthetic_path = os.path.join(CONFIG['save_dir'], 'synthetic_data.pt')
    if os.path.exists(synthetic_path):
        synthetic_ds = torch.load(synthetic_path, map_location='cpu')
    else:
        synthetic_ds = generate_refined_mix_data(
            diffusion, seen_train_loader, CONFIG['n_synthetic_samples'], CONFIG['unseen_class'], device
        )
        torch.save(synthetic_ds, synthetic_path)
    synthetic_val_size = max(1, int(CONFIG['synthetic_val_fraction'] * len(synthetic_ds)))
    synthetic_train_size = len(synthetic_ds) - synthetic_val_size
    synthetic_train, synthetic_validation = random_split(
        synthetic_ds,
        [synthetic_train_size, synthetic_val_size],
        generator=torch.Generator().manual_seed(CONFIG['seed']),
    )

    classifier = RobustClassifier(CONFIG['num_classes_output']).to(device)
    if os.path.exists(os.path.join(CONFIG['save_dir'], 'classifier.pth')):
        print("Load classifier...")
        classifier.load_state_dict(torch.load(
            os.path.join(CONFIG['save_dir'], 'classifier.pth'), map_location=device
        ))
    else:
        full_dataset = ConcatDataset([seen_train, synthetic_train])
        combined_loader = DataLoader(full_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
        train_classifier_robust(classifier, combined_loader, 
                              optim.Adam(classifier.parameters(), lr=CONFIG['lr_classifier'], weight_decay=1e-4), 
                              CONFIG['epoch_classifier'], device)
        torch.save(classifier.state_dict(), os.path.join(CONFIG['save_dir'], 'classifier.pth'))

    calibration = calibrate_unseen_logit_bias(
        classifier, seen_val_loader, synthetic_validation, device
    )
    with open(os.path.join(CONFIG['save_dir'], 'calibration.json'), 'w', encoding='utf-8') as handle:
        json.dump(calibration, handle, indent=2)

    acc_s, acc_u, h, cm, targets, predictions, rectification = evaluate_transductive_gzsl(
        classifier, seen_val_loader, seen_test_loader, unseen_test_loader, device
    )
    print(f"\nFinal Results:\nSeen Acc: {acc_s:.4f}\nUnseen Acc: {acc_u:.4f}\nH-score: {h:.4f}")
    
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ball','In','Out','Mix'], yticklabels=['Ball','In','Out','Mix'])
    plt.savefig(os.path.join(CONFIG['save_dir'], 'confusion_matrix.png'))
    plt.close()

    report = classification_report(
        targets, predictions, labels=[0, 1, 2, 3],
        target_names=['Ball', 'Inner', 'Outer', 'Mix'], output_dict=True, zero_division=0,
    )
    payload = {
        'seed': CONFIG['seed'],
        'protocol': 'strict-temporal-60-20-20-before-windowing-v2',
        'seen_accuracy': acc_s,
        'unseen_accuracy': acc_u,
        'h_score': h,
        'calibration': calibration,
        'rectification': rectification,
        'counts': {
            'train': len(seen_train_loader.dataset),
            'validation': len(seen_val_loader.dataset),
            'seen_test': len(seen_test_loader.dataset),
            'unseen_test': len(unseen_test_loader.dataset),
        },
        'runtime_seconds': time.perf_counter() - started_at,
        'config': CONFIG,
        'per_class': report,
    }
    with open(os.path.join(CONFIG['save_dir'], 'metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    with open(
        os.path.join(CONFIG['save_dir'], 'per_class_metrics.csv'), 'w', newline='', encoding='utf-8'
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(['class', 'precision', 'recall', 'f1-score', 'support'])
        for name in ['Ball', 'Inner', 'Outer', 'Mix']:
            row = report[name]
            writer.writerow([name, row['precision'], row['recall'], row['f1-score'], row['support']])

if __name__ == "__main__":
    main()
