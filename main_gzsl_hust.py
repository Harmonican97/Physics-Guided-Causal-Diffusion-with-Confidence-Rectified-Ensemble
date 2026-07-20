import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from hust_data import HUSTTemporalDataset, compute_seen_train_stats

try:
    from causal_diffusion_model import CompositionalUNet1D, PhysicsGuidedDiffusion
except ImportError:
    print("[Error] 'causal_diffusion_model.py' not found. Please ensure it is in the same directory.")
    exit(1)

FS = 51200.0 
INPUT_SPEED_RPM = 1800.0
SHAFT_FREQ = INPUT_SPEED_RPM / 60.0

COEFFS = {
    'Ball': 1.99,
    'Inner': 4.94,
    'Outer': 3.05,
}

def get_fault_frequencies(task_type):
    f_r = SHAFT_FREQ
    base_freqs = {
        0: [COEFFS['Ball'] * f_r],          
        1: [COEFFS['Inner'] * f_r],         
        2: [COEFFS['Outer'] * f_r]          
    }
    if task_type == 'IB':
        base_freqs[3] = [COEFFS['Inner'] * f_r, COEFFS['Ball'] * f_r]
    elif task_type == 'OB':
        base_freqs[3] = [COEFFS['Outer'] * f_r, COEFFS['Ball'] * f_r]
    return base_freqs

def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


CONFIG = {
    'epoch_diffusion': _env_int('PGCD_DIFFUSION_EPOCHS', 200),
    'epoch_classifier': _env_int('PGCD_CLASSIFIER_EPOCHS', 60),
    'batch_size': _env_int('PGCD_BATCH_SIZE', 64),
    'lr_diffusion': 1e-4,
    'lr_classifier': 5e-4,
    'signal_len': 1024,
    'stride': 1024,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'root_dir': './dataset/hust',
    'seed': int(os.environ.get('PGCD_SEED', '0')),
    'save_dir': f"./results_hust_generation_strict/seed_{os.environ.get('PGCD_SEED', '0')}",
    
    # Generation Params
    'guidance_scale': _env_float('PGCD_GUIDANCE_SCALE', 10.0),
    'n_synthetic_samples': _env_int('PGCD_SYNTHETIC_SAMPLES', 800),
    'refine_timestep': _env_int('PGCD_REFINE_STEPS', 150),
    'checkpoint_interval': _env_int('PGCD_CHECKPOINT_INTERVAL', 10),
    'train_auxiliary_classifier': bool(_env_int('PGCD_TRAIN_AUX_CLASSIFIER', 0)),
}

TRANSFER_PATH_WEIGHTS = {
    'IB': [1.5, 2.0],  # [Inner, Ball]
    'OB': [1.0, 2.0],  # [Outer, Ball]
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

def train_diffusion_model(diffusion, train_loader, optimizer, epochs, device, checkpoint_path):
    diffusion.train()
    print(f"\n>>> Diffusion model training...")
    state_path = checkpoint_path.replace('.pth', '_training_state.pt')
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
    torch.save(diffusion.state_dict(), checkpoint_path)

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

def evaluate_gzsl(model, seen_test_loader, unseen_test_loader, device):
    model.eval()
    
    def get_preds(loader):
        preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds.extend(torch.argmax(model(x), dim=1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        return preds, targets

    seen_p, seen_t = get_preds(seen_test_loader)
    unseen_p, unseen_t = get_preds(unseen_test_loader)
    
    acc_seen = accuracy_score(seen_t, seen_p)
    acc_unseen = accuracy_score(unseen_t, unseen_p)
    h_score = (2 * acc_seen * acc_unseen) / (acc_seen + acc_unseen + 1e-8)
    cm = confusion_matrix(seen_t + unseen_t, seen_p + unseen_p)
    return acc_seen, acc_unseen, h_score, cm

class RobustClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 64, 2, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 32, 2, 16), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 16, 2, 8), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 8, 1, 4), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)
    def extract_features(self, x):
        return self.features(x).squeeze(-1)

def generate_synthetic_data(diffusion, seen_loader, task_type, device):
    diffusion.eval()
    n_samples = CONFIG['n_synthetic_samples']
    print(f"\n>>> Generating Synthetic '{task_type}' Data...")
    
    if task_type == 'IB': comps = [0, 1] 
    else: comps = [0, 2]
    
    data_pool = {0: [], 1: [], 2: []}
    for x, y in seen_loader:
        for i in range(len(y)):
            lbl = y[i].item()
            if lbl in data_pool: data_pool[lbl].append(x[i])
    
    mixed_data = []
    for _ in range(n_samples):
        s_a = data_pool[comps[0]][np.random.randint(len(data_pool[comps[0]]))]
        s_b = data_pool[comps[1]][np.random.randint(len(data_pool[comps[1]]))]
        w = np.random.dirichlet((0.5, 0.5))
        mixed = w[0]*s_a + w[1]*s_b + torch.randn_like(s_a)*0.02
        mixed_data.append(mixed / (torch.sqrt(torch.mean(mixed**2)) + 1e-8))
    
    x0_coarse = torch.stack(mixed_data)
    refined_data = []
    batch_size = CONFIG['batch_size']
    target_freqs = get_fault_frequencies(task_type)[3]
    target_weights = TRANSFER_PATH_WEIGHTS[task_type]
    
    for i in tqdm(range(0, n_samples, batch_size)):
        x_batch = x0_coarse[i:i+batch_size].to(device)
        curr_bs = x_batch.size(0)
        
        target_hot = torch.zeros(curr_bs, 3, device=device)
        target_hot[:, comps[0]] = 1.0
        target_hot[:, comps[1]] = 1.0
        
        freqs_list = [target_freqs for _ in range(curr_bs)]
        weights_list = [target_weights for _ in range(curr_bs)]
        
        t_refine = torch.full((curr_bs,), CONFIG['refine_timestep'], device=device, dtype=torch.long)
        noise = torch.randn_like(x_batch)
        alpha_bar = diffusion.alphas_cumprod[t_refine][:, None, None]

        x_t = torch.sqrt(alpha_bar) * x_batch + torch.sqrt(1 - alpha_bar) * noise
        x_t = x_t.detach().requires_grad_(True)
        
        x_out = diffusion.physics_guided_sample_from_t(
            x_t, CONFIG['refine_timestep'],
            target_multihot=target_hot,
            target_freqs_list=freqs_list,
            target_weights_list=weights_list,
            guidance_scale=CONFIG['guidance_scale']
        )
        
        refined_data.append(x_out.detach().cpu())
        
    return TensorDataset(torch.cat(refined_data), torch.full((n_samples,), 3).long())


def run_gzsl_experiment(task_type):
    started_at = time.perf_counter()
    set_seed(CONFIG['seed'])
    seed_root = os.path.join('./results_hust_generation_strict', f"seed_{CONFIG['seed']}")
    CONFIG['save_dir'] = os.path.join(
        seed_root, f'Exp_{task_type}'
    )
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = CONFIG['device']
    
    print(f"\n{'='*40}\nStarting GZSL Experiment: Unseen = {task_type}\n{'='*40}")
    
    root = CONFIG['root_dir']
    seen_codes = ['B', 'I', 'O']
    # Use one normalization pair from every seen state, including normal, so
    # generated candidates and the downstream specialists share the same scale.
    stats = compute_seen_train_stats(root, ['N', 'B', 'I', 'O'])
    dataset_args = dict(
        root_dir=root,
        class_codes=seen_codes,
        labels=[0, 1, 2],
        normalization_stats=stats,
        signal_len=CONFIG['signal_len'],
        stride=CONFIG['stride'],
    )
    seen_train = HUSTTemporalDataset(split='train', **dataset_args)
    seen_val = HUSTTemporalDataset(split='val', **dataset_args)
    seen_test = HUSTTemporalDataset(split='test', **dataset_args)
    unseen_ds = HUSTTemporalDataset(
        root_dir=root,
        class_codes=[task_type],
        labels=[3],
        split='test',
        normalization_stats=stats,
        signal_len=CONFIG['signal_len'],
        stride=CONFIG['stride'],
        unseen=True,
    )
    print(
        'Strict temporal counts (train/val/seen-test/unseen-test):',
        len(seen_train), len(seen_val), len(seen_test), len(unseen_ds),
    )
    
    seen_train_loader = DataLoader(seen_train, batch_size=CONFIG['batch_size'], shuffle=True)
    seen_test_loader = DataLoader(seen_test, batch_size=CONFIG['batch_size'], shuffle=False)
    unseen_test_loader = DataLoader(unseen_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    unet = CompositionalUNet1D(num_fault_components=3)
    diffusion = PhysicsGuidedDiffusion(
        unet,
        device=device,
        fs=FS,
        physics_bandwidth_hz=FS / CONFIG['signal_len'],
        physics_harmonics=3,
    )
    
    diff_path = os.path.join(seed_root, 'compositional_diffusion.pth')
    legacy_diff_path = os.path.join(seed_root, 'causal_diffusion.pth')
    training_state_path = diff_path.replace('.pth', '_training_state.pt')
    completed_epochs = 0
    if os.path.exists(training_state_path):
        completed_epochs = int(torch.load(training_state_path, map_location='cpu')['epoch'])
    if (os.path.exists(diff_path) or os.path.exists(legacy_diff_path)) and completed_epochs >= CONFIG['epoch_diffusion']:
        print(">>> Loading pre-trained Diffusion...")
        diffusion.load_state_dict(torch.load(
            diff_path if os.path.exists(diff_path) else legacy_diff_path, map_location=device
        ))
    else:
        optimizer = optim.Adam(diffusion.parameters(), lr=CONFIG['lr_diffusion'])
        train_diffusion_model(
            diffusion, seen_train_loader, optimizer, CONFIG['epoch_diffusion'], device, diff_path
        )
        
    syn_path = os.path.join(CONFIG['save_dir'], 'synthetic_data.pt')
    if os.path.exists(syn_path):
        print(">>> Loading Synthetic Data...")
        synthetic_ds = torch.load(syn_path)
    else:
        synthetic_ds = generate_synthetic_data(diffusion, seen_train_loader, task_type, device)
        torch.save(synthetic_ds, syn_path)

    with open(os.path.join(CONFIG['save_dir'], 'generation_summary.txt'), 'w', encoding='utf-8') as handle:
        handle.write(f"task={task_type}\n")
        handle.write(f"seed={CONFIG['seed']}\n")
        handle.write(f"synthetic_samples={len(synthetic_ds)}\n")
        handle.write(f"runtime_seconds={time.perf_counter() - started_at:.6f}\n")
        handle.write(f"normalization_mean={stats[0]:.12g}\n")
        handle.write(f"normalization_std={stats[1]:.12g}\n")

    if not CONFIG['train_auxiliary_classifier']:
        print(">>> Candidate generation complete; auxiliary direct classifier disabled.")
        return

    classifier = RobustClassifier(num_classes=4).to(device)
    full_ds = ConcatDataset([seen_train, synthetic_ds])
    train_loader = DataLoader(full_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    optimizer = optim.Adam(classifier.parameters(), lr=CONFIG['lr_classifier'], weight_decay=1e-4)
    
    train_classifier_robust(classifier, train_loader, optimizer, CONFIG['epoch_classifier'], device)
    torch.save(classifier.state_dict(), os.path.join(CONFIG['save_dir'], 'classifier.pth'))
    
    acc_seen, acc_unseen, h_score, cm = evaluate_gzsl(classifier, seen_test_loader, unseen_test_loader, device)
    
    print(f"\nResults for {task_type}:")
    print(f"  Seen Accuracy:   {acc_seen*100:.2f}%")
    print(f"  Unseen Accuracy: {acc_unseen*100:.2f}%")
    print(f"  H-Score:         {h_score*100:.2f}%")
    
    plt.figure(figsize=(6, 5))
    lbls_txt = ['Ball', 'Inner', 'Outer', task_type]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=lbls_txt, yticklabels=lbls_txt)
    plt.title(f'GZSL Confusion Matrix ({task_type})')
    plt.savefig(os.path.join(CONFIG['save_dir'], 'cm.png'))
    plt.close()
    
    print(">>> Generating t-SNE...")
    classifier.eval()
    all_feats, all_labels = [], []
    combined_test = DataLoader(ConcatDataset([seen_test, unseen_ds]), batch_size=64)
    with torch.no_grad():
        for x, y in combined_test:
            all_feats.append(classifier.extract_features(x.to(device)).cpu().numpy())
            all_labels.append(y.numpy())
    
    all_feats = np.concatenate(all_feats)
    all_labels = np.concatenate(all_labels)
    if len(all_feats) > 2000:
        idx = np.random.choice(len(all_feats), 2000, replace=False)
        all_feats, all_labels = all_feats[idx], all_labels[idx]
        
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z = tsne.fit_transform(all_feats)
    plt.figure(figsize=(8, 6))
    txt = [lbls_txt[l] for l in all_labels]
    sns.scatterplot(x=z[:,0], y=z[:,1], hue=txt, palette='bright', style=txt, s=60)
    plt.title(f't-SNE Feature Space ({task_type})')
    plt.savefig(os.path.join(CONFIG['save_dir'], 'tsne.png'))
    plt.close()
    print("Experiment Complete.")

if __name__ == "__main__":
    run_gzsl_experiment('IB')
    run_gzsl_experiment('OB')
