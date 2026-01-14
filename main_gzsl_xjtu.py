import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score

from causal_diffusion_model import CausalUNet1D, PhysicsGuidedDiffusion
from data_loader import get_dataloaders

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

CONFIG = {
    'epoch_diffusion': 200, 
    'epoch_classifier': 80,
    'batch_size': 32,
    'lr_diffusion': 1e-4,
    'lr_classifier': 5e-4,
    'signal_len': 1024,
    'num_fault_components': 3,
    'num_classes_output': 4,
    'unseen_class': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results_adaptive_new',
    
    'guidance_scale': 10.0,      
    'n_synthetic_samples': 1200,
    'refine_timestep': 120       
}
os.makedirs(CONFIG['save_dir'], exist_ok=True)

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
    print("\n>>> Noise Injection + Adaptive Info...")
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
    print(f"\n>>> Mix-and-Refine (Adaptive Guidance)...")
    
    coarse_data, max_weights = create_stratified_mix_dataset(seen_loader, n_samples)
    
    refined_data = []
    refined_labels = []
    
    batch_size = 32
    target_freqs = FAULT_FREQUENCIES[target_label]
    t_refine = CONFIG['refine_timestep']
    
    for i in tqdm(range(0, n_samples, batch_size)):
        x0_coarse = coarse_data[i : i+batch_size].to(device)
        batch_max_w = max_weights[i : i+batch_size].to(device)
        current_bs = x0_coarse.size(0)
        
        scales = torch.ones(current_bs, 1, 1, device=device) * 8.0 # Default low scale
        hard_mask = batch_max_w > 0.85
        scales[hard_mask] = 40.0 # High scale for hard samples
        
        target_hot = torch.ones(current_bs, 3, device=device) 
        freqs_batch = [target_freqs for _ in range(current_bs)]
        
        t = torch.full((current_bs,), t_refine, device=device, dtype=torch.long)
        noise = torch.randn_like(x0_coarse)
        alpha_bar = diffusion.alphas_cumprod[t][:, None, None]
        x_t = torch.sqrt(alpha_bar) * x0_coarse + torch.sqrt(1 - alpha_bar) * noise
        
        with torch.enable_grad():
            x_refined = diffusion.physics_guided_sample_from_t(
                x_t, t_refine,
                target_multihot=target_hot,
                target_freqs_list=freqs_batch,
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
    for epoch in range(epochs):
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
    
    torch.save(diffusion.state_dict(), os.path.join(CONFIG['save_dir'], 'causal_diffusion.pth'))

def main():
    device = CONFIG['device']
    print(f"Device: {device}")
    
    # 1. Load Data
    all_seen_loader, unseen_test_loader = get_dataloaders(
        root_dir='./dataset/xjtu', batch_size=CONFIG['batch_size'], signal_len=CONFIG['signal_len'], num_workers=0
    )
    seen_ds = all_seen_loader.dataset
    train_sz = int(0.8 * len(seen_ds))
    seen_train, seen_test = random_split(seen_ds, [train_sz, len(seen_ds)-train_sz], generator=torch.Generator().manual_seed(42))
    
    seen_train_loader = DataLoader(seen_train, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    seen_test_loader = DataLoader(seen_test, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # 2. Train Diffusion
    unet = CausalUNet1D(num_fault_components=CONFIG['num_fault_components'])
    diffusion = PhysicsGuidedDiffusion(unet, device=device)
    
    if os.path.exists(os.path.join(CONFIG['save_dir'], 'causal_diffusion.pth')):
        print("Load diffusion model...")
        diffusion.load_state_dict(torch.load(os.path.join(CONFIG['save_dir'], 'causal_diffusion.pth')))
    else:
        train_diffusion_model(diffusion, seen_train_loader, optim.Adam(diffusion.parameters(), lr=CONFIG['lr_diffusion']), 
                            CONFIG['epoch_diffusion'], device)    
    
    classifier = RobustClassifier(CONFIG['num_classes_output']).to(device)
    if os.path.exists(os.path.join(CONFIG['save_dir'], 'classifier.pth')):
        print("Load classifier...")
        classifier.load_state_dict(torch.load(os.path.join(CONFIG['save_dir'], 'classifier.pth')))
    else:
        synthetic_ds = generate_refined_mix_data(
            diffusion, seen_train_loader, CONFIG['n_synthetic_samples'], CONFIG['unseen_class'], device
        )

        full_dataset = ConcatDataset([seen_train, synthetic_ds])
        combined_loader = DataLoader(full_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
        train_classifier_robust(classifier, combined_loader, 
                              optim.Adam(classifier.parameters(), lr=CONFIG['lr_classifier'], weight_decay=1e-4), 
                              CONFIG['epoch_classifier'], device)
        torch.save(classifier.state_dict(), os.path.join(CONFIG['save_dir'], 'classifier.pth'))
    
    acc_s, acc_u, h, cm = evaluate_gzsl(classifier, seen_test_loader, unseen_test_loader, device)
    print(f"\nFinal Results:\nSeen Acc: {acc_s:.4f}\nUnseen Acc: {acc_u:.4f}\nH-score: {h:.4f}")
    
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ball','In','Out','Mix'], yticklabels=['Ball','In','Out','Mix'])
    plt.savefig(os.path.join(CONFIG['save_dir'], 'confusion_matrix.png'))

if __name__ == "__main__":
    main()