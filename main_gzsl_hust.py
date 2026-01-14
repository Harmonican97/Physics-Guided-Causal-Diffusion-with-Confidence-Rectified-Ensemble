import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import scipy.io as sio

try:
    from causal_diffusion_model import CausalUNet1D, PhysicsGuidedDiffusion
except ImportError:
    print("[Error] 'causal_diffusion_model.py' not found. Please ensure it is in the same directory.")
    exit(1)

FS = 51200.0 
INPUT_SPEED_RPM = 1800.0
SHAFT_FREQ = INPUT_SPEED_RPM / 60.0

COEFFS = {
    'Ball': 2.35,  
    'Inner': 5.41, 
    'Outer': 3.59  
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

CONFIG = {
    'epoch_diffusion': 200, 
    'epoch_classifier': 60,
    'batch_size': 64,
    'lr_diffusion': 1e-4,
    'lr_classifier': 5e-4,
    'signal_len': 1024,
    'overlap_rate': 0.50, 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'root_dir': './dataset/hust',
    'save_dir': './results_gzsl_final',
    
    # Generation Params
    'guidance_scale': 10.0,      
    'n_synthetic_samples': 800, 
    'refine_timestep': 150      
}


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
    print(f"\n>>> Diffusion model training...")
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

class HUST_Dataset(Dataset):
    def __init__(self, file_paths, labels, signal_len=1024, overlap_rate=0.5):
        self.data = []
        self.labels = []
        stride = int(signal_len * (1 - overlap_rate))
        print(f"[Dataset] Loading files. Stride: {stride} (Overlap {overlap_rate*100}%)")
        
        for path, label in zip(file_paths, labels):
            if not os.path.exists(path):
                continue
            try:
                mat = sio.loadmat(path)
                raw = None
                for k in ['data', 'Data', 'vibration', 'signal']:
                    for mk in mat.keys():
                        if k in mk: raw = mat[mk].flatten(); break
                    if raw is not None: break
                if raw is None: # 兜底
                     for k,v in mat.items():
                        if isinstance(v, np.ndarray) and v.size > 10000:
                            raw = v.flatten(); break
                if raw is None: continue
                
                raw = (raw - np.mean(raw)) / (np.std(raw) + 1e-8)
                for i in range(0, len(raw) - signal_len + 1, stride):
                    self.data.append(raw[i : i + signal_len])
                    self.labels.append(label)
                print(f"  Loaded {os.path.basename(path)} -> Label {label}: {len(self.data)} total samples")
            except Exception: pass

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

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
    batch_size = 32
    target_freqs = get_fault_frequencies(task_type)[3]
    
    for i in tqdm(range(0, n_samples, batch_size)):
        x_batch = x0_coarse[i:i+batch_size].to(device)
        curr_bs = x_batch.size(0)
        
        target_hot = torch.zeros(curr_bs, 3, device=device)
        target_hot[:, comps[0]] = 1.0
        target_hot[:, comps[1]] = 1.0
        
        freqs_list = [target_freqs for _ in range(curr_bs)]
        
        t_refine = torch.full((curr_bs,), CONFIG['refine_timestep'], device=device, dtype=torch.long)
        noise = torch.randn_like(x_batch)
        alpha_bar = diffusion.alphas_cumprod[t_refine][:, None, None]

        x_t = torch.sqrt(alpha_bar) * x_batch + torch.sqrt(1 - alpha_bar) * noise
        x_t = x_t.detach().requires_grad_(True)
        
        x_out = diffusion.physics_guided_sample_from_t(
            x_t, CONFIG['refine_timestep'],
            target_multihot=target_hot,
            target_freqs_list=freqs_list,
            guidance_scale=CONFIG['guidance_scale']
        )
        
        refined_data.append(x_out.detach().cpu())
        
    return TensorDataset(torch.cat(refined_data), torch.full((n_samples,), 3).long())


def run_gzsl_experiment(task_type):
    CONFIG['save_dir'] = os.path.join('./results_gzsl_final', f'Exp_{task_type}')
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = CONFIG['device']
    
    print(f"\n{'='*40}\nStarting GZSL Experiment: Unseen = {task_type}\n{'='*40}")
    
    root = CONFIG['root_dir']
    seen_files = [os.path.join(root, f'{x}504.mat') for x in ['B', 'I', 'O']]
    unseen_file = os.path.join(root, f'{task_type}504.mat')
    
    seen_ds = HUST_Dataset(seen_files, [0, 1, 2], CONFIG['signal_len'], CONFIG['overlap_rate'])
    unseen_ds = HUST_Dataset([unseen_file], [3], CONFIG['signal_len'], CONFIG['overlap_rate'])
    
    train_sz = int(0.8 * len(seen_ds))
    seen_train, seen_test = random_split(seen_ds, [train_sz, len(seen_ds)-train_sz], generator=torch.Generator().manual_seed(42))
    
    seen_train_loader = DataLoader(seen_train, batch_size=CONFIG['batch_size'], shuffle=True)
    seen_test_loader = DataLoader(seen_test, batch_size=CONFIG['batch_size'], shuffle=False)
    unseen_test_loader = DataLoader(unseen_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    unet = CausalUNet1D(num_fault_components=3)
    diffusion = PhysicsGuidedDiffusion(unet, device=device)
    
    diff_path = os.path.join(CONFIG['save_dir'], 'causal_diffusion.pth')
    if os.path.exists(diff_path):
        print(">>> Loading pre-trained Diffusion...")
        diffusion.load_state_dict(torch.load(diff_path))
    else:
        optimizer = optim.Adam(diffusion.parameters(), lr=CONFIG['lr_diffusion'])
        train_diffusion_model(diffusion, seen_train_loader, optimizer, CONFIG['epoch_diffusion'], device)
        
    syn_path = os.path.join(CONFIG['save_dir'], 'synthetic_data.pt')
    if os.path.exists(syn_path):
        print(">>> Loading Synthetic Data...")
        synthetic_ds = torch.load(syn_path)
    else:
        synthetic_ds = generate_synthetic_data(diffusion, seen_train_loader, task_type, device)
        torch.save(synthetic_ds, syn_path)

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