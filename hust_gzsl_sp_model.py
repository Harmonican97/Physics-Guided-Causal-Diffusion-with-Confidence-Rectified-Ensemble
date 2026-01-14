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

CONFIG = {
    'batch_size': 64,
    'epochs': 50,
    'lr': 1e-4,
    'signal_len': 1024,
    'overlap_rate': 0.50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'root_dir': './dataset/hust',
    'save_dir': './results_gzsl_ensemble',
    
    'syn_ib_path': './results_gzsl_final/Exp_IB/synthetic_data.pt',
    'syn_ob_path': './results_gzsl_final/Exp_OB/synthetic_data.pt'
}
os.makedirs(CONFIG['save_dir'], exist_ok=True)

CLASSES_FINAL = ['N', 'B', 'I', 'O', 'IB', 'OB']

class RobustClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 64, 2, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 32, 2, 16), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 16, 2, 8), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 8, 1, 4), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.features(x).squeeze(-1)
        out = self.classifier(feat)
        return out, feat

class HUST_Dataset(Dataset):
    def __init__(self, file_paths, labels, signal_len=1024, overlap_rate=0.5):
        self.data, self.labels = [], []
        stride = int(signal_len * (1 - overlap_rate))
        for path, label in zip(file_paths, labels):
            if not os.path.exists(path): continue
            try:
                mat = sio.loadmat(path)
                raw = None
                for k in ['data', 'Data', 'vibration', 'signal']:
                    for mk in mat.keys():
                        if k in mk: raw = mat[mk].flatten(); break
                    if raw is not None: break
                if raw is None:
                     for k,v in mat.items():
                        if isinstance(v, np.ndarray) and v.size > 10000:
                            raw = v.flatten(); break
                if raw is None: continue
                raw = (raw - np.mean(raw)) / (np.std(raw) + 1e-8)
                for i in range(0, len(raw) - signal_len + 1, stride):
                    self.data.append(raw[i : i + signal_len])
                    self.labels.append(label)
                print(f"  Loaded {os.path.basename(path)} -> Label {label}: {len(self.data)}")
            except: pass
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

def train_specialist(model_name, syn_path, seen_train, device):
    print(f"\n>>> Training Specialist Model: {model_name}...")
    
    if not os.path.exists(syn_path):
        raise FileNotFoundError(f"{syn_path} not found.")
    syn_data = torch.load(syn_path)
    syn_ds = TensorDataset(syn_data.tensors[0], torch.full((len(syn_data),), 4, dtype=torch.long))
    
    train_ds = ConcatDataset([seen_train, syn_ds])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    
    model = RobustClassifier(num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    model.train()
    for epoch in range(CONFIG['epochs']):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            
    torch.save(model.state_dict(), f'./results_gzsl_ensemble/{model_name}.pth')
    return model

def main():
    print("=== Ensemble GZSL Strategy ===")
    device = CONFIG['device']
    
    root = CONFIG['root_dir']
    seen_files = [os.path.join(root, f'{x}504.mat') for x in ['N', 'B', 'I', 'O']]
    seen_ds = HUST_Dataset(seen_files, [0, 1, 2, 3], CONFIG['signal_len'], CONFIG['overlap_rate'])
    
    train_sz = int(0.8 * len(seen_ds))
    seen_train, seen_test = random_split(seen_ds, [train_sz, len(seen_ds)-train_sz], generator=torch.Generator().manual_seed(42))
    
    real_ib_ds = HUST_Dataset([os.path.join(root, 'IB504.mat')], [4], CONFIG['signal_len'], CONFIG['overlap_rate'])
    real_ob_ds = HUST_Dataset([os.path.join(root, 'OB504.mat')], [5], CONFIG['signal_len'], CONFIG['overlap_rate'])
    

    model_ib = train_specialist("Specialist_IB", CONFIG['syn_ib_path'], seen_train, device)
    
    model_ob = train_specialist("Specialist_OB", CONFIG['syn_ob_path'], seen_train, device)


if __name__ == "__main__":
    main()