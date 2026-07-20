import os
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
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from hust_data import HUSTTemporalDataset, compute_seen_train_stats

CONFIG = {
    'batch_size': int(os.environ.get('PGCD_BATCH_SIZE', '64')),
    'epochs': int(os.environ.get('PGCD_SPECIALIST_EPOCHS', '50')),
    'lr': float(os.environ.get('PGCD_SPECIALIST_LR', '1e-4')),
    'signal_len': 1024,
    'stride': 1024,
    'synthetic_val_fraction': 0.20,
    'seed': int(os.environ.get('PGCD_SEED', '0')),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'root_dir': './dataset/hust',
    'save_dir': f"./results_hust_ensemble_strict/seed_{os.environ.get('PGCD_SEED', '0')}",
    
    'syn_ib_path': f"./results_hust_generation_strict/seed_{os.environ.get('PGCD_SEED', '0')}/Exp_IB/synthetic_data.pt",
    'syn_ob_path': f"./results_hust_generation_strict/seed_{os.environ.get('PGCD_SEED', '0')}/Exp_OB/synthetic_data.pt"
}
os.makedirs(CONFIG['save_dir'], exist_ok=True)

CLASSES_FINAL = ['N', 'B', 'I', 'O', 'IB', 'OB']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def train_specialist(model_name, syn_path, seen_train, device):
    print(f"\n>>> Training Specialist Model: {model_name}...")
    model_path = os.path.join(CONFIG['save_dir'], f'{model_name}.pth')
    if os.path.exists(model_path):
        model = RobustClassifier(num_classes=5).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f">>> Loaded existing specialist: {model_path}")
        return model
    
    if not os.path.exists(syn_path):
        raise FileNotFoundError(f"{syn_path} not found.")
    syn_data = torch.load(syn_path)
    syn_ds = TensorDataset(syn_data.tensors[0], torch.full((len(syn_data),), 4, dtype=torch.long))
    val_size = max(1, int(CONFIG['synthetic_val_fraction'] * len(syn_ds)))
    train_size = len(syn_ds) - val_size
    syn_train, _ = random_split(
        syn_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['seed']),
    )
    
    train_ds = ConcatDataset([seen_train, syn_train])
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
            
    torch.save(model.state_dict(), model_path)
    return model

def main():
    set_seed(CONFIG['seed'])
    print("=== Ensemble GZSL Strategy ===")
    device = CONFIG['device']
    
    root = CONFIG['root_dir']
    seen_codes = ['N', 'B', 'I', 'O']
    stats = compute_seen_train_stats(root, seen_codes)
    seen_train = HUSTTemporalDataset(
        root_dir=root,
        class_codes=seen_codes,
        labels=[0, 1, 2, 3],
        split='train',
        normalization_stats=stats,
        signal_len=CONFIG['signal_len'],
        stride=CONFIG['stride'],
    )
    

    model_ib = train_specialist("Specialist_IB", CONFIG['syn_ib_path'], seen_train, device)
    
    model_ob = train_specialist("Specialist_OB", CONFIG['syn_ob_path'], seen_train, device)


if __name__ == "__main__":
    main()
