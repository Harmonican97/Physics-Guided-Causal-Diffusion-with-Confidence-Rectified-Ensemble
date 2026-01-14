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
from sklearn.cluster import KMeans
import scipy.io as sio

CONFIG = {
    'batch_size': 64,
    'signal_len': 1024,
    'overlap_rate': 0.50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'root_dir': './dataset/hust',
    'save_dir': './results_gzsl_confidence',
    
    'model_ib_path': './results_gzsl_ensemble/Specialist_IB.pth',
    'model_ob_path': './results_gzsl_ensemble/Specialist_OB.pth'
}
os.makedirs(CONFIG['save_dir'], exist_ok=True)

CLASSES = ['N', 'B', 'I', 'O', 'IB', 'OB']
LABEL_MAP = {k:v for v, k in enumerate(CLASSES)}

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

class ConfidenceRectifier:
    def __init__(self, model_a, model_b, device):
        self.model_a = model_a.eval().to(device)
        self.model_b = model_b.eval().to(device)
        self.device = device

    def extract_fused_features_and_probs(self, loader):
        feats, labels = [], []
        preds_a_cls = []
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                
                _, fa = self.model_a(x)
                _, fb = self.model_b(x)
                fa = F.normalize(fa, p=2, dim=1)
                fb = F.normalize(fb, p=2, dim=1)
                fused = torch.cat([fa, fb], dim=1)
                
                out_a, _ = self.model_a(x)
                pred = out_a.argmax(dim=1)
                
                feats.append(fused.cpu())
                labels.append(y)
                preds_a_cls.append(pred.cpu())
                
        return torch.cat(feats).numpy(), torch.cat(labels).numpy(), torch.cat(preds_a_cls).numpy()

    def get_cluster_confidence(self, cluster_feats, cluster_indices, raw_dataset):
        subset = torch.utils.data.Subset(raw_dataset, cluster_indices)
        loader = DataLoader(subset, batch_size=64, shuffle=False)
        
        conf_a_list = []
        conf_b_list = []
        
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                
                # Model A (IB Specialist): Unseen is Class 4
                out_a, _ = self.model_a(x)
                prob_a = F.softmax(out_a, dim=1)[:, 4]
                
                # Model B (OB Specialist): Unseen is Class 4
                out_b, _ = self.model_b(x)
                prob_b = F.softmax(out_b, dim=1)[:, 4]
                
                conf_a_list.append(prob_a.cpu().numpy())
                conf_b_list.append(prob_b.cpu().numpy())
                
        return np.mean(np.concatenate(conf_a_list)), np.mean(np.concatenate(conf_b_list))

    def predict_and_rectify(self, test_loader, full_dataset):
        print("Running Confidence-Based Rectification...")
        
        feats, true_labels, preds_a = self.extract_fused_features_and_probs(test_loader)
        final_preds = np.copy(preds_a)
        
        unseen_mask = (preds_a == 4)
        unseen_indices = np.where(unseen_mask)[0]
        
        print(f"Detected {len(unseen_indices)} Unseen Candidates.")
        
        if len(unseen_indices) > 0:
            candidate_feats = feats[unseen_indices]
            
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(candidate_feats)
            
            idx_c0 = unseen_indices[cluster_labels == 0]
            idx_c1 = unseen_indices[cluster_labels == 1]
            
            print(f"  Cluster 0 size: {len(idx_c0)}")
            print(f"  Cluster 1 size: {len(idx_c1)}")
            
            score_a_c0, score_b_c0 = self.get_cluster_confidence(None, idx_c0, full_dataset)
            score_a_c1, score_b_c1 = self.get_cluster_confidence(None, idx_c1, full_dataset)
            
            print(f"  Cluster 0 Scores -> Model A (IB): {score_a_c0:.4f}, Model B (OB): {score_b_c0:.4f}")
            print(f"  Cluster 1 Scores -> Model A (IB): {score_a_c1:.4f}, Model B (OB): {score_b_c1:.4f}")
            
            option_1 = score_a_c0 + score_b_c1
            option_2 = score_b_c0 + score_a_c1
            
            if option_1 > option_2:
                print("  Decision: Cluster 0 is IB, Cluster 1 is OB")
                mapping = {0: 4, 1: 5}
            else:
                print("  Decision: Cluster 0 is OB, Cluster 1 is IB")
                mapping = {0: 5, 1: 4}
            
            rectified_labels = np.vectorize(mapping.get)(cluster_labels)
            final_preds[unseen_indices] = rectified_labels
            
        return final_preds, true_labels, feats

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
                print(f"  Loaded {os.path.basename(path)}: {len(self.data)}")
            except: pass
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

def main():
    print("=== Confidence-Based GZSL Rectification ===")
    device = CONFIG['device']
    
    if not os.path.exists(CONFIG['model_ib_path']):
        print("Models not found.")
        return

    model_ib = RobustClassifier(num_classes=5)
    model_ib.load_state_dict(torch.load(CONFIG['model_ib_path']))
    model_ob = RobustClassifier(num_classes=5)
    model_ob.load_state_dict(torch.load(CONFIG['model_ob_path']))
    
    # Data
    root = CONFIG['root_dir']
    seen_ds = HUST_Dataset([os.path.join(root, f'{x}504.mat') for x in ['N', 'B', 'I', 'O']], [0,1,2,3], 1024, 0.5)
    train_sz = int(0.8 * len(seen_ds))
    _, seen_test = random_split(seen_ds, [train_sz, len(seen_ds)-train_sz], generator=torch.Generator().manual_seed(42))
    
    real_ib_ds = HUST_Dataset([os.path.join(root, 'IB504.mat')], [4], 1024, 0.5)
    real_ob_ds = HUST_Dataset([os.path.join(root, 'OB504.mat')], [5], 1024, 0.5)
    
    full_test_dataset = ConcatDataset([seen_test, real_ib_ds, real_ob_ds])
    test_loader = DataLoader(full_test_dataset, batch_size=64, shuffle=False)
    
    # Process
    rectifier = ConfidenceRectifier(model_ib, model_ob, device)
    preds, targets, feats = rectifier.predict_and_rectify(test_loader, full_test_dataset)
    
    # Metrics
    mask_seen = targets < 4
    acc_seen = accuracy_score(targets[mask_seen], preds[mask_seen])
    mask_unseen = targets >= 4
    acc_unseen = accuracy_score(targets[mask_unseen], preds[mask_unseen])
    h_score = (2 * acc_seen * acc_unseen) / (acc_seen + acc_unseen + 1e-8)
    
    print(f"\n{'='*30}")
    print(f"FINAL CONFIDENCE RESULTS")
    print(f"{'='*30}")
    print(f"Seen Acc:   {acc_seen*100:.2f}%")
    print(f"Unseen Acc: {acc_unseen*100:.2f}%")
    print(f"H-Score:    {h_score*100:.2f}%")
    
    # CM
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confidence-Rectified CM\nH-Score: {h_score*100:.2f}%')
    plt.savefig(os.path.join(CONFIG['save_dir'], 'conf_cm.png'))
    
    # t-SNE
    print("Generating t-SNE...")
    if len(feats) > 3000:
        idx = np.random.choice(len(feats), 3000, replace=False)
        feats, targets, preds = feats[idx], targets[idx], preds[idx]
        
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z = tsne.fit_transform(feats)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    txt_true = [CLASSES[t] for t in targets]
    sns.scatterplot(x=z[:,0], y=z[:,1], hue=txt_true, style=txt_true, palette='bright', s=60, ax=axes[0])
    axes[0].set_title("True Labels")
    
    txt_pred = [CLASSES[t] for t in preds]
    sns.scatterplot(x=z[:,0], y=z[:,1], hue=txt_pred, style=txt_pred, palette='bright', s=60, ax=axes[1])
    axes[1].set_title("Predicted Labels (Confidence-Aligned)")
    plt.savefig(os.path.join(CONFIG['save_dir'], 'conf_tsne.png'))
    print("Done.")

if __name__ == "__main__":
    main()