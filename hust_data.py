import os

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


SPLIT_RATIOS = (0.60, 0.20, 0.20)


def load_hust_signal(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mat = sio.loadmat(path)
    for preferred in ('data', 'Data', 'vibration', 'signal'):
        for key, value in mat.items():
            if preferred.lower() in key.lower() and isinstance(value, np.ndarray):
                if value.size > 1024:
                    return value.reshape(-1).astype(np.float32)
    candidates = [
        value.reshape(-1).astype(np.float32)
        for value in mat.values()
        if isinstance(value, np.ndarray) and value.size > 10000
    ]
    if not candidates:
        raise ValueError(f'No vibration array found in {path}')
    return max(candidates, key=len)


def temporal_block(raw, split):
    n = len(raw)
    train_end = int(SPLIT_RATIOS[0] * n)
    val_end = int((SPLIT_RATIOS[0] + SPLIT_RATIOS[1]) * n)
    if split == 'train':
        return raw[:train_end]
    if split == 'val':
        return raw[train_end:val_end]
    if split == 'test':
        return raw[val_end:]
    raise ValueError(f'Unsupported split: {split}')


def compute_seen_train_stats(root_dir, seen_codes):
    training_blocks = [
        temporal_block(load_hust_signal(os.path.join(root_dir, f'{code}504.mat')), 'train')
        for code in seen_codes
    ]
    merged = np.concatenate(training_blocks)
    return float(merged.mean()), float(merged.std() + 1e-8)


class HUSTTemporalDataset(Dataset):
    """HUST signals split chronologically before non-overlapping windowing."""

    def __init__(
        self,
        root_dir,
        class_codes,
        labels,
        split,
        normalization_stats,
        signal_len=1024,
        stride=None,
        unseen=False,
    ):
        if unseen and split != 'test':
            raise ValueError('Real unseen data may only be instantiated for the test split.')
        self.signal_len = int(signal_len)
        stride = int(stride if stride is not None else signal_len)
        mean, std = normalization_stats
        data, targets = [], []

        for code, label in zip(class_codes, labels):
            path = os.path.join(root_dir, f'{code}504.mat')
            raw = load_hust_signal(path)
            block = (temporal_block(raw, split) - mean) / std
            for start in range(0, len(block) - self.signal_len + 1, stride):
                data.append(block[start:start + self.signal_len])
                targets.append(label)

        if data:
            self.data = torch.from_numpy(np.stack(data).astype(np.float32)).unsqueeze(1)
            self.labels = torch.tensor(targets, dtype=torch.long)
        else:
            self.data = torch.empty((0, 1, self.signal_len), dtype=torch.float32)
            self.labels = torch.empty(0, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
