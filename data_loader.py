import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SPLIT_RATIOS = (0.60, 0.20, 0.20)


def _temporal_block(raw_signal, split):
    """Split the continuous trace before windowing to prevent leakage."""
    n = len(raw_signal)
    train_end = int(SPLIT_RATIOS[0] * n)
    val_end = int((SPLIT_RATIOS[0] + SPLIT_RATIOS[1]) * n)
    if split == 'train':
        return raw_signal[:train_end]
    if split == 'val':
        return raw_signal[train_end:val_end]
    if split == 'test':
        return raw_signal[val_end:]
    raise ValueError(f"Unsupported split: {split}")


def _segment(block, signal_len, stride):
    n_windows = (len(block) - signal_len) // stride + 1
    if n_windows <= 0:
        return np.empty((0, signal_len), dtype=np.float32)
    starts = np.arange(n_windows)[:, None] * stride
    offsets = np.arange(signal_len)[None, :]
    return block[starts + offsets].astype(np.float32, copy=False)


class XJTUGearboxDataset(Dataset):
    """Strict temporal-block XJTU dataset.

    Normalization statistics are computed once from the first 60% of the
    *seen* single-fault traces and reused for validation, seen test, and real
    unseen test data. Real compound-fault samples are exposed only through
    ``mode='unseen', split='test'``.
    """

    folder_map = {
        'ball': {'folder': '1ndBearing_ball', 'label': 0, 'type': 'seen'},
        'inner': {'folder': '1ndBearing_inner', 'label': 1, 'type': 'seen'},
        'outer': {'folder': '1ndBearing_outer', 'label': 2, 'type': 'seen'},
        'mix': {
            'folder': '1ndBearing_mix(inner+outer+ball)',
            'label': 3,
            'type': 'unseen',
        },
    }

    def __init__(
        self,
        root_dir='./dataset/xjtu',
        signal_len=1024,
        stride=None,
        mode='seen',
        split='train',
        normalize=True,
        force_reload=False,
    ):
        self.root_dir = root_dir
        self.signal_len = int(signal_len)
        self.stride = int(stride if stride is not None else signal_len)
        self.mode = mode
        self.split = split
        self.normalize = normalize

        if mode not in {'seen', 'unseen'}:
            raise ValueError("mode must be 'seen' or 'unseen'")
        if split not in {'train', 'val', 'test'}:
            raise ValueError("split must be train/val/test")
        if mode == 'unseen' and split != 'test':
            self.data = torch.empty((0, 1, self.signal_len), dtype=torch.float32)
            self.labels = torch.empty(0, dtype=torch.long)
            self.normalization_stats = None
            return

        cache_dir = os.path.join(root_dir, 'processed_cache')
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(
            cache_dir,
            f'xjtu_strict_temporal_v2_len{self.signal_len}_stride{self.stride}_norm{normalize}.pt',
        )
        if force_reload or not os.path.exists(self.cache_file):
            payload = self._process_all_splits()
            torch.save(payload, self.cache_file)
        else:
            payload = torch.load(self.cache_file, map_location='cpu')

        self.normalization_stats = payload['normalization_stats']
        selected_data, selected_labels = [], []
        for key, info in self.folder_map.items():
            if info['type'] != mode:
                continue
            data, labels = payload['splits'][split][key]
            if len(data):
                selected_data.append(data)
                selected_labels.append(labels)

        if selected_data:
            self.data = torch.cat(selected_data, dim=0)
            self.labels = torch.cat(selected_labels, dim=0)
        else:
            self.data = torch.empty((0, 1, self.signal_len), dtype=torch.float32)
            self.labels = torch.empty(0, dtype=torch.long)

    def _read_all_signals(self):
        signals = {}
        for key, info in self.folder_map.items():
            path = os.path.join(self.root_dir, info['folder'], 'Data_Chan1.txt')
            if not os.path.exists(path):
                raise FileNotFoundError(f'Missing XJTU signal file: {path}')
            # The distributed XJTU files contain a 15-line DASYLab acquisition
            # header followed by one floating-point vibration value per line.
            # Treating the whole file as a rectangular table fails because the
            # header rows have different field counts.
            signal = np.loadtxt(path, skiprows=15, dtype=np.float32)
            signal = np.asarray(signal, dtype=np.float32).reshape(-1)
            if signal.size <= self.signal_len or not np.isfinite(signal).all():
                raise ValueError(f'Invalid XJTU vibration stream: {path}')
            signals[key] = signal
        return signals

    def _process_all_splits(self):
        signals = self._read_all_signals()
        seen_train = np.concatenate(
            [
                _temporal_block(signals[key], 'train')
                for key, info in self.folder_map.items()
                if info['type'] == 'seen'
            ]
        )
        mean = float(np.mean(seen_train))
        std = float(np.std(seen_train) + 1e-8)

        split_payload = {split: {} for split in ('train', 'val', 'test')}
        for split in split_payload:
            for key, info in self.folder_map.items():
                if info['type'] == 'unseen' and split != 'test':
                    windows = np.empty((0, self.signal_len), dtype=np.float32)
                else:
                    block = _temporal_block(signals[key], split)
                    if self.normalize:
                        block = (block - mean) / std
                    windows = _segment(block, self.signal_len, self.stride)
                data = torch.from_numpy(windows).unsqueeze(1)
                labels = torch.full((len(data),), info['label'], dtype=torch.long)
                split_payload[split][key] = (data, labels)

        return {
            'protocol': 'strict-temporal-60-20-20-before-windowing-v2',
            'normalization_stats': {'mean': mean, 'std': std, 'source': 'seen-train-only'},
            'splits': split_payload,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_temporal_dataloaders(root_dir, batch_size=64, signal_len=1024, num_workers=0):
    common = dict(root_dir=root_dir, signal_len=signal_len, stride=signal_len)
    train_ds = XJTUGearboxDataset(mode='seen', split='train', **common)
    val_ds = XJTUGearboxDataset(mode='seen', split='val', **common)
    seen_test_ds = XJTUGearboxDataset(mode='seen', split='test', **common)
    unseen_test_ds = XJTUGearboxDataset(mode='unseen', split='test', **common)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(seen_test_ds, shuffle=False, **loader_kwargs),
        DataLoader(unseen_test_ds, shuffle=False, **loader_kwargs),
    )


def get_dataloaders(root_dir, batch_size=64, signal_len=1024, num_workers=0):
    """Backward-compatible two-loader view (train and real unseen test)."""
    train, _, _, unseen = get_temporal_dataloaders(
        root_dir, batch_size=batch_size, signal_len=signal_len, num_workers=num_workers
    )
    return train, unseen


if __name__ == '__main__':
    loaders = get_temporal_dataloaders('./dataset/xjtu', batch_size=32)
    print([len(loader.dataset) for loader in loaders])
