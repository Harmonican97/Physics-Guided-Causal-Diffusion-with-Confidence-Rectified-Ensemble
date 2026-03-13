import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class XJTUGearboxDataset(Dataset):
    def __init__(self, root_dir='./dataset/xjtu', signal_len=1024, stride=None, mode='seen', normalize=True, force_reload=False):
        """
        Args:
            root_dir (str)
            signal_len (int)
            stride (int)
            mode (str)
            normalize (bool)
            force_reload (bool)
        """
        self.root_dir = root_dir
        self.signal_len = signal_len
        self.stride = stride if stride is not None else signal_len
        self.mode = mode
        self.normalize = normalize
        
        self.cache_dir = os.path.join(root_dir, 'processed_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f'xjtu_data_len{signal_len}_stride{self.stride}_norm{normalize}.pt')

        self.folder_map = {
            'ball':  {'folder': '1ndBearing_ball',                 'label': 0, 'type': 'seen'},
            'inner': {'folder': '1ndBearing_inner',                'label': 1, 'type': 'seen'},
            'outer': {'folder': '1ndBearing_outer',                'label': 2, 'type': 'seen'},
            'mix':   {'folder': '1ndBearing_mix(inner+outer+ball)','label': 3, 'type': 'unseen'}
        }

        if os.path.exists(self.cache_file) and not force_reload:
            print(f"Loading data from cache: {self.cache_file} ...")
            try:
                self.all_data = torch.load(self.cache_file)
                print("Cache loaded successfully.")
            except Exception as e:
                print(f"Error loading cache: {e}. Re-processing data...")
                self._process_and_save()
        else:
            print(f"Cache not found or reload forced. Processing raw data from {root_dir}...")
            self._process_and_save()

        self._filter_data_by_mode()

    def _process_and_save(self):
        processed_data = {}
        
        for key, info in self.folder_map.items():
            folder_name = info['folder']
            file_path = os.path.join(self.root_dir, folder_name, 'Data_Chan1.txt')
            
            print(f"Processing {folder_name}...")
            if not os.path.exists(file_path):
                print(f"  [Warning] File not found: {file_path}")
                processed_data[key] = (torch.empty(0), torch.empty(0))
                continue
            
            try:
                df = pd.read_csv(file_path, header=None, sep=r'\s+', engine='c')
                raw_signal = df.values.flatten().astype(np.float32)
            except Exception as e:
                print(f"  [Error] Failed to read {file_path}: {e}")
                continue
                
            if self.normalize:
                mean = np.mean(raw_signal)
                std = np.std(raw_signal)
                raw_signal = (raw_signal - mean) / (std + 1e-8)
                
            n_samples = (len(raw_signal) - self.signal_len) // self.stride + 1
            if n_samples <= 0:
                print(f"  [Warning] Signal too short.")
                continue
                
            indexer = np.arange(self.signal_len)[None, :] + np.arange(n_samples)[:, None] * self.stride
            sliced_data = raw_signal[indexer]
            
            data_tensor = torch.from_numpy(sliced_data).unsqueeze(1).float()
            
            label_val = info['label']
            labels_tensor = torch.full((n_samples,), label_val, dtype=torch.long)
            
            processed_data[key] = (data_tensor, labels_tensor)
            print(f"  -> Generated {n_samples} samples.")
            
        print(f"Saving processed data to {self.cache_file}...")
        torch.save(processed_data, self.cache_file)
        self.all_data = processed_data

    def _filter_data_by_mode(self):

        data_list = []
        labels_list = []
        
        for key, info in self.folder_map.items():
            if self.mode == 'seen' and info['type'] == 'unseen':
                continue
            if self.mode == 'unseen' and info['type'] == 'seen':
                continue
            
            if key in self.all_data:
                d, l = self.all_data[key]
                if len(d) > 0:
                    data_list.append(d)
                    labels_list.append(l)
        
        if len(data_list) > 0:
            self.data = torch.cat(data_list, dim=0)
            self.labels = torch.cat(labels_list, dim=0)
            print(f"Mode '{self.mode}': Loaded {len(self.data)} samples.")
        else:
            self.data = torch.empty(0)
            self.labels = torch.empty(0)
            print(f"Mode '{self.mode}': No data found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloaders(root_dir, batch_size=64, signal_len=1024, num_workers=0):

    train_dataset = XJTUGearboxDataset(
        root_dir=root_dir, 
        signal_len=signal_len, 
        stride=signal_len, 
        mode='seen'
    )
    
    test_dataset = XJTUGearboxDataset(
        root_dir=root_dir, 
        signal_len=signal_len, 
        stride=signal_len, 
        mode='unseen'
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True 
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader

if __name__ == "__main__":

    if os.path.exists('./dataset/xjtu'):
        print("Testing optimized loader...")
        tl, vl = get_dataloaders('./dataset/xjtu', batch_size=32)
        
        import time
        start = time.time()
        for x, y in tl:
            pass
        print(f"Iterate through train loader time: {time.time() - start:.4f}s")