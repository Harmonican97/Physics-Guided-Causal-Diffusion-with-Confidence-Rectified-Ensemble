import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class XJTUGearboxDataset(Dataset):
    """
    带缓存加速的 XJTU Gearbox 数据集加载器。
    机制：首次运行读取 txt -> 处理 -> 保存为 .pt -> 下次直接读取 .pt
    """
    def __init__(self, root_dir='./dataset/xjtu', signal_len=1024, stride=None, mode='seen', normalize=True, force_reload=False):
        """
        Args:
            root_dir (str): 数据集根目录
            signal_len (int): 样本长度
            stride (int): 滑动窗口步长
            mode (str): 'seen' (训练/单故障) 或 'unseen' (测试/复合故障)
            normalize (bool): 是否标准化
            force_reload (bool): 是否强制重新生成缓存
        """
        self.root_dir = root_dir
        self.signal_len = signal_len
        self.stride = stride if stride is not None else signal_len
        self.mode = mode
        self.normalize = normalize
        
        # 定义缓存文件路径 (包含 signal_len 防止参数改变导致形状不匹配)
        self.cache_dir = os.path.join(root_dir, 'processed_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f'xjtu_data_len{signal_len}_stride{self.stride}_norm{normalize}.pt')

        # 文件夹映射
        self.folder_map = {
            'ball':  {'folder': '1ndBearing_ball',                 'label': 0, 'type': 'seen'},
            'inner': {'folder': '1ndBearing_inner',                'label': 1, 'type': 'seen'},
            'outer': {'folder': '1ndBearing_outer',                'label': 2, 'type': 'seen'},
            'mix':   {'folder': '1ndBearing_mix(inner+outer+ball)','label': 3, 'type': 'unseen'}
        }

        # 检查缓存是否存在
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

        # 根据 mode 筛选数据
        self._filter_data_by_mode()

    def _process_and_save(self):
        """
        读取原始 txt 文件，处理并保存为字典缓存
        """
        processed_data = {} # 存储所有类型的数据 {'ball': (data, labels), ...}
        
        for key, info in self.folder_map.items():
            folder_name = info['folder']
            file_path = os.path.join(self.root_dir, folder_name, 'Data_Chan1.txt')
            
            print(f"Processing {folder_name}...")
            if not os.path.exists(file_path):
                print(f"  [Warning] File not found: {file_path}")
                processed_data[key] = (torch.empty(0), torch.empty(0))
                continue
            
            # 1. 使用 Pandas 加速读取 (比 numpy loadtxt 快很多)
            try:
                # engine='c' 是最快的，header=None 假设无表头
                df = pd.read_csv(file_path, header=None, sep=r'\s+', engine='c')
                raw_signal = df.values.flatten().astype(np.float32)
            except Exception as e:
                print(f"  [Error] Failed to read {file_path}: {e}")
                continue
                
            # 2. 标准化 (Z-Score)
            if self.normalize:
                mean = np.mean(raw_signal)
                std = np.std(raw_signal)
                raw_signal = (raw_signal - mean) / (std + 1e-8)
                
            # 3. 向量化切片 (Vectorized Slicing) - 极快
            # 计算样本数量
            n_samples = (len(raw_signal) - self.signal_len) // self.stride + 1
            if n_samples <= 0:
                print(f"  [Warning] Signal too short.")
                continue
                
            # 利用 stride tricks 或索引数组进行快速切片
            # 这里的 shape 是 (n_samples, signal_len)
            indexer = np.arange(self.signal_len)[None, :] + np.arange(n_samples)[:, None] * self.stride
            sliced_data = raw_signal[indexer]
            
            # 转为 Tensor: (N, 1, L)
            data_tensor = torch.from_numpy(sliced_data).unsqueeze(1).float()
            
            # 生成标签
            label_val = info['label']
            labels_tensor = torch.full((n_samples,), label_val, dtype=torch.long)
            
            processed_data[key] = (data_tensor, labels_tensor)
            print(f"  -> Generated {n_samples} samples.")
            
        # 保存到磁盘
        print(f"Saving processed data to {self.cache_file}...")
        torch.save(processed_data, self.cache_file)
        self.all_data = processed_data

    def _filter_data_by_mode(self):
        """
        根据 mode ('seen' or 'unseen') 从内存中的 all_data 筛选需要的数据
        """
        data_list = []
        labels_list = []
        
        for key, info in self.folder_map.items():
            # 逻辑判断：是否加载该类别
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
    """
    获取 DataLoader。
    num_workers 默认为 0 (Windows下最稳妥)。
    如果是在 Linux 服务器上，可以设为 4 或 8。
    """
    # 1. 训练集 (Seen Classes)
    # 训练时通常数据不重叠，保持 stride=signal_len
    train_dataset = XJTUGearboxDataset(
        root_dir=root_dir, 
        signal_len=signal_len, 
        stride=signal_len, 
        mode='seen'
    )
    
    # 2. 测试集 (Unseen Class)
    # 测试集也不重叠
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
        pin_memory=True # 加速数据从 CPU 传输到 GPU
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader

# 测试代码
if __name__ == "__main__":
    # 第一次运行会生成缓存 (慢)
    # 第二次运行会直接加载缓存 (快)
    if os.path.exists('./dataset/xjtu'):
        print("Testing optimized loader...")
        tl, vl = get_dataloaders('./dataset/xjtu', batch_size=32)
        
        import time
        start = time.time()
        for x, y in tl:
            pass
        print(f"Iterate through train loader time: {time.time() - start:.4f}s")