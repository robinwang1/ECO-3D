import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


class ScanObjectNN(Dataset):
    def __init__(self, ROOT, subset, **kwargs):
        super().__init__()
        self.subset = subset
        self.root = ROOT
        
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        #pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        #if self.subset == 'train':
        #   np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, 0:1024].copy()
        
        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        
        return current_points, label

    def __len__(self):
        return self.points.shape[0]



class ScanObjectNN_midium(Dataset):
    def __init__(self, ROOT, subset, **kwargs):
        super().__init__()
        self.subset = subset
        self.root = ROOT
        
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmented25rot.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmented25rot.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        #pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        #if self.subset == 'train':
        #   np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, 0:1024].copy()
        

        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        
        return current_points, label

    def __len__(self):
        return self.points.shape[0]


class ScanObjectNN_hardest(Dataset):
    def __init__(self, ROOT, subset, **kwargs):
        super().__init__()
        self.subset = subset
        self.root = ROOT
        
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        #pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        #if self.subset == 'train':
        #    np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, 0:1024].copy()
        

        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        
        return current_points, label
        
    def __len__(self):
        return self.points.shape[0]