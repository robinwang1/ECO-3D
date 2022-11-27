import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
# from .build import DATASETS
import torch


# warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class RobustPointSet(Dataset):
    def __init__(self, DATA_PATH, TRANS, subset, N_POINTS=1024):
        self.root = DATA_PATH
        self.npoints = N_POINTS
        self.trans = TRANS
        split = subset
        self.subset = split
        self.label_path = os.path.join(self.root, 'labels_%s.npy' % split)
        self.data_path = os.path.join(self.root, '%s_%s.npy' % (split, self.trans))
        self.point_set, self.label = np.load(self.data_path), np.load(self.label_path)

    def __len__(self):
        return np.load(self.data_path).shape[0]

    def __getitem__(self, index):
        point_set = self.point_set[index, 0:self.npoints, :].copy()
        label = self.label[index, 0]
        #point_set = pc_normalize(point_set)
        return point_set, label


if __name__ == '__main__':
    dataset_train = RobustPointSet('train')
    dataset_test = RobustPointSet('test')
    trainloader = \
        torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    testloader = \
        torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    for point, label in tqdm(trainloader):
        # print(point.shape)
        # print(label.shape)
        pass
    for point, label in tqdm(testloader):
        # print(point.shape)
        # print(label.shape)
        pass