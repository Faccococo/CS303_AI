import torch
from torch.utils.data import Dataset

class dataSet(Dataset):
    def __init__(self, feature, label):
        self.features = feature
        self.labels = label

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]