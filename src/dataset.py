import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

def loadCSV(path: str):
    data = pd.read_csv(path)
    print(data)

class CO2Dataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values[:, :-1]
        self.targets = dataframe.values[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
                'features' : torch.tensor(self.data[idx], dtype=torch.float),
                'target': torch.tensor(self.targets[idx], dtype=torch.float)
                }
        return sample
