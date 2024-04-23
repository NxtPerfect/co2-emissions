import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch

def loadCSV(path: str):
    data = pd.read_csv(path)
    # print(data)
    label_encoder = {}
    # Should convert everything to values
    for col in ["year", "state-name", "sector-name", "fuel-name"]:
        label_encoder[col] = LabelEncoder()
        data[col] = label_encoder[col].fit_transform(data[col])
    # print(data)
    return data

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
        return torch.tensor(self.data[idx], dtype=torch.float),torch.tensor(self.targets[idx], dtype=torch.float)

