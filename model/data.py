#%%
import torch 
import numpy as np 
import scipy as sp
import pandas as pd
from collections.abc import Mapping, Sequence
from typing import List, Optional, Union
from torch.utils.data.dataloader import default_collate

class CountMatrixDataset(torch.utils.data.Dataset):
    def __init__(self, md, count_data):
        metadata = pd.read_csv(md)
        self.md = metadata
        count_matrix = torch.load(count_data)
        self.count_matrix = count_matrix
        
    def __len__(self):
        return self.md.shape[0]
    def __getitem__(self, idx):
        matrix_idx = self.md['matrix_idx'].iloc[idx]
        grp = self.md['group'].iloc[idx]
        obs = self.count_matrix[matrix_idx,:]
        return torch.tensor(obs.todense()).type(torch.float), idx, grp
# %%
class Collater:
    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            return torch.concat(batch)
        elif  isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # Deprecated...
        return self(batch)