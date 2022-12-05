#%%
import torch 
import numpy as np 
import scipy as sp

class CountMatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        modeldata = torch.load(data_file)
        count_matrix, col_names = modeldata
        self.count_matrix = count_matrix
        self.col_names = col_names
    def __len__(self):
        return self.count_matrix.shape[0]
    def __getitem__(self, idx):
        obs = self.count_matrix[idx,:]
        return torch.tensor(obs.todense())
# %%
