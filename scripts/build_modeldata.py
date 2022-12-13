#%%
import pandas as pd 
import numpy as np 
import scipy as sp
import pickle
import torch
import glob
import sys
from sklearn.model_selection import train_test_split
#stem ="/data/vss2134/scTopic/data/sciplex_counts_5kg"
#%%
stem = sys.argv[1]

allcounts = sp.io.mmread(f"{stem}/counts.mtx").tocsr()
#%%
target_genes = pd.read_csv(f"{stem}/counts.mtx.colnames", names = ['gene_id']).assign(
    gene_id = lambda x: x.gene_id.str.replace("\.\d+", "")
)
#%%
cell_barcodes = pd.read_csv(f"{stem}/counts.mtx.rownames", names = ['barcode'])

# %%
avail_proteins = pd.DataFrame().assign(path = glob.glob("/data/vss2134/scTopic/data/protein_embedding/GRCH38/*")).assign(
    transcript_id = lambda x: x.path.str.findall("ENST\d+").apply(lambda x: x[0]) )
# %%
gene_mappings = pd.read_csv("/data/vss2134/scTopic/data/grch38_protein_seqs.csv").merge(
    target_genes
).merge(
    avail_proteins
)
# %%
assert gene_mappings.shape[0] == target_genes.shape[0]
gene_mappings_matched = gene_mappings.set_index("gene_id").loc[target_genes['gene_id'], :]


# %%
def read_pickle(f):
    with open(f, 'rb')as infl:
        return pickle.load(infl)
# %%
rho_tensor = torch.tensor(np.array([read_pickle(f) for f in gene_mappings_matched.path.to_numpy()]))
# %%



# %%
cell_meta_data = pd.read_csv("/data/vss2134/scTopic/data/GSM4150378_sciPlex3_pData.txt", sep = " ").pipe(
    lambda x: x[x.cell.isin(cell_barcodes['barcode'])]
).set_index("cell").loc[cell_barcodes['barcode'], :].assign(
    matrix_idx = list(range(cell_barcodes.shape[0]))
).reset_index(drop = False)
assert cell_meta_data.shape[0] == cell_barcodes.shape[0]
# %%
drop_groups = cell_meta_data.groupby(["cell_type", "dose_pattern", "dose", "treatment"]).size().sort_values()
drop_groups = drop_groups[drop_groups< 100].reset_index(drop = False)

drop_barcodes = cell_meta_data.merge(drop_groups, how = 'inner')['cell']
cell_meta_data_filtered = cell_meta_data.pipe(
    lambda x: x[~x.cell.isin(drop_barcodes)]
).assign(
    group = lambda x: x["cell_type"] + ":" +  x["dose_pattern"].astype(str) + ":" + x["dose"].astype(str) + ":" + x["treatment"]
).pipe(lambda x: x[~x.cell_type.isna()])
#%%



# %%
test_val_split = cell_meta_data_filtered.groupby(["cell_type", "dose_pattern", "dose", "treatment"]).sample(frac = .2, random_state = 234)
train_split = cell_meta_data_filtered.pipe(
    lambda x: x[~x.cell.isin(test_val_split['cell'])]
)


#%%
test_split = test_val_split.groupby(["cell_type", "dose_pattern", "dose", "treatment"]).sample(frac = .5, random_state = 1233123)
val_split = test_val_split.pipe(lambda x: x[~x.cell.isin(test_split['cell'])])


# %%
torch.save(allcounts, f"{stem}/count.pt")
torch.save(rho_tensor, f"{stem}/rho_tensor.pt")
train_split.to_csv(f"{stem}/train.csv", index = False)
val_split.to_csv(f"{stem}/val.csv", index = False)
test_split.to_csv(f"{stem}/test.csv", index = False)
# %%
