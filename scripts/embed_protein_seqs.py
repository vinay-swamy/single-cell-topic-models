#%%

import pandas as pd
import sys
import pickle 
import torch 
import esm
import pytorch_lightning as pl
import subprocess as sp 
import os 
_, alt_seq_file,batch_size,outpath = sys.argv 

class LitESM(pl.LightningModule):
    def __init__(self, model, seq_df):
        super().__init__()     
        self.model = model
        self.seq_df = seq_df

    def forward(self, batch):
        ## bseqs is a list of tuples(transcript_id, protein_seq)
        idx, batch_tokens = batch
        results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
        results = results.cpu()
        idx = idx.cpu()
        for pos_j, orig_idx in enumerate(idx.numpy()):
            outpath = self.seq_df['outpaths'].iloc[orig_idx]
            seq_len = self.seq_df['aa_seq_len'].iloc[orig_idx]
            transcript_id = self.seq_df['transcript_id'].iloc[orig_idx]
            embedding = results[pos_j, 1 : seq_len + 1]
            col_mean_embedding = embedding.mean(axis = 0).cpu().numpy()
            with open(outpath, "wb+") as ofl:
                pickle.dump(col_mean_embedding, ofl)
        return 
class CustomDL:
    def __init__(self, data, batchsize, converter):
        self.data = data 
        self.batch_size = batchsize
        self.len = len(data)
        self.converter = converter
    def __len__(self):
        return self.len
    def __iter__(self):
        for i in range(0,self.len, self.batch_size):
            p_idx,_, p_tokens = self.converter(self.data[i:i+self.batch_size])
            yield torch.tensor(p_idx), p_tokens

all_seq_df = (pd.read_csv(alt_seq_file)
      .pipe(lambda x: x[~x.protein_seq.isna()])
      .assign(aa_seq_len = lambda x: x.aa_seq_len.astype(int))
      .sort_values("aa_seq_len")
      .reset_index(drop = True)
      )

outpaths = outpath + "/" + all_seq_df['sampleid'] + "/" + all_seq_df['transcript_id'] + ".pickle"
all_seq_df = all_seq_df.assign(outpaths = outpaths)
all_outdirs = outpath + "/" + all_seq_df['sampleid']
make_dirs = True
check_output_exists = True 

if make_dirs:
    all_outdirs.drop_duplicates().apply(lambda x: sp.run(f"mkdir -p {x}", shell = True) )
if check_output_exists:
    n_completed = all_seq_df.outpaths.apply(lambda x: os.path.exists(x)).sum()
    all_seq_df = all_seq_df[~all_seq_df.outpaths.apply(lambda x: os.path.exists(x))].reset_index(drop = True)
    if all_seq_df.shape[0] == 0:
        print("All embedding already generated")
        sys.exit(0)
    print(f'skipping {n_completed} seqs, {all_seq_df.shape[0]} remaining')


# #if mode == "lightning":
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
tokenizer = alphabet.get_batch_converter()
all_seq_df = all_seq_df.reset_index(drop = True) ##
litmodel = LitESM(model, all_seq_df)
protein_seqs = list(enumerate(all_seq_df.protein_seq.to_numpy()))
batch_size = int(batch_size)
p_dl = CustomDL(protein_seqs, batch_size, tokenizer)
trainer = pl.Trainer(accelerator='gpu', devices=[0,1], strategy='ddp')
trainer.predict(litmodel,p_dl, return_predictions=False)
# #elif mode == 'fsdp':

