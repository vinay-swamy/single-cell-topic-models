#%%
import torch 
import pytorch_lightning as pl 
import torch.nn.functional as F
from layers import *
import numpy as np 
import pandas as pd 
from data import CountMatrixDataset, Collater
from torch.utils.data import DataLoader
from itertools import combinations

NUM_WORKERS=3
class LitDataModule(pl.LightningDataModule):
    def __init__(self, mconfig):
        super().__init__()
        data_conf = mconfig['data']
        count_file = f"{data_conf['root_dir']}/{data_conf['count_file']}"
        self.train_ds = CountMatrixDataset(f"{data_conf['root_dir']}/{data_conf['train_md']}", count_file)
        self.val_ds = CountMatrixDataset(f"{data_conf['root_dir']}/{data_conf['val_md']}",count_file)
        self.test_ds = CountMatrixDataset(f"{data_conf['root_dir']}/{data_conf['test_md']}",count_file)
        self.batch_size = data_conf['batch_size']
        
    def train_dataloader(self): 
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,
            shuffle = True,
            num_workers=NUM_WORKERS,
            collate_fn=Collater()
        )
    def val_dataloader(self): 
        return DataLoader(
            self.val_ds, batch_size=self.batch_size,
            shuffle = False,
            num_workers=NUM_WORKERS,
            collate_fn=Collater()
        )
    def test_dataloader(self): 
        return DataLoader(
            self.test_ds, batch_size=self.batch_size,
            shuffle = False,
            num_workers=NUM_WORKERS,
            collate_fn=Collater()
        )
        


class LitVAE(pl.LightningModule):
    '''
    https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    https://github.com/williamFalcon/pytorch-lightning-vae/blob/main/vae.py
    '''
    def __init__(self,  model_config,  optim_config):
        super().__init__()

        #self.loss_fn = _loss_fn(**loss_config['loss_fn_kwargs'])
        self.optim_config = optim_config
        self.encoder = eval(model_config['encoder_fn'])(**model_config['encoder_kwargs'])
        self.decoder = eval(model_config['decoder_fn'])(**model_config['decoder_kwargs'])
        self.reparameterizer = eval(model_config['reparameterizer'])()
        self.val_X_hat = None
        self.val_theta = None
        self.val_beta  = None
        self.val_s_idx = None
        self.tau = 0.325
        self.n_topics = model_config['encoder_kwargs']['n_topics']
    def forward(self, X):
         ## the tensor you choose MUST be the same datatype as the tensor you are creating 
        latent_dist_params = self.encoder(X)
        ###
        # : using definging the distribution q and the sampling to obtain 
        # z via `q.rsample()` internally implements the re-parameterization trick
        # so gradients should be able to work here
        # https://pytorch.org/docs/stable/distributions.html
        theta_logits = self.reparameterizer(latent_dist_params)
        X_hat, theta = self.decoder(theta_logits)
        RC_loss = self.reconstruction_loss(X,X_hat)
        KL_loss = self.reparameterizer.KL(theta, latent_dist_params)
        elbo =  (KL_loss *(self.tau) )  + RC_loss

        return elbo, KL_loss, RC_loss, theta, X_hat
        
    
    def training_step(self, batch, batch_idx):
        ## X is batch of tokenized and padded protein seqs
        X, s_idx, group=batch
        batch_size = X.shape[0]
        elbo, KL_loss, RC_loss, theta, X_hat = self.forward(X)
        loss = elbo.mean() ## sum ?
        self.log("train_avg_ELBO_loss", loss, batch_size=batch_size )
        self.log("train_avg_RC_loss", RC_loss.mean(), batch_size=batch_size )
        self.log("train_avg_KL_loss", KL_loss.mean(), batch_size=batch_size)
        return {"loss": loss, 
                "x_hat":X_hat.detach().cpu().numpy(), 
                "theta": theta.detach().cpu().numpy(),
                "s_idx":s_idx.detach().cpu().numpy(),
                "group":group
                }
    def validation_step(self, batch, batch_idx):
        X, s_idx, group=batch
        batch_size = X.shape[0]
        elbo, KL_loss, RC_loss, theta, X_hat = self.forward(X)
        loss = elbo.mean() ## sum ?
        self.log("val_avg_ELBO_loss", loss, batch_size=batch_size )
        self.log("val_avg_RC_loss", RC_loss.mean(), batch_size=batch_size )
        self.log("val_avg_KL_loss", KL_loss.mean(), batch_size=batch_size )
        return {"loss": loss, 
                "x_hat":X_hat.detach().cpu().numpy(), 
                "theta": theta.detach().cpu().numpy(),
                "s_idx":s_idx.detach().cpu().numpy(),
                "group":group
                }
    
    def validation_epoch_end(self, validation_step_outputs):    
        self.calc_per_epoch_metrics(validation_step_outputs, "val")
        return
    def predict_step(self,batch, batch_idx):
        X, s_idx, group=batch
        batch_size = X.shape[0]
        elbo, KL_loss, RC_loss, theta, X_hat = self.forward(X)
        loss = elbo.mean() ## sum ?
        return {"loss": loss, 
                "x_hat":X_hat.detach().cpu().numpy(), 
                "theta": theta.detach().cpu().numpy(),
                "s_idx":s_idx.detach().cpu().numpy(),
                "group":group
                }
    def predict_epoch_end(self,ops):
        return ops 

    
    def training_epoch_end(self, train_step_outputs):
        self.calc_per_epoch_metrics(train_step_outputs, "train")
        return

    def calc_per_epoch_metrics(self, step_output, dtype):
        ### calculate per group variance:
        ## for each group, calculate the proportion std per topic
        ## and take the median of these values 
        topics_per_group = {}
        for batch_output in step_output:
            b_theta = batch_output['theta']
            b_group = batch_output['group']
            for i in range(b_theta.shape[0]):
                g = b_group[i]
                t = b_theta[i]
                if g in topics_per_group:
                    topics_per_group[g].append(t)
                else:
                    topics_per_group[g]=[t]

        all_theta_topic_jac = []
        all_theta_topic_cor = []
        for group in topics_per_group.values():
            ## theta is cells x topics
            all_group_theta = np.array(group)
            group_theta_jac = topn_pw_jac(all_group_theta, int(self.n_topics/10), 1000)
            all_theta_topic_jac.append(group_theta_jac)
            cor_mat=  pd.DataFrame(all_group_theta.T).corr(method = "spearman").to_numpy()
            med_group_theta_cor = np.median(cor_mat[np.triu_indices(cor_mat.shape[0], k=1 )])
            all_theta_topic_cor.append(med_group_theta_cor)


        med_theta_jac = np.median(all_theta_topic_jac)
        med_theta_corr = np.median(all_theta_topic_cor)
        ## calculate pairwise correlation between topics
        beta = self.decoder.get_beta().detach().cpu().numpy()
        cor_mat=  pd.DataFrame(beta.T).corr(method = "spearman").to_numpy()
        med_beta_cor = np.median(cor_mat[np.triu_indices(cor_mat.shape[0], k=1 )])
        med_beta_jac = topn_pw_jac(beta, 100, 1000000)
    
        self.log(f"{dtype}_med_group_theta_topic_corr", med_theta_corr)
        self.log(f"{dtype}_med_group_theta_topic_jac", med_theta_jac)
        self.log(f"{dtype}_med_pw_beta_gene_corr", med_beta_cor)
        self.log(f"{dtype}_med_pw_beta_gene_jac", med_beta_jac)
        return 
        

    ## need to dig a little more into this 
    def reconstruction_loss(self, obs_counts,X_hat):
        recon_loss = -torch.sum(obs_counts * torch.log(X_hat + 1e-10), dim=1)
        
        return recon_loss
    
    def configure_optimizers(self):
        optim_fn = eval(self.optim_config['optim_fn'])
        optimizer = optim_fn(self.parameters(), **self.optim_config['optim_kwargs'])
        ret_val = {'optimizer': optimizer}

        # Add a lr_scheduler if specified in configs
        # if ('lr_scheduler' in self.optim_config) and (self.optim_config['lr_scheduler'] is not None):
        #     sched_fn = self.optim_config['lr_scheduler']
        #     scheduler = sched_fn(optimizer, **self.optim_config['lr_scheduler_kwargs'])
        #     lr_scheduler_configs = {"scheduler": scheduler}
        #     lr_scheduler_configs.update(**self.optim_config['lr_scheduler_configs'])
        #     ret_val['lr_scheduler'] = lr_scheduler_configs

        return ret_val
# %%
def jaccard_like(l1, l2):
    intersect = len(l1.intersection(l2))
    return intersect / len(l2)
def topn_pw_jac(all_group_theta, n, ss):
    pairs = np.array(list(combinations(list(range(all_group_theta.shape[0])), 2)))
    if len(pairs) > ss:
        idxs = np.random.choice(len(pairs), size = ss)
        pairs = pairs[idxs]
    res = []
    for p in pairs:
        l = set(np.argpartition(all_group_theta[p[0], :], -n)[-n:])
        r = set(np.argpartition(all_group_theta[p[1], :], -n)[-n:])
        res.append(jaccard_like(l,r))
    return np.median(res)