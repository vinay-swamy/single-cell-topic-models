#%%
import torch 
import pytorch_lightning as pl 
import torch.nn.functional as F
from layers import *

class LitDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        


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
    def forward(self, X):
         ## the tensor you choose MUST be the same datatype as the tensor you are creating 
        latent_dist_params = self.encoder(X)
        ###
        # : using definging the distribution q and the sampling to obtain 
        # z via `q.rsample()` internally implements the re-parameterization trick
        # so gradients should be able to work here
        # https://pytorch.org/docs/stable/distributions.html
        theta = self.reparameterizer(latent_dist_params)
        X_hat, beta = self.decoder(theta)
        RC_loss = self.reconstruction_loss(X,X_hat)
        KL_loss = self.reparameterizer.KL(theta, latent_dist_params)
        elbo =  KL_loss + RC_loss
        #elbo = RC_loss

        return elbo, KL_loss, RC_loss, theta, beta 
        
    
    def training_step(self, batch, batch_idx):
        ## X is batch of tokenized and padded protein seqs
        X=batch
        elbo, KL_loss, RC_loss, _,= self.forward(X)
        loss = elbo.mean() ## sum ?
        self.log("train_avg_ELBO_loss", loss)
        self.log("train_avg_RC_loss", RC_loss.mean())
        self.log("train_avg_KL_loss", KL_loss.mean())
        return loss
    def validation_step(self, batch, batch_idx):
        X=batch
        elbo, KL_loss, RC_loss, _,= self.forward(X)
        loss = elbo.mean() ## sum ?
        self.log(f"val_avg_ELBO_loss", loss)
        self.log(f"val_avg_RC_loss", RC_loss.mean())
        self.log(f"val_avg_KL_loss", KL_loss.mean())
        return 

    ## need to dig a little more into this 
    def reconstruction_loss(self, X_embedded,X_hat):
        recon_loss = -(X_hat * X_embedded).sum(1)
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
