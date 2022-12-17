#%%
from lightning_modules import *

from pathlib import Path 
import sys
import json 
import wandb
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

#%%


args = sys.argv[1:]
if len(args) == 1:
    ## using a run 
    configfile =  args[0]
    with open(configfile) as m_stream:
        mconfig = json.load(m_stream)
elif args[1] == "TEST":
    configfile =  args[0]
    with open(configfile) as m_stream:
        mconfig = json.load(m_stream)
    mconfig['trainer']['max_epochs']=1
    #mconfig['data']["max_steps_per_epoch"]=2000
    mconfig['wandb_project']='test'
    # mconfig['training']['devices']=1
    # del mconfig['training']['strategy']



#%%

## Load data
data_conf=mconfig['data']
datamodule = LitDataModule(mconfig)
train_dl = datamodule.train_dataloader()
val_dl = datamodule.val_dataloader()
#test_dl = datamodule.test_dataloader()

## instance model 

litmodel = LitVAE(mconfig['model'], mconfig['optim'])
#%%


## instance trainer
run_name = os.path.splitext(configfile)[0].split("/")[-1]
plg= WandbLogger(project = mconfig['wandb_project'],
                 name = run_name,
                 entity = 'vinay-swamy', 
                 config=mconfig, 
                 save_dir = f"/data/vss2134/scTopic/model_out")

plg.watch(litmodel)
checkpoint_cb = ModelCheckpoint(save_top_k=-1, every_n_epochs = None,every_n_train_steps = None, train_time_interval = None)


trainer_conf = mconfig['trainer']
trainer_conf['logger'] = plg
#lr_monitor = LearningRateMonitor(logging_interval='step')

trainer_conf['callbacks'] = [checkpoint_cb]
trainer = pl.Trainer(**trainer_conf)

if mconfig['dryrun']:
    print("Successfully loaded everything. Quitting")
    sys.exit()

trainer.fit(litmodel, train_dataloaders = train_dl, val_dataloaders=val_dl)
print("Run Completed Succsessfully.")

# if mconfig['run_test']:
#     trainer.test(litmodel, test_dl)

# %%