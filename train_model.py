import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.utils.data as data
from torchvision import datasets
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from model_fastvdnet import FastVDnet,FastVDnet_7,FastVDnet_9
from spiral_data import *
from arg_parser import arg_parser


class LitModel(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.encoder(x)
        loss = F.mse_loss(y, y_hat) + F.l1_loss(y,y_hat)
        ims = []
        for idx in range(x.shape[0]):
            ims.append(torch.concat((x[idx,-1,:,:],y[idx,-1,:,:],y_hat[idx,-1,:,:]),axis=1))
        
        wandb_logger.log_image(key="train_images", images=ims)
        
        l1_loss  = F.l1_loss(y,y_hat)
        mse_loss = F.mse_loss(y, y_hat)
        loss = l1_loss + mse_loss
        self.log("train_loss", loss,sync_dist=True)
        self.log("train_loss_mse", mse_loss,sync_dist=True)
        self.log("train_loss_l1", l1_loss,sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_hat = self.encoder(x)
        ims = []
        for idx in range(x.shape[0]):
            ims.append(torch.concat((x[idx,-1,:,:],y[idx,-1,:,:],y_hat[idx,-1,:,:]),axis=1))
        
        wandb_logger.log_image(key="test_images", images=ims)
        
        l1_loss  = F.l1_loss(y,y_hat)
        mse_loss = F.mse_loss(y, y_hat)
        test_loss = l1_loss + mse_loss
        self.log("test_loss", test_loss,sync_dist=True)
        self.log("test_loss_mse", mse_loss,sync_dist=True)
        self.log("test_loss_l1", l1_loss,sync_dist=True)


    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.encoder(x)
        ims = []
        for idx in range(x.shape[0]):
            ims.append(torch.concat((x[idx,-1,:,:],y[idx,-1,:,:],y_hat[idx,-1,:,:]),axis=1))
        wandb_logger.log_image(key="val_images", images=ims)

        l1_loss  = F.l1_loss(y,y_hat)
        mse_loss = F.mse_loss(y, y_hat)
        val_loss = l1_loss + mse_loss
        self.log("val_loss", val_loss,sync_dist=True)
        self.log("val_loss_mse", mse_loss,sync_dist=True)
        self.log("val_loss_l1", l1_loss,sync_dist=True)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        # return {
        # "optimizer": optimizer,
        # "lr_scheduler": {
        #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        #     "monitor": "train_loss",
        #     "frequency": 1,
        #     "interval": "epoch",
        #     # If "monitor" references validation metrics, then "frequency" should be set to a
        #     # multiple of "trainer.check_val_every_n_epoch".
        # },
        # }
        return optimizer

#configure 
config_default = arg_parser()
#loading data

for file in config_default.train_files:
    if not os.path.exists(file):
        raise RuntimeError(f"File not found: {file}")
    print(f"file exist - {file}")

total_keys = []
h5files    = []
for file in config_default.train_files:

    if not os.path.exists(file):
        raise RuntimeError(f"File not found: {file}")

    logging.info(f"reading from file: {file}")
    h5file = h5py.File(file, mode='r')
    keys = list(h5file.keys())

    ratio = [0.7,0.15,0.15]

    keys = [k+f"/{config_default.usample}" for k in keys]
    
    #fix for a bug in AJ's data gen
    keyu =[]
    for k in keys:
        if config_default.use_non_appended_keys or len(k)>15:
            keyu.append(k)

    total_keys.append(keyu)
    h5files.append(h5file)




train_set=[]
for (i,h_file) in enumerate(h5files):
    train_set.append(MRIReconSpiralDatasetTrain(h5file=[h_file], keys=[total_keys[i]], max_load=-1, data_type='2dt', conf=config_default))

#flatten
train_set = [x for xs in train_set for x in xs]

# use 20% of training data for validation
train_set_size = int(len(train_set) * ratio[0])
test_set_size = int(len(train_set) * ratio[2])

valid_set_size = int(len(train_set)-train_set_size-test_set_size)

#split the data set into three
seed = torch.Generator().manual_seed(42)
train_set, valid_set, test_set = data.random_split(train_set, [train_set_size, valid_set_size,test_set_size], generator=seed)

# model
if(config_default.time == 5):
    m = FastVDnet(config_default)
elif(config_default.time == 7):
    m = FastVDnet_7(config_default)
elif(config_default.time == 9):
    m = FastVDnet_9(config_default)

model = LitModel(m)

wandb_logger = WandbLogger(entity='gadgetron',project="FASTVDNET_dealiasing", log_model="all",name=config_default.exp_name)
wandb_logger.experiment.log({"test_set": test_set.indices})

# train model
if(config_default.cuda):
    trainer = L.Trainer(devices=config_default.gpus,
                    accelerator="gpu",
                    #strategy="deepspeed_stage_2",
                    strategy="ddp",
                    #precision="16-mixed",
                    max_epochs=config_default.epochs,
                    logger=wandb_logger,
                    log_every_n_steps=4)   
else: 
    trainer = L.Trainer(accelerator="mps", 
                    devices=1,
                    max_epochs=config_default.epochs,
                    overfit_batches=1,
                    logger=wandb_logger)


trainer.fit(model=model, train_dataloaders=DataLoader(train_set, num_workers=config_default.num_workers,batch_size=config_default.batch_size),
            val_dataloaders=DataLoader(valid_set, num_workers=config_default.num_workers,batch_size=config_default.batch_size))

trainer.test(model=model, dataloaders=DataLoader(test_set, num_workers=config_default.num_workers,batch_size=config_default.batch_size))

