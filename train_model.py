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
from model_fastvdnet import FastVDnet
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
        loss = F.mse_loss(y, y_hat)
        ims = []
        for idx in range(x.shape[0]):
            ims.append(torch.concat((x[idx,-1,:,:],y[idx,-1,:,:],y_hat[idx,-1,:,:]),axis=1))
        
        wandb_logger.log_image(key="train_images", images=ims)
        self.log("train_loss", loss)

        return loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_hat = self.encoder(x)
        ims = []
        for idx in range(x.shape[0]):
            ims.append(torch.concat((x[idx,-1,:,:],y[idx,-1,:,:],y_hat[idx,-1,:,:]),axis=1))
        
        wandb_logger.log_image(key="test_images", images=ims)
        test_loss = F.mse_loss(y, y_hat)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.encoder(x)
        ims = []
        for idx in range(x.shape[0]):
            ims.append(torch.concat((x[idx,-1,:,:],y[idx,-1,:,:],y_hat[idx,-1,:,:]),axis=1))
        wandb_logger.log_image(key="val_images", images=ims)

        val_loss = F.mse_loss(y, y_hat)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

#configure 
config_default = arg_parser()
#loading data

for file in config_default.train_files:
    if not os.path.exists(file):
        raise RuntimeError(f"File not found: {file}")
    print(f"file exist - {file}")

for file in config_default.train_files:
    
    total_keys = []
    h5files    = []
    if not os.path.exists(file):
        raise RuntimeError(f"File not found: {file}")

    logging.info(f"reading from file: {file}")
    h5file = h5py.File(file, libver='earliest', mode='r')
    keys = list(h5file.keys())

    ratio = [0.7,0.2,0.1]

    keys = [k+f"/{config_default.usample}" for k in keys]
    total_keys.append(keys)
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
model = LitModel(FastVDnet(config_default))

wandb_logger = WandbLogger(entity='gadgetron',project="FASTVDNET_dealiasing", log_model="all",name='Testing models')

# train model
if(config_default.cuda):
    trainer = L.Trainer(devices=4,
                    accelerator="gpu",
                    strategy="deepspeed_stage_2",
                    precision="16-mixed",
                    max_epochs=config_default.epochs,
                    logger=wandb_logger)   
else: 
    trainer = L.Trainer(accelerator="mps", 
                    devices=1,
                    max_epochs=config_default.epochs,
                    overfit_batches=1,
                    logger=wandb_logger)


trainer.fit(model=model, train_dataloaders=DataLoader(train_set, batch_size=config_default.batch_size),
            val_dataloaders=DataLoader(valid_set, batch_size=config_default.batch_size))

trainer.test(model=model, dataloaders=DataLoader(test_set, batch_size=config_default.batch_size))

