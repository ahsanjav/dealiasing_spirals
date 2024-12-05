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
from utils.util_func import real_imag2complex
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure


# Define the LIT model
class LitModel(L.LightningModule):
    def __init__(self, encoder,config_default,wandb_logger):
        super().__init__()
        self.encoder = encoder
        self.config_default = config_default
        self.wandb_logger   = wandb_logger

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.encoder(x)
        loss = F.mse_loss(y, y_hat) + F.l1_loss(y,y_hat)
        ims = []
        for idx in range(x.shape[0]):
            if(self.config_default.complex_i):
                ims.append(torch.abs(real_imag2complex(torch.concat((x[idx,-2:,:,:],y[idx,:,:,:],y_hat[idx,:,:,:]),axis=2),axis=0)))
            else:
                ims.append(torch.concat((x[idx,-1,:,:],y[idx,-1,:,:],y_hat[idx,-1,:,:]),axis=1))
        
        self.wandb_logger.log_image(key="train_images", images=ims)
        
        l1_loss  = F.l1_loss(y,y_hat)
        mse_loss = F.mse_loss(y, y_hat)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0,kernel_size=3,sigma=0.5).to(x.device)

        if(self.config_default.complex_i):
            pred = torch.abs(real_imag2complex(y,axis=1))
            target = torch.abs(real_imag2complex(y_hat,axis=1))
            pred = pred[:,np.newaxis,:,:]
            target = target[:,np.newaxis,:,:]
        else:
            pred = torch.abs(y)
            target = torch.abs(y_hat)
        ssim_loss = ssim(pred,target)
        
        
        loss = l1_loss #+ mse_loss

        if(self.config_default.ssim_only):
            loss += 1 - ssim_loss
        
        self.log("train_loss_mse", mse_loss,sync_dist=True)
        self.log("train_loss_l1", l1_loss,sync_dist=True)
        
        self.log("train_loss", loss,sync_dist=True)
        self.log("train_loss_ssim", ssim_loss,sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_hat = self.encoder(x)
        ims = []
        for idx in range(x.shape[0]):
            if(self.config_default.complex_i):
                ims.append(torch.abs(real_imag2complex(torch.concat((x[idx,-2:,:,:],y[idx,:,:,:],y_hat[idx,:,:,:]),axis=2),axis=0)))
            else:
                ims.append(torch.concat((x[idx,-1,:,:],y[idx,-1,:,:],y_hat[idx,-1,:,:]),axis=1))
                
        self.wandb_logger.log_image(key="test_images", images=ims)
        
        l1_loss  = F.l1_loss(y,y_hat)
        mse_loss = F.mse_loss(y, y_hat)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0,kernel_size=3,sigma=0.5).to(x.device)
        if(self.config_default.complex_i):
            pred = torch.abs(real_imag2complex(y,axis=1))
            target = torch.abs(real_imag2complex(y_hat,axis=1))
            pred = pred[:,np.newaxis,:,:]
            target = target[:,np.newaxis,:,:]
        else:
            pred = torch.abs(y)
            target = torch.abs(y_hat)
        ssim_loss = ssim(pred,target)  

        test_loss = l1_loss #+ mse_loss

        if(self.config_default.ssim_only):
            test_loss += 1 - ssim_loss
        
        self.log("test_loss_mse", mse_loss,sync_dist=True)
        self.log("test_loss_l1", l1_loss,sync_dist=True)

        self.log("test_loss_ssim", ssim_loss,sync_dist=True)
        self.log("test_loss", test_loss,sync_dist=True)



    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.encoder(x)
        ims = []
        for idx in range(x.shape[0]):
            if(self.config_default.complex_i):
                ims.append(torch.abs(real_imag2complex(torch.concat((x[idx,-2:,:,:],y[idx,:,:,:],y_hat[idx,:,:,:]),axis=2),axis=0)))
            else:
                ims.append(torch.concat((x[idx,-1,:,:],y[idx,-1,:,:],y_hat[idx,-1,:,:]),axis=1))
        
        self.wandb_logger.log_image(key="val_images", images=ims)

        l1_loss  = F.l1_loss(y,y_hat)
        mse_loss = F.mse_loss(y, y_hat)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0,kernel_size=3,sigma=0.5).to(x.device)
        if(self.config_default.complex_i):
            pred = torch.abs(real_imag2complex(y,axis=1))
            target = torch.abs(real_imag2complex(y_hat,axis=1))
            pred = pred[:,np.newaxis,:,:]
            target = target[:,np.newaxis,:,:]
        else:
            pred = torch.abs(y)
            target = torch.abs(y_hat)
        ssim_loss = ssim(pred,target)
        val_loss = l1_loss #+ mse_loss


        if(self.config_default.ssim_only):
            val_loss += 1 - ssim_loss

        self.log("val_loss_mse", mse_loss,sync_dist=True)
        self.log("val_loss_l1", l1_loss,sync_dist=True)
    
        self.log("val_loss", val_loss,sync_dist=True)
        self.log("val_ssim_loss", ssim_loss,sync_dist=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config_default.lr)
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
