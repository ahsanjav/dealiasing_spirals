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
from LITmodel import LitModel
from torch.utils.data import ConcatDataset
import torch._dynamo
# Code run 
TORCH_LOGS="+dynamo"
TORCHDYNAMO_VERBOSE=1
#configure from arg parser
config_default = arg_parser()
torch.set_float32_matmul_precision("medium")
torch._dynamo.config.suppress_errors = True
#loading data
for file in config_default.train_files:
    if not os.path.exists(file):
        raise RuntimeError(f"File not found: {file}")
    print(f"file exist - {file}")

total_keys = []
h5files    = []
for file in config_default.train_files:
#check files if they exist
    if not os.path.exists(file):
        raise RuntimeError(f"File not found: {file}")
#Get all keys
    logging.info(f"reading from file: {file}")
    with h5py.File(file, libver='earliest',mode='r') as f:
        keys = list(f.keys())

        ratio = [0.7,0.15,0.15]

    #Only get the data for the undersampling rate specified in training
        keys = [k+f"/{config_default.usample}" for k in keys]
        


        #fix for a bug in AJ's data gen
        keyu =[]
        for k in keys:
            if config_default.use_non_appended_keys or len(k)>15:
                keyu.append(k)

        total_keys.append(keyu)
        h5files.append(file)




train_set=[]
for (i,h_file) in enumerate(h5files):
    train_set.append(MRIReconSpiralDatasetTrain(h5file=[h_file], keys=[total_keys[i]], max_load=-1, data_type='2dt', conf=config_default))

#flatten
train_set = ConcatDataset(train_set)#[x for xs in train_set for x in xs]

# use 20% of training data for validation
train_set_size = int(len(train_set) * ratio[0])
test_set_size = int(len(train_set) * ratio[2])

valid_set_size = int(len(train_set)-train_set_size-test_set_size)

#split the data set into three
seed = torch.Generator().manual_seed(42)
train_set, valid_set, test_set = data.random_split(train_set, [train_set_size, valid_set_size,test_set_size], generator=seed)

# model choice
if(config_default.time == 5):
    m = FastVDnet(config_default)
elif(config_default.time == 7):
    m = FastVDnet_7(config_default)
elif(config_default.time == 9):
    m = FastVDnet_9(config_default)

wandb_logger = WandbLogger(entity='gadgetron',project="FASTVDNET_dealiasing", log_model="all",name=config_default.exp_name)
wandb_logger.experiment.log({"test_set": test_set.indices})

model = LitModel(m,config_default,wandb_logger)


model = torch.compile(model)

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

print("Start Training")
trainer.fit(model=model, train_dataloaders=DataLoader(train_set, num_workers=config_default.num_workers,batch_size=config_default.batch_size,pin_memory=True,persistent_workers=True),
            val_dataloaders=DataLoader(valid_set, num_workers=config_default.num_workers,batch_size=config_default.batch_size,pin_memory=True,persistent_workers=True))

trainer.test(model=model, dataloaders=DataLoader(test_set, num_workers=config_default.num_workers,batch_size=config_default.batch_size,pin_memory=True,persistent_workers=True))

