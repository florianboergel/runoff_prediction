
import torch
import torch.nn as nn
import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split

from BalticRiverPrediction.BaltNet import BaltNet
from BalticRiverPrediction.BaltNet import LightningModel
from BalticRiverPrediction.BaltNet import AtmosphereDataModule

if __name__ == "__main__":

    # Our GPU has tensor cores, hence mixed precision training is enabled
    # see https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html
    # for more

    torch.set_float32_matmul_precision("medium")
    
    # set random seed for reproducibility
    L.pytorch.seed_everything(123)


    # Loads the atmospheric data in batches
    dataLoader = AtmosphereDataModule(
    datapath="/silor/boergel/paper/runoff_prediction/data",
    batch_size=64
    )

    # Note that this set of parameters will be defined by runTuning.py

    modelParameters = {
    "input_dim":30, # timesteps
    "hidden_dim":1, # Channels -> right now only precipitation
    "kernel_size":(4,4), # applied for spatial convolutions
    "num_layers":2, # number of convLSTM layers
    "batch_first":True, # first index is batch
    "bias":True, 
    "return_all_layers": False, 
    "dimensions": (191, 206) # dimensions of atmospheric forcing
    }

    ### Setup model

    num_epochs = 50

    # initalize model
    pyTorchBaltNet = BaltNet(modelPar=modelParameters)

    # Lightning model wrapper
    LighningBaltNet = LightningModel(
        pyTorchBaltNet,
        learning_rate=1e-3,
        cosine_t_max=num_epochs
    )

    # save best model 
    callbacks = [
        ModelCheckpoint(
            dirpath="/silor/boergel/paper/runoff_prediction/data/modelWeights/",
            filename="BaltNetTopOne",
            save_top_k=1,
            mode="max",
            monitor="val_mse",
            save_last=True
        )
    ]

    trainer = L.Trainer(
        callbacks=callbacks,
        max_epochs=num_epochs,
        accelerator="cuda",
        devices=2,
        logger=CSVLogger(
            save_dir="/silor/boergel/paper/runoff_prediction/logs",
            name="BaltNet1"
        ),
        deterministic=True,
    )

    trainer.fit(model=LighningBaltNet, datamodule=dataLoader)
    # trainer.test(model=LighningBaltNet, datamodule=dataLoader)