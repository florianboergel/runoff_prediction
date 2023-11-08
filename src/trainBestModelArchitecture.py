import numpy as np
import argparse
import sys

import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from BalticRiverPrediction.convLSTM import ConvLSTM
from BalticRiverPrediction.BaltNet import BaltNet, LightningModel, AtmosphereDataModule

if __name__ == "__main__":

    # Set seed for reproducible
    L.seed_everything(123)

    # Use available tensor cores
    torch.set_float32_matmul_precision("high")

    # X 
    datapath="/silor/boergel/paper/runoff_prediction/data"
    datapathPP="/fast/boergel/paper/runoff_prediction"

    runoff = xr.open_dataset(f"{datapathPP}/runoff.nc").load()
    runoff = runoff.roflux

    DataRain = xr.open_dataset(f"{datapathPP}/rain2.nc")
    DataRain = DataRain.sel(time=slice("1979", "2011"))
    DataRain = DataRain.RAIN.squeeze()

    DataShumi = xr.open_dataset(f"{datapathPP}/shumi.nc")
    DataShumi = DataShumi.sel(time=slice("1979", "2011"))
    DataShumi = DataShumi.shumi.squeeze()

    DataWindMagnitude = xr.open_dataset(f"{datapathPP}/windxy.nc")
    DataWindMagnitude['wind_magnitude'] = (DataWindMagnitude['windx']**2 + DataWindMagnitude['windy']**2)**0.5
    DataWindMagnitude = DataWindMagnitude.sel(time=slice("1979", "2011"))
    DataWindMagnitude = DataWindMagnitude.wind_magnitude.squeeze(dim="height", drop=True)
    
    assert DataShumi.time[0] == DataRain.time[0] == DataWindMagnitude.time[0]
    assert len(DataShumi.time) == len(DataRain.time) == len(DataWindMagnitude.time)

    data = xr.merge([DataRain, DataShumi, DataWindMagnitude])
    assert len(runoff.time) == len(data.time)

    modelParameters = {
        "input_dim": 3,
        "hidden_dim": 6,
        "kernel_size": (7,7),
        "num_layers": 2,
        "batch_first": True,
        "bias": True,
        "return_all_layers": False,
        "dimensions": (191, 206),
        "input_size": 30
    }

    # Loads the atmospheric data in batches
    dataLoader = AtmosphereDataModule(
    atmosphericData=data,
    runoff=runoff,
    batch_size=64,
    input_size=modelParameters["input_size"],
    num_workers=8
    )

    num_epochs = 2000

    pyTorchBaltNet = BaltNet(modelPar=modelParameters)

    LightningBaltNet = LightningModel(
        model=pyTorchBaltNet,
        learning_rate=1e-3,
        cosine_t_max=num_epochs
    )

    callbacks = [
        ModelCheckpoint(
            dirpath="/silor/boergel/paper/runoff_prediction/data/modelWeights/",
            filename=f"ProductionModelTopOne",
            save_top_k=1,
            mode="min",
            monitor="val_mse",
            save_last=True,
            )
        ]

    logger = TensorBoardLogger(
        save_dir="/silor/boergel/paper/runoff_prediction/logs",
        name=f"ProductionModel"
        )   

    trainer = L.Trainer(
        # precision="bf16-mixed",
        callbacks=callbacks,
        max_epochs=num_epochs,
        accelerator="cuda",
        devices=2,
        logger=logger
        )

    trainer.fit(model=LightningBaltNet, datamodule=dataLoader)