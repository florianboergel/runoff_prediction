import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchmetrics import Metric
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from BalticRiverPrediction.BaltNet import AtmosphereDataModule
from BalticRiverPrediction.BaltNet import BaltNet, LightningModel
from BalticRiverPrediction.sharedUtilities import PredictionPlottingCallback
import torch
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description="Parse model parameters")

    parser.add_argument("--modelName", type=str, default="BaltNet")
    parser.add_argument("--hidden_dim", type=int, default=6, help="Hidden states")
    parser.add_argument("--kernel_size", type=str, default="(5,5)", help="Kernel size for spatial convolutions (format: x,y)")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of convLSTM layers")
    parser.add_argument("--batch_first", type=bool, default=True, help="If first index is batch")
    parser.add_argument("--bias", type=bool, default=True, help="Bias parameter")
    parser.add_argument("--return_all_layers", type=bool, default=False, help="Return all layers")
    parser.add_argument("--dimensions", type=str, default="(222, 244)", help="Dimensions (format: x,y)")
    parser.add_argument("--input_size", type=int, default=30, help="Input size")
    parser.add_argument("--num_epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")

    args = parser.parse_args()

    # Convert string tuples to actual tuples
    args.kernel_size = tuple(map(int, args.kernel_size.strip('()').split(',')))
    args.dimensions = tuple(map(int, args.dimensions.strip('()').split(',')))

    return args

def pretty_format_args(args):
    formatted_args = "\n".join([f"{arg}: {getattr(args, arg)}" for arg in vars(args)])
    return formatted_args

if __name__ == "__main__":

    # Set seed for reproducible
    L.seed_everything(123)

    # Use available tensor cores
    torch.set_float32_matmul_precision("medium")

    datapath="/silor/boergel/paper/runoff_prediction/data"
    datapathPP="/silod9/boergel/runoff_prediction_ERA5_downscaled_coupled_model/resampled"

    runoff = xr.open_dataset(f"{datapath}/runoff.nc").load()
    runoff = runoff.sel(time=slice("1979", "2011"))
    runoff = runoff.roflux

    DataRain = xr.open_dataset(f"{datapathPP}/rain.nc")
    DataRain = DataRain.sel(time=slice("1979", "2011"))
    DataRain = DataRain.rain.squeeze()
    DataRain = DataRain.drop(["lon","lat"])
    DataRain = DataRain.rename({"rlon":"x","rlat":"y"})

    DataShumi = xr.open_dataset(f"{datapathPP}/QV.nc")
    DataShumi = DataShumi.sel(time=slice("1979", "2011"))
    DataShumi = DataShumi.QV.squeeze()
    DataShumi = DataShumi.drop(["lon","lat"])
    DataShumi = DataShumi.rename({"rlon":"x","rlat":"y"})

    DataWindSpeed = xr.open_dataset(f"{datapathPP}/speed.nc")
    DataWindSpeed = DataWindSpeed.sel(time=slice("1979", "2011"))
    DataWindSpeed = DataWindSpeed.speed.squeeze()
    DataWindSpeed = DataWindSpeed.drop(["lon","lat"])
    DataWindSpeed = DataWindSpeed.rename({"rlon":"x","rlat":"y"})

    DataTemp = xr.open_dataset(f"{datapathPP}/T.nc")
    DataTemp = DataTemp.sel(time=slice("1979", "2011"))
    DataTemp = DataTemp.T.squeeze()
    DataTemp = DataTemp.drop(["lon","lat"])
    DataTemp = DataTemp.rename({"rlon":"x","rlat":"y"})

    assert DataShumi.time[0] == DataRain.time[0] == DataWindSpeed.time[0]
    assert len(DataShumi.time) == len(DataRain.time) == len(DataWindSpeed.time)

    data = xr.merge([DataRain, DataShumi, DataWindSpeed, DataTemp])
    assert len(runoff.time) == len(data.time)

    args = parse_args()

    modelParameters = {
        "input_dim": 4,
        "hidden_dim": args.hidden_dim,
        "kernel_size": args.kernel_size,
        "num_layers": args.num_layers,
        "batch_first": args.batch_first,
        "bias": args.bias,
        "return_all_layers": args.return_all_layers,
        "dimensions": (222, 244),
        "input_size": args.input_size
    }

    print(f"""
    ==================================
    Model Configuration:
    ==================================
        {pretty_format_args(args)}
    ==================================
    """)

    # Loads the atmospheric data in batches
    dataLoader = AtmosphereDataModule(
        atmosphericData=data,
        runoff=runoff,
        batch_size=50,
        input_size=modelParameters["input_size"],
        num_workers=16
    )

    num_epochs = args.num_epochs

    pyTorchBaltNet = BaltNet(modelPar=modelParameters)

    if args.checkpoint: 
        print("Loading from checkpoint")
        LightningBaltNet = LightningModel.load_from_checkpoint(
            checkpoint_path=f"{args.checkpoint}",
            learning_rate=1e-6,
            map_location="cpu",
            model=pyTorchBaltNet,
            cosine_t_max=num_epochs//50
        )
    else:
        LightningBaltNet = LightningModel(
            model=pyTorchBaltNet,
            learning_rate=1e-5,
            cosine_t_max=num_epochs,
            alpha=3
        )

    callbacks = [
        ModelCheckpoint(
            dirpath="/silor/boergel/paper/runoff_prediction/data/modelWeights/",
            filename=f"{args.modelName}TopOne",
            save_top_k=1,
            mode="min",
            monitor="val_mse",
            save_last=True,
            ),
        PredictionPlottingCallback()
        ]

    logger = TensorBoardLogger(
        save_dir="/silor/boergel/paper/runoff_prediction/logs",
        name=f"{args.modelName}"
        )   

    trainer = L.Trainer(
        callbacks=callbacks,
        max_epochs=num_epochs,
        accelerator="cuda",
        devices=2,
        logger=logger
        )

    trainer.fit(model=LightningBaltNet, datamodule=dataLoader)
