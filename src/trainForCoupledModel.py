import numpy as np
import argparse
import sys

import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchmetrics import Metric
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from BalticRiverPrediction.BaltNet import AtmosphericDataset
from BalticRiverPrediction.BaltNet import BaltNet

import torch
import torch.nn as nn

class EnhancedMSELoss(nn.Module):
    def __init__(self, alpha=1.5):
        """
        Initialize the enhanced MSE loss module.

        Args:
            alpha (float): Exponential factor to increase penalty for larger errors.
        """
        super(EnhancedMSELoss, self).__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        """
        Calculate the enhanced MSE loss.

        Args:
            predictions (torch.Tensor): The predicted values.
            targets (torch.Tensor): The ground truth values.

        Returns:
            torch.Tensor: The calculated loss.
        """
        error = predictions - targets
        mse_loss = torch.mean(error**2)
        enhanced_error = torch.mean(torch.abs(error) ** self.alpha)
        enhanced_mse_loss = mse_loss + enhanced_error
        return enhanced_mse_loss
    
class EnhancedMSEMetric(Metric):
    def __init__(self, alpha=1.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.alpha = alpha
        self.add_state("sum_enhanced_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        error = predictions - targets
        mse_loss = torch.mean(error ** 2)
        enhanced_error = torch.mean(torch.abs(error) ** self.alpha)

        self.sum_enhanced_error += (mse_loss + enhanced_error) * targets.numel()
        self.total += targets.numel()

    def compute(self):
        return self.sum_enhanced_error / self.total



class AtmosphereDataModule(L.LightningDataModule):
    
    def __init__(self, atmosphericData, runoff, batch_size=64, num_workers=8, add_first_dim=True, input_size=30):
        super().__init__()

        self.data = atmosphericData
        self.runoff = runoff
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.add_first_dim = add_first_dim
        self.input_size = input_size
    
    def setup(self, stage:str):
        UserWarning("Loading atmospheric data ...")
        dataset = AtmosphericDataset(
            atmosphericData=self.data,
            runoff=self.runoff,
            input_size=self.input_size
            )
        n_samples = len(dataset)
        train_size = int(0.9 * n_samples)
        val_size = n_samples - train_size
        # self.train, self.val = train_test_split(dataset)
        self.train, self.val, = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True, 
            drop_last=True, 
            num_workers=self.num_workers,
            pin_memory=False  # Speed up data transfer to GPU
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=False  # Speed up data transfer to GPU
        )

class LightningModel(L.LightningModule):
    """
    A PyTorch Lightning model for training and evaluation.
    
    Attributes:
        model (nn.Module): The neural network model.
        learning_rate (float): Learning rate for the optimizer.
        cosine_t_max (int): Maximum number of iterations for the cosine annealing scheduler.
        train_mse (torchmetrics.MeanSquaredError): Metric for training mean squared error.
        val_mse (torchmetrics.MeanSquaredError): Metric for validation mean squared error.
        test_mse (torchmetrics.MeanSquaredError): Metric for testing mean squared error.
    """
    
    def __init__(self, model, learning_rate, cosine_t_max):
        """
        Initializes the LightningModel.

        Args:
            model (nn.Module): The neural network model.
            learning_rate (float): Learning rate for the optimizer.
            cosine_t_max (int): Maximum number of iterations for the cosine annealing scheduler.
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.cosine_t_max = cosine_t_max
        self.loss_function = EnhancedMSELoss(alpha=2)

        # Save hyperparameters except the model
        self.save_hyperparameters(ignore=["model"])

        # Define metrics
        self.train_mse = EnhancedMSEMetric(alpha=2)
        self.val_mse = EnhancedMSEMetric(alpha=2)
        self.test_mse = EnhancedMSEMetric(alpha=2)

    def forward(self, x):
        """Defines the forward pass of the model."""
        return self.model(x)
    
    def _shared_step(self, batch, debug=False):
        """
        Shared step for training, validation, and testing.

        Args:
            batch (tuple): Input batch of data.
            debug (bool, optional): If True, prints the loss. Defaults to False.

        Returns:
            tuple: Computed loss, true labels, and predicted labels.
        """
        features, true_labels = batch
        logits = self.model(features)
        loss = self.loss_function(logits, true_labels)
        if debug:
            print(loss)
        return loss, true_labels, logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, true_labels, predicted_labels = self._shared_step(batch)
        mse = self.train_mse(predicted_labels, true_labels)
        metrics = {"train_mse": mse, "train_loss": loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, true_labels, predicted_labels = self._shared_step(batch)
        mse = self.val_mse(predicted_labels, true_labels)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_mse", mse, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, _):
        """Test step."""
        loss, true_labels, predicted_labels = self._shared_step(batch)
        mse = self.test_mse(predicted_labels, true_labels)
        self.log("test_loss", loss, rank_zero_only=True)
        self.log("test_mse", mse, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """Prediction step."""
        _, _, predicted_labels = self._shared_step(batch)
        return predicted_labels

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            tuple: List of optimizers and list of learning rate schedulers.
        """
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, verbose=False)
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val_mse"}

def parse_args():
    parser = argparse.ArgumentParser(description="Parse model parameters")

    parser.add_argument("--modelName", type=str, default="BaltNet")
    parser.add_argument("--hidden_dim", type=int, default=6, help="Hidden states")
    parser.add_argument("--kernel_size", type=str, default="(5,5)", help="Kernel size for spatial convolutions (format: x,y)")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of convLSTM layers")
    parser.add_argument("--batch_first", type=bool, default=True, help="If first index is batch")
    parser.add_argument("--bias", type=bool, default=True, help="Bias parameter")
    parser.add_argument("--return_all_layers", type=bool, default=False, help="Return all layers")
    parser.add_argument("--dimensions", type=str, default="(191, 206)", help="Dimensions (format: x,y)")
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
    torch.set_float32_matmul_precision("high")

    datapath="/silor/boergel/paper/runoff_prediction/data"
    datapathPP="/silod9/boergel/runoff_prediction_ERA5_downscaled_coupled_model/resampled"

    runoff = xr.open_dataset(f"{datapath}/runoff.nc").load()
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

    assert DataShumi.time[0] == DataRain.time[0] == DataWindSpeed.time[0]
    assert len(DataShumi.time) == len(DataRain.time) == len(DataWindSpeed.time)

    data = xr.merge([DataRain, DataShumi, DataWindSpeed])
    assert len(runoff.time) == len(data.time)

    args = parse_args()

    modelParameters = {
        "input_dim": 3,
        "hidden_dim": args.hidden_dim,
        "kernel_size": args.kernel_size,
        "num_layers": args.num_layers,
        "batch_first": args.batch_first,
        "bias": args.bias,
        "return_all_layers": args.return_all_layers,
        "dimensions": (222,244),
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
    batch_size=32,
    input_size=modelParameters["input_size"],
    num_workers=8
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
            learning_rate=1e-3,
            cosine_t_max=num_epochs
        )

    callbacks = [
        ModelCheckpoint(
            dirpath="/silor/boergel/paper/runoff_prediction/data/modelWeights/",
            filename=f"{args.modelName}TopOne",
            save_top_k=1,
            mode="min",
            monitor="val_mse",
            save_last=True,
            )
        ]

    logger = TensorBoardLogger(
        save_dir="/silor/boergel/paper/runoff_prediction/logs",
        name=f"{args.modelName}"
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
