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


class BaltNet(nn.Module):
    def __init__(self, modelPar):
        super(BaltNet, self).__init__()

        # Initialize all attributes
        for k, v in modelPar.items():
            setattr(self, k, v)

        self.encoder = ConvLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            bias=self.bias,
            return_all_layers=False
        )

        self.decoder = ConvLSTM(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            num_layers=1,
            batch_first=self.batch_first,
            bias=self.bias,
            return_all_layers=False
        )

        self.linear_dim = self.dimensions[0] * self.dimensions[1] * self.hidden_dim 

        # Single fully connected network for all rivers
         
        self.river_predictors = nn.Sequential(
            nn.Linear(self.linear_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 97)
        )

        # Creating separate attention weights for each river
        self.attention_weights = nn.Parameter(torch.randn(self.hidden_dim, 1, 1), requires_grad=True)  # 97 rivers

    def spatial_attention(self, x):
        """Spatial attention mechanism."""
        B, T, C, H, W = x.size()

        x = x.view(B * T, C, H, W)
        
        # Apply attention weights for all rivers
        self.attention_map = torch.sigmoid(F.conv2d(x, self.attention_weights.unsqueeze(0), bias=None, stride=1, padding=0))
        
        # Weighted sum
        output = x * self.attention_map  # B*T, C, H, W
        output = output.view(B, T, C, H, W)  # B, T, C

        return output

    def forward(self, x):
        B, _, _, _, _ = x.size()

        # Pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(x)

        # Use the entire encoder output as input to the decoder
        decoder_input = encoder_outputs[0][:,-1,:,:,:].unsqueeze(1)

        # Pass through decoder using the final hidden state of the encoder
        decoder_outputs, _ = self.decoder(decoder_input, encoder_hidden)

        # Apply spatial attention
        decoder_with_spatial_attention = self.spatial_attention(decoder_outputs[0])  # B, T, C, H, W
            
        # Flatten the temporal sequence
        decoder_with_spatial_attention_flattened = decoder_with_spatial_attention.view(B, -1)  #
            
        # Pass through its own predictor
        output = self.river_predictors(decoder_with_spatial_attention_flattened)  # B, -1

        return output

#| export

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

        # Save hyperparameters except the model
        self.save_hyperparameters(ignore=["model"])

        # Define metrics
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

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
        loss = F.mse_loss(logits, true_labels)
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

    
class AtmosphericDataset(Dataset):
    def __init__(self, input_size, atmosphericData, runoff, transform=None):

        # Length of the sequence
        self.input_size = input_size

        # input data (x) 
        atmosphericDataStd = atmosphericData.std("time") # dimension will be channel, lat, lon
        atmosphericDataMean = atmosphericData.mean("time")
        self.atmosphericStats = (atmosphericDataMean, atmosphericDataStd)

        # output data - label (y)
        runoffData = runoff.transpose("time", "river")
        runoffDataMean = runoffData.mean("time")
        runoffDataSTD = runoffData.std("time")
        self.runoffDataStats = (runoffDataMean, runoffDataSTD)
        
        # normalize data
        X = ((atmosphericData - atmosphericDataMean)/atmosphericDataStd).compute()
        y = ((runoffData - runoffDataMean)/runoffDataSTD).compute()
        
        # an additional dimension for the channel is added
        # to end up with (time, channel, lat, lon)
        xStacked = X.to_array(dim='variable')
        xStacked = xStacked.transpose("time", "variable", "y", "x")

        assert xStacked.data.ndim == 4
        self.x = torch.tensor(xStacked.data, dtype=torch.float32)
        self.y = torch.tensor(y.data, dtype=torch.float32)

    def __getitem__(self, index):
        return self.x[index:index+(self.input_size)], self.y[index+int(self.input_size)]

    def __len__(self):
        return self.y.shape[0]-(self.input_size)
    
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
        train_size = int(0.85 * n_samples)
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

def save_attention_maps(model, input_data, filename='attention_maps.nc'):
    model.eval()
    with torch.no_grad():
        # Forward pass
        model(input_data)

        # Extract attention maps
        attention_maps_np = model.attention_map.cpu().detach().numpy()

        # Convert to xarray DataArray
        attention_da = xr.DataArray(attention_maps_np, dims=('batch', 'channel', 'height', 'width'))

        # Save to netCDF
        attention_da.to_netcdf(filename)


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

    args = parse_args()

    modelParameters = {
        "input_dim": 3,
        "hidden_dim": args.hidden_dim,
        "kernel_size": args.kernel_size,
        "num_layers": args.num_layers,
        "batch_first": args.batch_first,
        "bias": args.bias,
        "return_all_layers": args.return_all_layers,
        "dimensions": args.dimensions,
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
    batch_size=64,
    input_size=modelParameters["input_size"],
    num_workers=8
    )

    num_epochs = args.num_epochs

    pyTorchBaltNet = BaltNet(modelPar=modelParameters)

    if args.checkpoint: 
        LightningBaltNet = LightningModel.load_from_checkpoint(
            checkpoint_path=f"{args.checkpoint}",
            learning_rate=1e-6,
            map_location="cpu",
            model=pyTorchBaltNet
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
        devices=1,
        logger=logger
        )

    trainer.fit(model=LightningBaltNet, datamodule=dataLoader)

    dataLoader.setup(stage="")

    save_attention_maps(pyTorchBaltNet, dataLoader.train[0][0].unsqueeze(0), filename=f"{args.modelName}_attention_maps.nc")