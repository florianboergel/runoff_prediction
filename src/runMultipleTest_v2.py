import os
import torch
import torch.nn as nn
import lightning as L
import numpy as np
import xarray as xr
from glob import glob
from tqdm import tqdm

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
import torchmetrics

from BalticRiverPrediction.BaltNet import BaltNet
from BalticRiverPrediction.BaltNet import LightningModel
from BalticRiverPrediction.BaltNet import AtmosphereDataModule
from BalticRiverPrediction.sharedUtilities import read_netcdfs, preprocess
from BalticRiverPrediction.convLSTM import ConvLSTM

#| export
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

        # # save data
        # np.savetxt(
        #     "/silor/boergel/paper/runoff_prediction/data/modelStats.txt",
        #     [runoffDataMean, runoffDataSTD]
        # )
        
        # normalize data
        X = ((atmosphericData - atmosphericDataMean)/atmosphericDataStd).compute()
        y = ((runoffData - runoffDataMean)/runoffDataSTD).compute()
        
        # an additional dimension for the channel is added
        # to end up with (time, channel, lat, lon)
        xStacked = X.to_array(dim='variable')
        xStacked = xStacked.transpose("time", "variable", "lat", "lon")

        assert xStacked.data.ndim == 4
        self.x = torch.tensor(xStacked.data, dtype=torch.float16)
        self.y = torch.tensor(y.data, dtype=torch.float16)

    def __getitem__(self, index):
        return self.x[index:index+(self.input_size)], self.y[index+int(self.input_size)]

    def __len__(self):
        return self.y.shape[0]-(self.input_size)

#| export
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
        self.train, self.val, = random_split(dataset, [train_size, val_size])
        # val_size = int(0.1 * n_samples)
        # test_size = n_samples - train_size  - val_size
        # self.train, self.val, self.test = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True, 
            drop_last=True, 
            num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True)

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers, 
    #         drop_last=True)

#| export
class BaltNet(nn.Module):
    def __init__(self, modelPar):
        super(BaltNet, self).__init__()

        # initialize all attributes
        for k, v in modelPar.items():
            setattr(self, k, v)

        self.linear_dim = self.dimensions[0]*self.dimensions[1]*self.hidden_dim

        self.convLSTM = ConvLSTM(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                kernel_size=self.kernel_size,
                num_layers=self.num_layers,
                batch_first=self.batch_first,
                bias=self.bias,
                return_all_layers=self.return_all_layers
        )

        self.convLSTM2 = ConvLSTM(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                kernel_size=self.kernel_size,
                num_layers=1,
                batch_first=self.batch_first,
                bias=self.bias,
                return_all_layers=self.return_all_layers
        )

        # CNN layers to map the output of convLSTM2 to 97 rivers
        # self.cnn_layers = nn.Sequential(
        #     nn.Conv2d(self.hidden_dim, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
        #     nn.Flatten(),
        #     nn.Linear(32, 97)
        # )

        # CNN layers to map the output of convLSTM2 to 97 rivers
        # self.cnn_layers = nn.Sequential(
        #     nn.Conv2d(self.hidden_dim, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(256 * (self.dimensions[0] // 4) * (self.dimensions[1] // 4), 97)
        # )
        
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(self.linear_dim, 256),
            torch.nn.ReLU(),
            # torch.nn.Linear(512, 256),
            # torch.nn.ReLU(),
            torch.nn.Linear(256, 97)
            )

    def forward(self, x):
        _, encode_state = self.convLSTM(x)
        decoder_out, _ = self.convLSTM2(x[:,-1,:,:,:].unsqueeze(dim=1), encode_state)
        x = decoder_out[0].squeeze(1)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x).squeeze()
        return x

if __name__ == "__main__":

    L.seed_everything(123)

    torch.set_float32_matmul_precision("medium")

    datapath="/silor/boergel/paper/runoff_prediction/data"

    if os.path.isfile(f"{datapath}/rain.nc"):
        data = xr.open_dataset(f"{datapath}/rain.nc")
    else:
        data = read_netcdfs(
            files=f"{datapath}/atmosphericForcing/????/rain.mom.dta.nc",
            dim="time",
            transform_func=lambda ds:preprocess(ds)
            )         

        data = data.drop(["lon_bnds", "lat_bnds"])
        data = data.rename(
            {
                "x":"lon",
                "y":"lat"
            }
        )
        data.to_netcdf(f"{datapath}/rain.nc")
    if os.path.isfile(f"{datapath}/runoff.nc"):
        runoff = xr.open_dataset(f"{datapath}/runoff.nc")
    else:
        runoff = read_netcdfs(
            f"{datapath}/runoffData/combined_fastriver_*.nc",
            dim="river",
            transform_func= lambda ds:ds.sel(time=slice(str(1979), str(2011))).roflux.resample(time="1D").mean(),
            cftime=False
            )   
        runoff.to_netcdf(f"{datapath}/runoff.nc")

    modelParameters = {
    "input_dim": 1, # Number of channel, right now only precipitation
    "hidden_dim": 8, # hidden states
    "kernel_size":(5,5), # applied for spatial convolutions
    "num_layers": 3, # number of convLSTM layers
    "batch_first":True, # first index is batch
    "bias":True, 
    "return_all_layers": False, 
    "dimensions": (191, 206) # dimensions of atmospheric forcing
    }

        ### Setup model

    # Loads the atmospheric data in batches
    dataLoader = AtmosphereDataModule(
    atmosphericData=data,
    runoff=runoff,
    batch_size=64,
    input_size=30
    )

    num_epochs = 5

    # initalize model
    pyTorchBaltNet = BaltNet(modelPar=modelParameters)

    # Lightning model wrapper
    LighningBaltNet = LightningModel(
        pyTorchBaltNet,
        learning_rate=1e-3,
        cosine_t_max=40
    )

    # save best model 
    callbacks = [
        ModelCheckpoint(
            dirpath="/silor/boergel/paper/runoff_prediction/data/modelWeights/",
            filename="BaltNetTopOne",
            save_top_k=1,
            mode="min",
            monitor="val_mse",
            save_last=True
        )
    ]

    trainer = L.Trainer(
        precision=16,
        callbacks=callbacks,
        max_epochs=num_epochs,
        accelerator="cuda",
        devices=2,
        logger=CSVLogger(
            save_dir="/silor/boergel/paper/runoff_prediction/logs",
            name="BaltNet1"
        ),
    )

    trainer.fit(model=LighningBaltNet, datamodule=dataLoader)
