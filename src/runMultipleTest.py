
import torch
import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from BalticRiverPrediction.BaltNet import BaltNet
from BalticRiverPrediction.BaltNet import LightningModel
from BalticRiverPrediction.BaltNet import AtmosphereDataModule
from BalticRiverPrediction.sharedUtilities import read_netcdfs, preprocess

if __name__ == "__main__":

    # set seed for reproducibility   
    L.seed_everything(123)

    # Our GPU has tensor cores, hence mixed precision training is enabled
    # see https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html

    torch.set_float32_matmul_precision("medium")

    datapath="/silor/boergel/paper/runoff_prediction/data"

    data = read_netcdfs(
        files=f"{datapath}/atmosphericForcing/????/rain.mom.dta.nc",
        dim="time",
        transform_func=lambda ds:preprocess(ds)
        ) 

    # data = read_netcdfs(
    #     files=f"{datapath}/atmosphericForcing/????/shumi.mom.dta.nc",
    #     dim="time",
    #     transform_func=lambda ds:preprocess(ds)
    #     )             

    runoff = read_netcdfs(
        f"{datapath}/runoffData/combined_fastriver_*.nc",
        dim="river",
        transform_func= lambda ds:ds.sel(time=slice(str(1979), str(2011))).roflux.resample(time="1D").mean(),
        cftime=False
        )   

    # Note that this set of parameters will be defined by runTuning.py

    modelParameters = {
    "input_dim": 1, # Number of channel, right now only precipitation
    "hidden_dim":30, # hidden states
    "kernel_size":(5,5), # applied for spatial convolutions
    "num_layers":3, # number of convLSTM layers
    "batch_first":True, # first index is batch
    "bias":True, 
    "return_all_layers": False, 
    "dimensions": (191, 206) # dimensions of atmospheric forcing
    }

    # Loads the atmospheric data in batches
    dataLoader = AtmosphereDataModule(
    atmosphericData=data,
    runoff=runoff,
    batch_size=32,
    input_size=30
    )

    ### Setup model

    num_epochs = 20

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
