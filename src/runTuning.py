import torch
import torch.nn as nn
import pytorch_lightning as pl

from BalticRiverPrediction.BaltNet import BaltNet
from BalticRiverPrediction.BaltNet import LightningModel
from BalticRiverPrediction.BaltNet import AtmosphereDataModule
from BalticRiverPrediction.sharedUtilities import read_netcdfs, preprocess

import ray
from ray import tune
import ray.train.lightning
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import FIFOScheduler


def train_func(config):
    torch.set_float32_matmul_precision("medium")
  
    # Loads the atmospheric data in batches
    dataLoader = AtmosphereDataModule(
    data=data,
    runoff=runoff,
    batch_size=16,
    input_size=config["input_dim"]
    )

    num_epochs = 50

    # initalize model
    pyTorchBaltNet = BaltNet(modelPar=config)

    # Lightning model wrapper
    LighningBaltNet = LightningModel(
        pyTorchBaltNet,
        learning_rate=1e-3,
        cosine_t_max=num_epochs
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy=ray.train.lightning.RayDDPStrategy(),
        callbacks=[ray.train.lightning.RayTrainReportCallback()],
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        max_epochs=num_epochs
    )

    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model=LighningBaltNet, datamodule=dataLoader)

def tune_BaltNet_FIFO(ray_trainer, config, num_samples=1):
    scheduler = FIFOScheduler()

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": config},
        tune_config=tune.TuneConfig(
            metric="val_mse",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=2
        ),
    )
    return tuner.fit()

if __name__ == "__main__":

    torch.set_float32_matmul_precision("medium")

    datapath="/silor/boergel/paper/runoff_prediction/data"
    
    data = read_netcdfs(
        files=f"{datapath}/atmosphericForcing/????/rain.mom.dta.nc",
        dim="time",
        transform_func=lambda ds:preprocess(ds)
        )       

    runoff = read_netcdfs(
        f"{datapath}/runoffData/combined_fastriver_*.nc",
        dim="river",
        transform_func= lambda ds:ds.sel(time=slice(str(1979), str(2011))).roflux.resample(time="1D").mean(),
        cftime=False
        )  
    
    #TODO: May be not needed to call ray.init()
    ray.init(num_gpus=2)

    modelParameters = {
    "input_dim": tune.choice([16,32]), # timesteps
    "hidden_dim":1, # Channels -> right now only precipitation
    "kernel_size": tune.choice([(5,5),(7,7)]), # applied for spatial convolutions, only uneven numbers
    "num_layers": tune.choice([3,4,7]), # number of convLSTM layers
    "batch_first":True, # first index is batch
    "bias":True, 
    "return_all_layers": False, 
    "dimensions": (191, 206) # dimensions of atmospheric forcing
    }

    # Scaling config provides the resource allocated for the trainer
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker={"GPU":1, "CPU":16}
        )

    # Runtime configuration for training and tuning runs
    run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="val_mse",
        checkpoint_score_order="min",
        ),
        local_dir="/silor/boergel/paper/runoff_prediction/data/",
        storage_path="/silor/boergel/paper/runoff_prediction/data/modelWeights/",
        name="TestHyperParameterSpace"
    )

    # Trainer configuration
    ray_trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config,
    )

    results = tune_BaltNet_FIFO(ray_trainer, modelParameters, num_samples=8)