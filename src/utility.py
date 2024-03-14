import numpy as np
import xarray as xr
from torch.utils.data import DataLoader, Dataset

def loadData():
    datapath="/silor/boergel/paper/runoff_prediction/data"
    datapathPP="/silod9/boergel/runoff_prediction_ERA5_downscaled_coupled_model/resampled"

    runoff = xr.open_dataset(f"{datapath}/runoff.nc").load()
    runoff = runoff.sel(time=slice("1960", "2011"))
    runoff = runoff.resample(time="1D").mean()
    runoff = runoff.roflux

    DataRain = xr.open_dataset(f"{datapathPP}/rain.nc")
    DataRain = DataRain.sel(time=slice("1960", "2011"))
    DataRain = DataRain.rain.squeeze()
    DataRain = DataRain.drop(["lon","lat"])
    DataRain = DataRain.rename({"rlon":"x","rlat":"y"})

    DataShumi = xr.open_dataset(f"{datapathPP}/QV.nc")
    DataShumi = DataShumi.sel(time=slice("1960", "2011"))
    DataShumi = DataShumi.QV.squeeze()
    DataShumi = DataShumi.drop(["lon","lat"])
    DataShumi = DataShumi.rename({"rlon":"x","rlat":"y"})

    DataWindSpeed = xr.open_dataset(f"{datapathPP}/speed.nc")
    DataWindSpeed = DataWindSpeed.sel(time=slice("1960", "2011"))
    DataWindSpeed = DataWindSpeed.speed.squeeze()
    DataWindSpeed = DataWindSpeed.drop(["lon","lat"])
    DataWindSpeed = DataWindSpeed.rename({"rlon":"x","rlat":"y"})

    DataTemp = xr.open_dataset(f"{datapathPP}/T.nc")
    DataTemp = DataTemp.sel(time=slice("1960", "2011"))
    DataTemp = DataTemp.T.squeeze()
    DataTemp = DataTemp.drop(["lon","lat"])
    DataTemp = DataTemp.rename({"rlon":"x","rlat":"y"})

    # I think runoff can now be just a placeholder 

    assert DataShumi.time[0] == DataRain.time[0] == DataWindSpeed.time[0]
    assert len(DataShumi.time) == len(DataRain.time) == len(DataWindSpeed.time)

    data = xr.merge([DataRain, DataShumi, DataWindSpeed, DataTemp]).resample(time="1D").mean()

    assert len(runoff.time) == len(data.time)

    return data, runoff

class PredictionData(Dataset):
    def __init__(self, input_size, atmosphericData, runoff, atmosStats, runoffStats, transform=None):

        self.input_size = input_size
        runoffData = runoff.transpose("time", "river")
        
        X = ((atmosphericData - atmosStats[0])/atmosStats[1]).compute()
        y = ((runoffData - runoffStats[0])/runoffStats[1]).compute()
        
        xStacked = X.to_array(dim='variable')
        xStacked = xStacked.transpose("time", "variable", "y", "x")

        assert xStacked.data.ndim == 4
        self.x = torch.tensor(xStacked.data, dtype=torch.float32)
        self.y = torch.tensor(y.data, dtype=torch.float32)

    def __getitem__(self, index):
        return self.x[index:index+(self.input_size)], self.y[index+int(self.input_size)]

    def __len__(self):
        return self.y.shape[0]-(self.input_size)