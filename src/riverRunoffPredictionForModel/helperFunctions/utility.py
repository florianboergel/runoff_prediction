import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os

def get_grid(path):
    """
    Loads grid file    
    Arguments:
        path {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    from netCDF4 import Dataset
    dset = Dataset(path)
    lon_grid = dset['grid_x_T'][:]
    lat_grid = dset['grid_y_T'][:]
    return lon_grid, lat_grid

def get_lon_lat_rivers(path_to_rivers, path):
    """
    Reads all river files in folder. 
    Lon and lat coordinates are compared to matching grid points of MOM grid.
    Searching for time index of UERRA period (starts 1961)
    
    Arguments:
        path_to_rivers {str]} -- Folder where rivers are located
        path {str} -- Folder where grid_spec.nc is located
    
    Returns:
        df_rivers {dataFrame} -- DataFrame that contains all informations of all rivers
    """
    rivers_nc = os.listdir(path_to_rivers)
    lons = []
    lats = []
    river_name = []
    time = []
    time_index = []
    for river_nc in rivers_nc:
        dset = Dataset(path_to_rivers + river_nc)
        lon = dset['roflux'].lon
        lat = dset['roflux'].lat
        time.append(len(dset['time']))
        lons.append(lon)
        lats.append(lat)
        river_name.append(river_nc)
        time_index.append(select_year_1961(dset,timevar="time"))
    lon_grid, lat_grid = get_grid(path)
    matching_grid_lons = []
    matching_grid_lats = []
    for i in range(len(lons)):
        matching_grid_lons.append(int(find_nearest_grid_point(lon_grid, lons[i])))
        matching_grid_lats.append(int(find_nearest_grid_point(lat_grid, lats[i])))
    df_rivers = pd.DataFrame(np.column_stack((lons, lats, time, time_index, river_name,matching_grid_lons, matching_grid_lats)), columns = ["lon_river", "lat_river","time","time_index","river" ,"matching_index_lon", "matching_index_lat"])
    return df_rivers

def find_nearest_grid_point(grid, value):
    """
    Finds nearest grid point 
    
    Arguments:
        grid {[type]} -- [description]
        value {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    import numpy as np
    return (np.abs(grid-value)).argmin()