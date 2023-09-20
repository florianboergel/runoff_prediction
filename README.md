# Runoff prediction


## Maksfile

The file `mask.nc` contains masks for each catchment area belonging to a particular basin.
Each basin is identified with a mask values.
This file can be used as a basis to train a neural network to map atmospheric variables onto a rivers that supply the individual basins.


## River locations

The file `rivers.json` contains a list of river mouths and their locations in longitudes and latitudes.


## Grid file

The file `t_grid.nc` specifies the grid on which the runoff model will work.
This model is curently named `ROFF`.
The file is in the `SCRIP` format and contains the region that is masked in `mask.nc`.
However, the grid is transformed to a grid with rank one, i.e. all grid points are aligned in a one-dimensional array.


## River locations on the model grids

The file `model_points.json` contains a list of all rivers in `rivers.json` with their location mapped on the involved model grid points, i.e. nearest neighboring points of the true locations.