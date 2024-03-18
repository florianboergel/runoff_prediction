import os

for year in range(1961, 2020):
    # datapath1 = f"/silod9/boergel/MOMconvRiverNew2/{year}/ocean_day3d.nc"
    datapath2 = f"/silod9/boergel/MOMoriginalBMIP/{year}/ocean_day3d.nc"

    # os.system(f"module load cdo && cdo vertmean -yearmean -selvar,salt {datapath1} /silod9/boergel/MOMconvRiverNew2/{year}/salt_fld_{year}.nc")
    os.system(f"module load cdo && cdo vertmean -yearmean -selvar,salt {datapath2} /silod9/boergel/MOMoriginalBMIP/{year}/salt_fld_{year}.nc")