import os
from glob import glob

gitRepoPath = "/silor/boergel/paper/runoff_prediction/"
datapath = "/silod5/boergel/ocean_models_forcing/MOM_UERRA_FORCING/"

data = glob(f"{datapath}/????")
os.makedirs(f"{gitRepoPath}/data/atmosphericForcing", exist_ok=True)
forcingPath = f"{gitRepoPath}/data/atmosphericForcing"

for singleData in data:
    os.system(f"ln -s {singleData} {forcingPath}/")