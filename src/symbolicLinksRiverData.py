import os
from glob import glob

gitRepoPath = "/silor/boergel/paper/runoff_prediction/"
datapath = "/silod5/boergel/ocean_models_forcing/MOM_UERRA_FORCING/e-hype-runoff-raw"

data = glob(f"{datapath}/*")
os.makedirs(f"{gitRepoPath}/data/runoffData", exist_ok=True)
forcingPath = f"{gitRepoPath}/data/runoffData"

for singleData in data:
    os.system(f"ln -s {singleData} {forcingPath}/")