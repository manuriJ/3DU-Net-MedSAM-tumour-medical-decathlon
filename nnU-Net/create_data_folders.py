import os
import shutil
from collections import OrderedDict

import json
import matplotlib.pyplot as plt
import nibabel as nib

import numpy as np
import torch

import nnunetv2

# !pip install wandb

# import wandb
# wandb.init(project="nnU-Net_Workshop")

# check whether GPU accelerated computing is available
print(torch.cuda.is_available())  # if there is an error here, enable GPU in the Runtime

data_dir = os.path.join("./data")


# 5. Setting up nnU-Nets folder structure and environment variables
# nnUnet expects a certain folder structure and environment variables.

def make_if_dont_exist(folder_path, overwrite=False):
    ''' creates a folder if it does not exists input:
    folder_path : relative path of the folder which needs to be created
    over_write :(default: False) if True overwrite the existing folder
    '''
    if os.path.exists(folder_path):

        if not overwrite:
            print(f"{folder_path} exists.")
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)

    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")


## 5.1 Set environment Variables and creating folders"""

# Maybe move path of preprocessed data directly on content - this may be signifcantely faster!
print("Current Working Directory {}".format(os.getcwd()))
path_dict = {
    "nnUNet_raw": os.path.join(data_dir, "nnUNet_raw"),
    "nnUNet_preprocessed": os.path.join(data_dir, "nnUNet_preprocessed"),  # 1 experiment: 1 epoch took 112s
    "nnUNet_results": os.path.join(data_dir, "nnUNet_results"),
    "RAW_DATA_PATH": os.path.join(data_dir, "RawData"),
    # This is used here only for convenience (not necessary for nnU-Net)!
}

# Write paths to environment variables
for env_var, path in path_dict.items():
    os.environ[env_var] = path

# Check whether all environment variables are set correct!
for env_var, path in path_dict.items():
    if os.getenv(env_var) != path:
        print("Error:")
        print("Environment Variable {} is not set correctly!".format(env_var))
        print("Should be {}".format(path))
        print("Variable is {}".format(os.getenv(env_var)))
    make_if_dont_exist(path, overwrite=False)

print("If No Error Occured Continue Forward. =)")

# 6. Using nnU-Net on Medical Decathlon datasets

# print(os.path.exists(
#     os.path.join(path_dict["RAW_DATA_PATH"], "Task01_BrainTumour")))  # check whether the file is correctly downloaded
