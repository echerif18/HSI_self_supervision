import sys
import os
import warnings
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


from src.utils_data import *
from src.transformation_utils import *
from src.utils_all import *
from src.Multi_trait.multi_model import *
from src.Multi_trait.trainer_multi import *

import pandas as pd
from datetime import datetime

import gc
import wandb 
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# from tqdm import tqdm
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import TensorDataset, Sampler

# from sklearn.utils import resample
# from collections import Counter

# Check if GPU is available
if torch.cuda.is_available():
    # Set the device to GPU
    device = torch.device("cuda")
    print("GPU is available. Using GPU for computation.")
else:
    # If GPU is not available, fall back to CPU
    device = torch.device("cpu")
    print("GPU is not available. Using CPU for computation.")


path_data_lb = os.path.join(project_root, "Datasets/50SHIFT_all_lb_prosailPro.csv") ##50SHIFT_all_lb_prosailPro 49_all_lb_prosailPro
path_save = '/home/mila/e/eya.cherif/scratch/multitrait'
project = 'multitrait_withScaler'

ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time in YYMMDD_HHMM format
formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")

seed = 155 #350 155 43   #random.randint(0, 500)
seed_all(seed=seed) ###155

batch_size = 256
num_epoch = 150
lr = 5*1e-4

# augmentation = True #
# trans = False
# percentage_tr = 1

for percentage_tr in [1]: # , 0.8, 0.6, 0.4, 0.2
    # Optional: Summarize GPU memory usage
    print(torch.cuda.memory_summary())

    run = 'multitrait_{}_{}labels_{}'.format(formatted_datetime, percentage_tr*100, seed)
    checkpoint_dir = os.path.join(path_save, "checkpoints_{}".format(run)) #'./checkpoints'
    print(run)

    ################ Data ###############
    db_lb_all = pd.read_csv(path_data_lb, low_memory=False).drop(['Unnamed: 0'], axis=1)   
    
    ### external
    groups = db_lb_all.groupby('dataset')
    
    val_ext_idx = list(groups.get_group(32).index)+list(groups.get_group(3).index)+list(groups.get_group(50).index)
    samples_val_ext = db_lb_all.loc[val_ext_idx,:]
    db_lb_all.drop(val_ext_idx, inplace=True)
    
    X_labeled, y_labeled, _ = data_prep_db(db_lb_all, ls_tr, weight_sample=True)
    metadata = db_lb_all.iloc[:, :8]  # The metadata (dataset of origin)
    
    
    red_ed = X_labeled.loc[:,750]
    red_end = X_labeled.loc[:,1300]
    red1000_ = X_labeled.loc[:,1000]
    
    idx = X_labeled[(red_end>red1000_) & (red_ed>red1000_)].index
    
    if(len(idx)>0):
        X_labeled.drop(idx, inplace=True)
        y_labeled.drop(idx, inplace=True)
        metadata.drop(idx, inplace=True)
    
    
    # Split labeled data into train (80%), validation (10%), and test (10%)
    fr_sup, X_val= train_test_split(X_labeled, test_size=0.2, stratify=metadata.dataset, random_state=300)
    
    y_sup = y_labeled.loc[fr_sup.index,:]
    y_val = y_labeled.loc[X_val.index,:]
    
    meta_train = metadata.loc[fr_sup.index,:]
    meta_val = metadata.loc[X_val.index,:]
    
    if(percentage_tr<1):
        fr_sup, _= train_test_split(fr_sup, test_size=1-percentage_tr, stratify=meta_train.dataset, random_state=300)
        
        y_sup = y_sup.loc[fr_sup.index,:]
        meta_train = meta_train.loc[fr_sup.index,:]
    
    
    db_tr = balanceData(pd.concat([fr_sup, y_sup], axis=1), meta_train, ls_tr, random_state=300,percentage=1)##.groupby('dataset').count().numSamples
    fr_sup = db_tr.loc[:,400:2450] ##### full range
    # fr_sup = db_tr.loc[:,400:900] ##### half range

    y_sup = db_tr.loc[:,'cab':]
    meta_train = db_tr.iloc[:,:8]
    
    
    # Create the dataset
    train_dataset = SpectraDataset(fr_sup, y_sup, meta_train, augmentation=True, aug_prob=0.7)
    # Define DataLoader with the custom collate function for fair upsampling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = SpectraDataset(X_train=X_val, y_train=y_val, meta_train=meta_val, augmentation=False)
    # Create DataLoader for the test dataset
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    ########## Scaler ###
    scaler_list = None
    scaler_model = save_scaler(y_sup, standardize=True, scale=True)
    
    
    # Example usage:
    settings_dict = {
        'epochs': num_epoch,
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'checkpoint_dir': checkpoint_dir,
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': 1e-4,
        'early_stop': True,
        'patience': 10,
        'scaler_model': scaler_model,
        # 'logger':wandb
    }
    
    sets = Settings()
    # sets.pretrained_model = test_mae.model
    sets.update_from_dict(settings_dict)

    wandb.init(
        # Set the project where this run will be logged
        project=project,
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{run}",
        # Track hyperparameters and run metadata
        config=settings_dict,
        dir = checkpoint_dir
        )
    
    test_reg = Trainer_MultiTraits(sets)
    test_reg.settings.logger = wandb

    test_reg.dataset_setup()
    test_reg.model_setup()
    test_reg.prepare_optimizers()
    test_reg.gpu_mode()
    test_reg.early_stopping_setup()
    test_reg.train_loop(epoch_start=1, num_epochs=test_reg.settings.epochs)

    # Clean up after training
    del test_reg
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Completed training model {percentage_tr}. GPU memory cleared.\n")