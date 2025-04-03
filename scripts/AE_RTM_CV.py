from src.utils_data import *
from src.transformation_utils import *
from src.utils_all import *

#### Model definition ###
from src.rtm_torch.Resources.PROSAIL.call_model import *
from src.rtm_torch.rtm import RTM
from src.AE_RTM.AE_RTM_architectures import *
from src.AE_RTM.trainer_ae_rtm import *


import os
from scipy.signal import savgol_filter
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer

import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import wandb 
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Sampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import warnings
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision


# Check if GPU is available
if torch.cuda.is_available():
    # Set the device to GPU
    device = torch.device("cuda")
    print("GPU is available. Using GPU for computation.")
else:
    # If GPU is not available, fall back to CPU
    device = torch.device("cpu")
    print("GPU is not available. Using CPU for computation.")

from datetime import datetime

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time in YYMMDD_HHMM format
formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")

seed_all(seed=155)
path_save = '/home/mila/e/eya.cherif/scratch/ae_rtm'
project = 'rtm_py_withScaler'

lr = 1e-5 #4
n_epochs = 200
ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
batch_size = 128  # This should match the batch size for unlabeled data

path_data = '/home/mila/e/eya.cherif/Gan_project_test/Unlabeled_clean_veg.csv'


################ Data ###############
db_un = pd.read_csv(path_data, low_memory=False).drop(['Unnamed: 0'], axis=1).reset_index(drop=True)
# metadata_un = db_un.iloc[:, :8]

unlabeled_data = db_un.loc[:, '400':'2450']
unlabeled_data.columns = unlabeled_data.columns.astype('int')
unlabeled_data.loc[:,ls_tr] = float('nan')

## new ###
path_data = '/home/mila/e/eya.cherif/Gan_project_test/EnmapSampling_merge22_meta_veg_clean.csv'

db_un = pd.read_csv(path_data, low_memory=False).drop(['Unnamed: 0'], axis=1).reset_index(drop=True).loc[:, '400':'2450']
db_un.columns = db_un.columns.astype('int')

samples_un = pd.concat([unlabeled_data, db_un]).reset_index(drop=True)
fr_unsup = samples_un.iloc[:,:-8] 


path_data_lb = '/home/mila/e/eya.cherif/Gan_project_test/49_all_lb_prosailPro.csv'
db_lb_all = pd.read_csv(path_data_lb, low_memory=False).drop(['Unnamed: 0'], axis=1)   

### external
groups = db_lb_all.groupby('dataset')

val_ext_idx = list(groups.get_group(32).index)+list(groups.get_group(3).index)
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


X_labeled.reset_index(drop=True, inplace=True)
y_labeled.reset_index(drop=True, inplace=True)
metadata.reset_index(drop=True, inplace=True)

# Create the StratifiedKFold object with n_splits (number of folds you want)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits)


# Loop through the StratifiedKFold splits
for fold_index, (train_index, test_index) in enumerate(skf.split(X_labeled, metadata.dataset)):
    print(f"Fold {fold_index + 1}")
    
    # You can access train and test data using these indices
    X_train, X_val = X_labeled.loc[train_index, :], X_labeled.loc[test_index,:]
    y_train, y_val = y_labeled.loc[train_index, :], y_labeled.loc[test_index,:]
    
    meta_val = metadata.loc[test_index, :]
    meta_train = metadata.loc[train_index, :]
    
    run = 'AE_RTM_{}_CV{}labels'.format(formatted_datetime, fold_index + 1)
    checkpoint_dir = os.path.join(path_save, "checkpoints_{}".format(run)) #'./checkpoints'
    
    
    ########## Scaler ###
    # scaler_list = save_scaler(y_train, standardize=True, scale=True)
    scaler_list = save_scaler(y_train, standardize=True, scale=True, save=True, dir_n=checkpoint_dir, k='all_cv{}'.format(fold_index + 1))
    
    # Create the dataset
    train_dataset = SpectraDataset(X_train, y_train, meta_train, augmentation=True, aug_prob=0.8)
    # Define DataLoader with the custom collate function for fair upsampling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    
    test_dataset = SpectraDataset(X_train=X_val, y_train=y_val, meta_train=meta_val, augmentation=False)
    # Create DataLoader for the test dataset
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create the dataset
    untrain_dataset = SpectraDataset(fr_unsup, augmentation=True, aug_prob=0.5)
    # Define DataLoader with the custom collate function for fair upsampling
    unlabeled_loader = DataLoader(untrain_dataset, batch_size=batch_size, shuffle=True)
    
    
    
    ### with wandb #####
    wandb.init(
        # Set the project where this run will be logged
        project=project,
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{run}",
        # Track hyperparameters and run metadata
        # config=settings_to_dict(test.settings)
        )
    
    
    ######
    # Example usage:
    settings_dict = {
        'epochs': n_epochs,
        'train_loader': train_loader,
        'unlabeled_loader' : unlabeled_loader,
        'valid_loader': valid_loader,
        'checkpoint_dir': checkpoint_dir,
        'batch_size': batch_size,
        'learning_rate': lr,
        'early_stop': True,
        'patience': 10,
        'scaler_model': scaler_list,
        'input_shape' : 1721,
        'log_epoch' : 10,
        'lamb': 1e0,
        'loss_recons_criterion': CosineSimilarityLoss(), #CosineSimilarityLoss() #mse_loss
        'logger': wandb
    }

    # Log all these settings to wandb
    # wandb.config.update(settings_dict)
    
    sets = Settings_ae()
    sets.update_from_dict(settings_dict)
    
    test_reg = Trainer_AE_RTM(sets)
    
    test_reg.dataset_setup()
    test_reg.model_setup()
    test_reg.prepare_optimizers()
    test_reg.gpu_mode()
    test_reg.early_stopping_setup()
    test_reg.train_loop(epoch_start=1, num_epochs=sets.epochs) #epoch_start=1, num_epochs=500