import sys
import os
import warnings
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils_data import *
from src.transformation_utils import *
from src.utils_all import *

#### Model definition ###
from src.rtm_torch.Resources.PROSAIL.call_model import *
# from src.rtm_torch.rtm import RTM
from src.AE_RTM.AE_RTM_architectures import *
from src.AE_RTM.trainer_ae_rtm import *

from datetime import datetime

import glob
import gc
import wandb 
import torch

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


# from scipy.signal import savgol_filter
# import numpy as np
# from sklearn.preprocessing import PowerTransformer
# import random
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader, TensorDataset, Sampler
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim import lr_scheduler


# Check if GPU is available
if torch.cuda.is_available():
    # Set the device to GPU
    device = torch.device("cuda")
    print("GPU is available. Using GPU for computation.")
else:
    # If GPU is not available, fall back to CPU
    device = torch.device("cpu")
    print("GPU is not available. Using CPU for computation.")


# Get the current date and time
current_datetime = datetime.now()

# Format the date and time in YYMMDD_HHMM format
formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")

seed = 240 ## 240 155 318  random.randint(0, 500)
seed_all(seed=seed) ###155

path_save = '/home/mila/e/eya.cherif/scratch/ae_rtm'
project = 'rtm_py_withScaler'

lr = 1e-3 
n_epochs = 500 #300
ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
batch_size = 128  # This should match the batch size for unlabeled data
# percentage_tr = 1

directory_path = os.path.join(project_root, "Splits")
file_paths = glob.glob(os.path.join(directory_path, "*.csv"))

path_data_lb = os.path.join(project_root, "Datasets/50SHIFT_all_lb_prosailPro.csv") ##50SHIFT_all_lb_prosailPro 49_all_lb_prosailPro
print(path_data_lb)

for percentage_tr in [1, 0.8, 0.6, 0.4, 0.2]: 
    # Optional: Summarize GPU memory usage
    print(torch.cuda.memory_summary())

    run = 'AE_RTM_{}_{}labels_{}'.format(formatted_datetime, percentage_tr*100, seed)
    checkpoint_dir = os.path.join(path_save, "checkpoints_{}".format(run)) #'./checkpoints'

    # file_paths = file_paths[:int(percentage_tr*len(file_paths))]
      
    ################ Lbeled ###############
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
        # X_labeled.loc[idx,:].T.plot(legend=False)
        X_labeled.drop(idx, inplace=True)
        y_labeled.drop(idx, inplace=True)
        metadata.drop(idx, inplace=True)
    
    
    # Split labeled data into train (80%), validation (10%), and test (10%)
    X_train, X_val= train_test_split(X_labeled, test_size=0.2, stratify=metadata.dataset, random_state=300)
    
    y_train = y_labeled.loc[X_train.index,:]
    y_val = y_labeled.loc[X_val.index,:]
    
    meta_train = metadata.loc[X_train.index,:]
    meta_val = metadata.loc[X_val.index,:]
    
    if(percentage_tr<1):
        X_train, _= train_test_split(X_train, test_size=1-percentage_tr, stratify=meta_train.dataset, random_state=300)
        
        y_train = y_train.loc[X_train.index,:]
        meta_train = meta_train.loc[X_train.index,:]
    
    # db_tr = balanceData(pd.concat([X_train, y_train], axis=1), meta_train, ls_tr, random_state=300,percentage=1)
    # X_train = db_tr.loc[:,400:2450]
    # y_train = db_tr.loc[:,'cab':]
    # meta_train = db_tr.iloc[:,:8]
    
    # fr_unsup = samples_un.iloc[:,:-8] 
    
    
    ########## Scaler ###
    scaler_list = save_scaler(y_train, standardize=True, scale=True, save=True, dir_n=checkpoint_dir, k='all_{}'.format(100*percentage_tr))
    
    # Create the dataset
    train_dataset = SpectraDataset(X_train, y_train, meta_train, augmentation=True, aug_prob=0.8)
    # Define DataLoader with the custom collate function for fair upsampling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    
    test_dataset = SpectraDataset(X_train=X_val, y_train=y_val, meta_train=meta_val, augmentation=False)
    # Create DataLoader for the test dataset
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    # # Create the dataset
    untrain_dataset = MultiFileAugmentedCSVDataset(file_paths, chunk_size=1000, augmentation=True, aug_prob=0.5, scale=False) ## No scaling of spectra
    unlabeled_loader = DataLoader(untrain_dataset, batch_size=batch_size, 
                            shuffle=True
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
        'weight_decay' : 1e-5,
        'early_stop': True,
        'patience': 10,
        'scaler_model': scaler_list,
        'input_shape' : 1721,
        'log_epoch' : 10,
        'lamb': 1e0,
        'loss_recons_criterion': CosineSimilarityLoss(), #CosineSimilarityLoss() #mse_loss
        # 'logger': wandb
    }

    ### with wandb #####
    wandb.init(
        # Set the project where this run will be logged
        project=project,
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{run}",
        # Track hyperparameters and run metadata
        config=settings_dict
        )

    sets = Settings_ae()
    sets.update_from_dict(settings_dict)
    
    test_reg = Trainer_AE_RTM(sets)
    test_reg.settings.logger = wandb  # Attach wandb logger to the trainer.
    
    test_reg.dataset_setup()
    test_reg.model_setup()
    test_reg.prepare_optimizers(test_reg.settings.epochs)
    test_reg.gpu_mode()
    test_reg.early_stopping_setup()
    test_reg.train_loop(epoch_start=1, num_epochs=sets.epochs)

    # Clean up after training
    del test_reg
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Completed training model {percentage_tr}. GPU memory cleared.\n")