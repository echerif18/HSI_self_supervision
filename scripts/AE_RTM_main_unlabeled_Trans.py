import sys
import os
import warnings
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

os.getcwd()

from src.utils_data import *
from src.transformation_utils import *
from src.utils_all import *

#### Model definition ###
from src.rtm_torch.Resources.PROSAIL.call_model import *
from src.rtm_torch.rtm import RTM
from src.AE_RTM.AE_RTM_architectures import *
from src.AE_RTM.trainer_ae_rtm import *


import glob
import gc

import pandas as pd
import wandb 
import torch
from torch.utils.data import DataLoader

# import math
# from scipy.signal import savgol_filter
# import numpy as np
# from sklearn.preprocessing import PowerTransformer
# import random
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
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

from datetime import datetime

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time in YYMMDD_HHMM format
formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")

seed = 155 # 240 155 318 ##random.randint(0, 500)
seed_all(seed=seed) ###155

path_save = '/home/mila/e/eya.cherif/scratch/ae_rtm'
project = 'rtm_py_withScaler'

lr = 1e-3
n_epochs = 500 
ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
batch_size = 128  # This should match the batch size for unlabeled data

percentage_tr = 1

######## Data ########
directory_path = os.path.join(project_root, "Splits")
file_paths = glob.glob(os.path.join(directory_path, "*.csv"))
file_paths = file_paths[:int(percentage_tr*len(file_paths))]

################ Lbeled ###############
path_data_lb = os.path.join(project_root, "Datasets/50SHIFT_all_lb_prosailPro.csv") ##50SHIFT_all_lb_prosailPro 49_all_lb_prosailPro


# for i in range(5):
for gp in [2]: #2, 32, 50, 47, 6, 38
    # gp = random.randint(1, 50)
    run = 'AE_RTM_{}_gp{}UNlabels_{}'.format(formatted_datetime, gp, seed)
    checkpoint_dir = os.path.join(path_save, "checkpoints_{}".format(run)) #'./checkpoints'
    
    # Optional: Summarize GPU memory usage
    print(torch.cuda.memory_summary())
    print(gp)

    ### external
    db_lb_all = pd.read_csv(path_data_lb, low_memory=False).drop(['Unnamed: 0'], axis=1)   
    groups = db_lb_all.groupby('dataset')
    
    val_ext_idx = groups.get_group(gp).index
    samples_val_ext = db_lb_all.loc[val_ext_idx,:]
    db_lb_all.drop(val_ext_idx, inplace=True)

    X_val, y_val, _ = data_prep_db(samples_val_ext, ls_tr, weight_sample=True)
    meta_val = samples_val_ext.iloc[:, :8]
    
    X_train, y_train, _ = data_prep_db(db_lb_all, ls_tr, weight_sample=True)
    meta_train = db_lb_all.iloc[:, :8]  # The metadata (dataset of origin)
    
    
    red_ed = X_train.loc[:,750]
    red_end = X_train.loc[:,1300]
    red1000_ = X_train.loc[:,1000]
    
    idx = X_train[(red_end>red1000_) & (red_ed>red1000_)].index
    
    if(len(idx)>0):
        X_train.drop(idx, inplace=True)
        y_train.drop(idx, inplace=True)
        meta_train.drop(idx, inplace=True)
    
    
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
        'input_shape' : 1721, #500, #1721,
        'type':'full',
        'log_epoch' : 10,
        'lamb': 1e0,
        'loss_recons_criterion': CosineSimilarityLoss(), #CosineSimilarityLoss() #mse_loss
    }
    
    ### with wandb #####
    wandb.init(
        # Set the project where this run will be logged
        project=project,
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{run}",
        # Track hyperparameters and run metadata
        config=settings_dict,      # Log the configuration parameters.
        dir= checkpoint_dir
        )
    
    
    # Log all these settings to wandb
    # wandb.config.update(settings_dict)
    #>>> Model input 1721 !!!
    
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
    
    test_reg.settings.logger.finish()
    
    ##############
    preds = torch.empty(0,8).to(device)
    ori = torch.empty(0,8).to(device)
    test_reg.model.eval()
    
    with torch.no_grad():
        for batch_idx, val_sample in enumerate(test_reg.valid_loader):
            data_val, lb_bx_val, _ = val_sample
            
            lb_bx_val = lb_bx_val.to(device).float()
            data_val = data_val.squeeze().float().to(device)
            
            x_val, out_val = test_reg.model(data_val)
            output_val = out_val.data[:,list(range(951))+list(range(1031,1401))+list(range(1651,2051))]
    
            val_pred = out_val.data[:,list(range(951))+list(range(1031,1401))+list(range(1651,2051))]
            
            ori = torch.cat((ori.data,lb_bx_val.data), dim=0)
            preds = torch.cat((preds.data,x_val[:,:].data), dim=0)
    
    
    if(test_reg.settings.scaler_model is not None):
        preds = test_reg.transformation_layer_inv(preds) ### shoud keep the sam eorder of labels !!!
        
    ori_lb = pd.DataFrame(ori.cpu(), columns=ls_tr[:])
    df_tr_val = pd.DataFrame(preds.cpu(), columns=ls_tr[:])
    df_tr_val['cbc'] = df_tr_val['cm']-df_tr_val['cp']

    ori_lb.to_csv(os.path.join(checkpoint_dir, "Obs_CV{}".format(gp)))
    df_tr_val.to_csv(os.path.join(checkpoint_dir, "Preds_CV{}".format(gp)))
    
    val_mertics = eval_metrics(ori_lb, df_tr_val)
    val_mertics.to_csv(os.path.join(checkpoint_dir, "ValidationMetrics_CV{}".format(gp)))
    
    # Clean up after training
    del test_reg
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Completed training model {percentage_tr}. GPU memory cleared.\n")