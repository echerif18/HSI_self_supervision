# Import all necessary modules and functions from local utilities and MAE packages.
import sys
import os
import warnings
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils_data import *
from src.transformation_utils import *
from src.utils_all import *

from src.MAE.utils_mae import *
from src.MAE.trainer_mae import *
from src.MAE.MAE_1D import *
from src.MAE.multi_trait import *
from src.MAE.trainer_trait import *

import glob
from datetime import datetime

import numpy as np
import pandas as pd
import itertools
import torch
import torch.nn as nn
import wandb 

# import random
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader, TensorDataset, Sampler
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split

# Set the device to GPU if available, otherwise CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time in YYMMDD_HHMM format (used to tag experiment runs)
formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")


#######################
# Global Configuration
#######################
# Base path for saving models and checkpoints.
path_save = '/home/mila/e/eya.cherif/scratch/mae/'
# Project name for wandb logging.
project = 'MAE_wandb_test'

# Experiment hyperparameters and settings.
seed = 155
batch_size = 128  # Adjusted batch size (previously 256)
augmentation = True
scale = False

n_epochs = 500
lr = 5e-4 
weight_decay = 1e-4
mask_ratio = 0.75

############# Data Configuration #############
# Path to the directory containing CSV data files.
directory_path = os.path.join(project_root, "Splits")
file_paths = glob.glob(os.path.join(directory_path, "*.csv"))

# Path to the labeled data file and list of target traits.
path_data_lb = os.path.join(project_root, "Datasets/50SHIFT_all_lb_prosailPro.csv") ##50SHIFT_all_lb_prosailPro 49_all_lb_prosailPro
ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
percentage_tr = 1

#######################
# Grid Search Setup
#######################
# # Define parameter grid for grid search over transformer depth and number of heads.
depth = 10
num_heads = 16
w_loss = 1

param_grid = list([10, 20, 40, 430]) ##10, 20, 40, 430

# Initialize containers to store results from grid search.
results = []
results_DS = []

# # Loop over each combination of depth and number of heads.
for seq_size in param_grid:
    # Create a unique run name using the formatted datetime, depth, num_heads, and seed.
    run_mae = 'MAE_{}_Training_SeqSize{}_allUNlabels_{}'.format(formatted_datetime, seq_size, seed)
    
    # Define a checkpoint directory for this experiment run.
    checkpoint_dir_mae = os.path.join(path_save, "checkpoints_{}".format(run_mae))
    
    # Prepare a dictionary with all the training settings for the MAE.
    # NOTE: Ensure commas are used between dictionary entries.
    settings_dict_mae = {
        'seed': seed,  # Seed for reproducibility.
        'epochs': n_epochs,
        'batch_size': batch_size,
        'augmentation': augmentation,
        'learning_rate': lr,
        'weight_decay': weight_decay,

        'file_paths': file_paths,
        'mask_ratio': mask_ratio,
        'w_loss': w_loss,
        
        'n_bands': 1720,
        'seq_size': seq_size,
        'in_chans': 1,
        'embed_dim': 128,
        'depth': depth,  # Using current depth from grid.
        'num_heads': num_heads,  # Using current number of heads.
        'decoder_embed_dim': 128,
        'decoder_depth': 6,  # Fixed decoder depth.
        'decoder_num_heads': 4,
        'mlp_ratio': 4.0,
        'norm_layer': nn.LayerNorm,
        'cls_token': False,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
        'checkpoint_dir': checkpoint_dir_mae,
        'early_stop': True,
        'patience': 10,
    }
    
    # Instantiate a settings object and update it with the above dictionary.
    sets = Settings()
    sets.update_from_dict(settings_dict_mae)
    
    # Initialize a new wandb run for logging experiment details.
    wandb.init(
        project=project,            # Project name in wandb.
        name=f"experiment_{run_mae}",  # Unique run name.
        config=settings_dict_mae      # Log the configuration parameters.
    )
    
    # Instantiate the MAE trainer with the current settings.
    test = Trainer(sets)
    test.settings.logger = wandb  # Attach wandb logger to the trainer.
    test.train()  # Start training the model.
    
    # Finish the wandb run after training is complete.
    wandb.finish()

    # Save the final validation loss for this run.
    final_val_loss = test.valid_loss_list
    results.append((seq_size, final_val_loss[-1]))
    
    # Run downstream evaluation on the trained model.
    val_r2, val_nrmse = downstream(test.model)
    # Save selected evaluation metrics using quantiles (70th percentile).
    results_DS.append((seq_size, np.mean(val_r2), np.mean(val_nrmse)))


# Compile the results from grid search into pandas DataFrames.
df_results = pd.DataFrame(results, columns=["seq_size", "Final_Val_Loss"])
df_DS_results = pd.DataFrame(results_DS, columns=["seq_size", "Final_Val_r2", "Final_Val_nrmse"])

# Save the results to CSV files in the checkpoint directory.
df_results.to_csv(os.path.join(checkpoint_dir_mae, 'experimentMAE_{}_{}_ValLossLast.csv'.format(formatted_datetime, seed)))
df_DS_results.to_csv(os.path.join(checkpoint_dir_mae, 'experimentMAE_{}_{}_ValDSLast.csv'.format(formatted_datetime, seed)))