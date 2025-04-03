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
import gc
from datetime import datetime

import torch
import torch.nn as nn
import wandb 

# import numpy as np
# import random
# import pandas as pd
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader, TensorDataset, Sampler
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split

seed = 155 #random.randint(0, 500)
# seed_all(seed=seed) ###155

device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time in YYMMDD_HHMM format
formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")

### inputs: 
# percentage_tr = 1

batch_size = 256
augmentation = True
scale = False
n_epochs = 500
lr = 5e-4 
weight_decay = 1e-4
mask_ratio = 0.75

### fix:
path_save = '/home/mila/e/eya.cherif/scratch/mae/'
project = 'MAE_wandb_test'

ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]

directory_path = os.path.join(project_root, "Splits")

# percentage_tr = 1
for percentage_tr in [1]: #1,0.8, 0.6, 0.4, 0.2
    # Optional: Summarize GPU memory usage
    print(torch.cuda.memory_summary())

    file_paths = glob.glob(os.path.join(directory_path, "*.csv"))
    file_paths = file_paths[:int(percentage_tr*len(file_paths))]

    run_mae = 'MAE_{}_Training{}HR_allUNlabels_{}'.format(formatted_datetime, 100*percentage_tr, seed)
    path_model_mae = os.path.join(path_save, "{}.h5".format(run_mae))
    checkpoint_dir_mae = os.path.join(path_save, "checkpoints_{}".format(run_mae)) #'./checkpoints'
    
    #####
    settings_dict_mae = {
        'seed': seed,
        'epochs': n_epochs,
        'batch_size': batch_size,
        'augmentation': augmentation,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'load_model_path': None, #load_model_path
        'file_paths': file_paths,
        
        'w_loss': 1,
        'mask_ratio': mask_ratio,
        'n_bands': 500, ##500, 1720
        'type':'half',
        'seq_size': 20,
        'in_chans': 1,
        'embed_dim': 128,
        'depth': 10, #6,
        'num_heads': 16, #4,
        'decoder_embed_dim': 128,
        'decoder_depth': 6, #4
        'decoder_num_heads': 4,
        'mlp_ratio': 4.0,
        'norm_layer': nn.LayerNorm,
        'cls_token': False,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
        'checkpoint_dir': checkpoint_dir_mae,
        'early_stop': True,
        'patience': 10,
    }
    
    
    sets = Settings()
    sets.update_from_dict(settings_dict_mae)
    
    wandb.init(
    # Set the project where this run will be logged
    project=project,
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_{run_mae}",
    # Track hyperparameters and run metadata
    config=settings_dict_mae,
    dir = checkpoint_dir_mae
    )
    
    test = Trainer(sets)
    test.settings.logger = wandb
    test.train()

    wandb.finish()

    # Clean up after training
    del test
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Completed training model {percentage_tr}. GPU memory cleared.\n")