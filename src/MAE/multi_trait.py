import time
import sys
import os

######## the file looks at the parent directory ###
# Set the parent directory in the system path for relative imports.
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

# Import utility modules from the project.
from transformation_utils import *
from utils_all import *
from utils_data import *

from MAE.utils_mae import *
from MAE.trainer_mae import *
from MAE.MAE_1D import *
from MAE.trainer_trait import *

import torch
import torch.nn as nn
import torchvision.models as models

from sklearn.model_selection import train_test_split
import gc
import numpy as np
import pandas as pd

#########################################
# Helper Function: half_latent
#########################################
def half_latent(x, model):
    """
    Extract latent features using a half masking strategy.

    Steps:
        1. Embed patches using the model's sequence embedding.
        2. Add positional embeddings (excluding the cls token if present).
        3. Apply half masking to the embedded patches.
        4. Pass through the Transformer blocks.
        5. Normalize the resulting features.
    
    Args:
        x (torch.Tensor): Input spectral data.
        model (nn.Module): MAE model containing the embedding, masking, and Transformer blocks.
    
    Returns:
        torch.Tensor: The latent representation.
    """
    # Embed patches from the input
    x = model.seq_embed(x)
    
    # Add positional embeddings without the cls token
    x = x + model.pos_embed[:, np.sum(model.is_cls_token):, :]
    
    # Apply half masking to the embedded patches
    x, mask, ids_restore = model.half_masking(x)
    
    # Pass through the Transformer blocks sequentially
    for blk in model.blocks:
        x = blk(x)
    # Normalize the output latent features
    z = model.norm(x)
    return z

#########################################
# LatentRegressionModel Definition
#########################################
class LatentRegressionModel(nn.Module):
    """
    Regression model that leverages a pretrained MAE encoder to extract latent features,
    then applies an aggregation strategy and a regression head to predict target values.

    Args:
        pretrained_encoder (nn.Module): Pretrained MAE encoder.
        latent_dim (int): Dimension of the latent vector.
        output_dim (int): Dimension of the output (e.g., number of regression targets).
        input_dim (int, optional): Dimension of the input spectra. Default is 1720.
        type_sp (str, optional): Strategy type ('full' for full extraction or otherwise for half masking). Default is 'full'.
        hidden_dims (list of int, optional): Sizes of hidden layers for dense regression head. If None, use a single layer.
        aggregation (str, optional): Aggregation strategy ('none', 'mean', or 'custom').
        normalize_latent (bool, optional): Whether to apply LayerNorm to the latent space.
        freeze_encoder (bool, optional): Whether to freeze the encoder during training.
    """
    def __init__(self, pretrained_encoder, latent_dim, output_dim, input_dim=1720, type_sp='full',
                 hidden_dims=None, aggregation="none", normalize_latent=False, freeze_encoder=True):
        super().__init__()
        self.encoder = pretrained_encoder
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.type_sp = type_sp
        self.aggregation = aggregation
        self.normalize_latent = normalize_latent

        # Optional normalization layer for latent features.
        if self.normalize_latent:
            self.normalization = nn.LayerNorm(self.latent_dim)
        else:
            self.normalization = None

        # Calculate the number of sequences (patches) from the input.
        seq_size = (self.input_dim // self.encoder.seq_embed.seq_size)
        
        # Determine the input dimension for the regression head based on the aggregation method.
        input_dim = {
            "none": seq_size * latent_dim,
            "mean": latent_dim,
            "custom": 3 * latent_dim
        }.get(aggregation, seq_size * latent_dim)

        # Build the regression head with optional hidden layers.
        self.regression_head = self._build_regression_head(input_dim, hidden_dims, output_dim)

    def _build_regression_head(self, input_dim, hidden_dims, output_dim):
        """
        Constructs a regression head composed of dense layers and activation functions.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dims (list of int or None): List of hidden layer sizes.
            output_dim (int): Dimension of the output layer.
        
        Returns:
            nn.Sequential: A sequential model representing the regression head.
        """
        layers = []
        if hidden_dims is not None:
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(hidden_dim))
                input_dim = hidden_dim
        # Final output layer
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass to extract latent representations and perform regression.

        Depending on the type_sp parameter, either the full encoder or half masking strategy is applied.
        Then, an aggregation strategy is applied to the latent features before passing them
        through the regression head.

        Args:
            x (torch.Tensor): Input spectral data.
        
        Returns:
            torch.Tensor: Regression predictions.
        """
        # Extract latent representations using full or half masking strategy.
        if self.type_sp == 'full':
            latent, _, _ = self.encoder.forward_encoder(x, mask_ratio=0)  # No masking during regression
        else:
            latent = half_latent(x, self.encoder)
        
        # Normalize the latent features if enabled.
        if self.normalize_latent:
            latent = self.normalization(latent)

        # Apply aggregation strategy:
        if self.aggregation == "none":  # Use full sequence flattening
            latent = latent.flatten(start_dim=1)
        elif self.aggregation == "mean":  # Mean pooling across the sequence dimension
            latent = latent.mean(dim=1)
        elif self.aggregation == "custom":  # Custom aggregation (e.g., VIS/NIR/SWIR concatenation)
            # Example for custom aggregation with three segments:
            vis = torch.mean(latent[:, :10, :], dim=1)
            nir = torch.mean(latent[:, 10:48, :], dim=1)
            swir = torch.mean(latent[:, 48:, :], dim=1)
            
            latent = torch.cat([vis, nir, swir], dim=1)
            latent = latent.flatten(start_dim=1)

        # Pass the aggregated latent features through the regression head.
        return self.regression_head(latent)




#########################################
# Downstream Regression Pipeline
#########################################


def validation(valid_loader, model, scaler_model, scaler_list , test_tr, sp_type = 'full', n_bands=500, ext=False):
    outputs_val = []
    lb_val = []
    
    if (ext):
        # Evaluate the model on the external set.
        for labeled_examples, labels in tqdm(valid_loader):
            model.eval()
            with torch.no_grad():
                # Preprocess inputs for validation.
                if(sp_type != 'full'):
                    labeled_examples = labeled_examples.to(device)[:,:n_bands] ####half_range
                else: 
                    labeled_examples = labeled_examples.to(device)[:, :-1]
                labels = labels.to(device)
                output_val = model(labeled_examples.squeeze().float())
                
                # Optionally invert scaling if a scaler is available.
                if scaler_list is not None:
                    outputs_val.append(scaler_list.inverse_transform(output_val.cpu().numpy()))
                    lb_val.append(scaler_list.inverse_transform(labels.cpu().numpy()))
                
                if scaler_model is not None:
                    output_val = scaler_model(output_val)
                 
                outputs_val.append(output_val.cpu().numpy())
                lb_val.append(labels.cpu().numpy())
        
        # Combine predictions and ground truth into DataFrames.
        pred = pd.DataFrame(np.concatenate(outputs_val, axis=0), columns=test_tr)
        obs_pf = pd.DataFrame(np.concatenate(lb_val, axis=0), columns=test_tr)
        
        # Evaluate metrics and return selected metric values.
        val_mertics = eval_metrics(obs_pf, pred)
    else:
        # Evaluate the model on the validation set.
        for labeled_examples, labels, ds in tqdm(valid_loader):
            model.eval()
            with torch.no_grad():
                # Preprocess inputs for validation.
                if(sp_type != 'full'):
                    labeled_examples = labeled_examples.to(device)[:,:n_bands] ####half_range
                else: 
                    labeled_examples = labeled_examples.to(device)[:, :-1]
                labels = labels.to(device)
                output_val = model(labeled_examples.squeeze().float())
                
                # Optionally invert scaling if a scaler is available.
                if scaler_list is not None:
                    outputs_val.append(scaler_list.inverse_transform(output_val.cpu().numpy()))
                    lb_val.append(scaler_list.inverse_transform(labels.cpu().numpy()))
                
                if scaler_model is not None:  
                    output_val = scaler_model(output_val)
                 
                outputs_val.append(output_val.cpu().numpy())
                lb_val.append(labels.cpu().numpy())
        
        # Combine predictions and ground truth into DataFrames.
        pred = pd.DataFrame(np.concatenate(outputs_val, axis=0), columns=test_tr)
        obs_pf = pd.DataFrame(np.concatenate(lb_val, axis=0), columns=test_tr)
        
        # Evaluate metrics and return selected metric values.
        val_mertics = eval_metrics(obs_pf, pred)

    return val_mertics


def downstream(pretrained_model, FT = False, num_epochs=200):
    """
    Sets up and runs the downstream trait regression task using a pretrained MAE model.

    This function:
      1. Loads and preprocesses labeled data.
      2. Splits data into training and validation sets.
      3. Applies filtering and scaling to the labels.
      4. Constructs datasets and dataloaders.
      5. Configures training settings and instantiates the regression trainer.
      6. Trains the regression model.
      7. Evaluates the model on validation data and returns evaluation metrics.

    Args:
        pretrained_model (nn.Module): Pretrained MAE model.
    
    Returns:
        tuple: Evaluation metrics (arrays of selected metric values).
    """
    # Define file path and target trait names
    path_data_lb = '/home/mila/e/eya.cherif/Gan_project_test/49_all_lb_prosailPro.csv'
    path_data_lb_ext = '/home/mila/e/eya.cherif/Gan_project_test/DB50_all_F_SHIFT.csv'
    ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
    
    percentage_tr = 1  # Fraction of training data to use
    batch_size = 256
    
    ########## Trait regression: Data Preparation ###############
    # Load labeled data and drop unnecessary columns.
    db_lb_all = pd.read_csv(path_data_lb, low_memory=False).drop(['Unnamed: 0'], axis=1)   
    
    # Separate external validation samples based on group 'dataset'
    groups = db_lb_all.groupby('dataset')
    val_ext_idx = list(groups.get_group(32).index) + list(groups.get_group(3).index)
    samples_val_ext = db_lb_all.loc[val_ext_idx, :]
    db_lb_all.drop(val_ext_idx, inplace=True)

    ##### External samples OOD ###
    db_lb_all_ext = pd.read_csv(path_data_lb_ext, low_memory=False).drop(['Unnamed: 0'], axis=1)   
    ext_all = pd.concat([samples_val_ext,db_lb_all_ext])
    ext_val_x = feature_preparation(ext_all.loc[:, '400':'2500']).loc[:, 400:2450] #ext_all samples_val_ext
    ext_val_y = ext_all[ls_tr]
    
    # Validation dataset
    x_p_val = torch.tensor(ext_val_x.values, dtype=torch.float)#.unsqueeze(dim=1) ##ext_val_x.values scaler_spectra.transform(ext_val_x.iloc[:,:-1])
    lb_p_val = torch.tensor(ext_val_y.values,dtype=torch.float)
    
    # Prepare features and labels
    X_labeled, y_labeled, _ = data_prep_db(db_lb_all, ls_tr, weight_sample=True)
    metadata = db_lb_all.iloc[:, :8]  # Metadata (e.g., dataset of origin)
    
    # Filter out problematic samples based on spectral indices
    red_ed = X_labeled.loc[:, 750]
    red_end = X_labeled.loc[:, 1300]
    red1000_ = X_labeled.loc[:, 1000]
    idx = X_labeled[(red_end > red1000_) & (red_ed > red1000_)].index
    if len(idx) > 0:
        X_labeled.drop(idx, inplace=True)
        y_labeled.drop(idx, inplace=True)
        metadata.drop(idx, inplace=True)
    
    # Split labeled data into training (80%) and validation (20%) sets.
    fr_sup, val_x = train_test_split(X_labeled, test_size=0.2, stratify=metadata.dataset, random_state=300)
    y_sup = y_labeled.loc[fr_sup.index, :]
    meta_train = metadata.loc[fr_sup.index, :]
    val_y = y_labeled.loc[val_x.index, :]
    meta_val = metadata.loc[val_x.index, :]
    
    if percentage_tr < 1:
        fr_sup, _ = train_test_split(fr_sup, test_size=1 - percentage_tr, stratify=meta_train.dataset, random_state=300)
        y_sup = y_sup.loc[fr_sup.index, :]
        meta_train = meta_train.loc[fr_sup.index, :]
    
    # ########## Scaler Setup ###
    scaler_list = None
    scaler_model = save_scaler(y_sup, standardize=True, scale=True)
    
    ######### Create Datasets and DataLoaders ############
    train_dataset = SpectraDataset(fr_sup, y_sup, meta_train, augmentation=True, aug_prob=0.6)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = SpectraDataset(X_train=val_x, y_train=val_y, meta_train=meta_val, augmentation=False)
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    external_dataset = TensorDataset(x_p_val, lb_p_val) # x_p_val
    valid_ext_loader = DataLoader(external_dataset, batch_size=batch_size, shuffle=False)
    
    # Prepare training settings in a dictionary.
    settings_dict = {
        'train_loader': train_dataset_loader,
        'valid_loader': valid_loader,
        'batch_size': batch_size,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'pretrained_model': pretrained_model,
        'early_stop': False,
        'patience': 10,
        'scaler_model': scaler_model,
    }
    
    # Create settings object and update with the dictionary.
    sets = Settings_reg()
    sets.update_from_dict(settings_dict)
    
    # Instantiate the trainer for MAE regression.
    test_reg = Trainer_MAE_Reg(sets)
    test_reg.dataset_setup()
    
    ############ Model Modification ############
    # Extract latent dimension from a specific block of the pretrained model.
    latent_dim = pretrained_model.blocks[3].mlp.fc2.out_features
    
    # Instantiate the LatentRegressionModel.
    test_reg.model = LatentRegressionModel(
        pretrained_encoder=pretrained_model,
        latent_dim=latent_dim,
        output_dim=len(ls_tr),
        hidden_dims=None,  # No additional hidden layers
        aggregation="mean",
        normalize_latent=True,  # Enable latent normalization
    )
    
    ############ Training Setup and Execution ############
    # Clean up memory before training.
    gc.collect()
    torch.cuda.empty_cache()
    
    test_reg.transformation_setup()
    test_reg.criterion_setup()
    test_reg.prepare_optimizers()
    test_reg.gpu_mode()
    # Optionally set up early stopping if desired.
    # test_reg.early_stopping_setup()
    
    # Freeze the encoder parameters to prevent training.
    for param in test_reg.model.encoder.parameters():
        param.requires_grad = FT #False
    
    # Train the regression model for a specified number of epochs.
    test_reg.train_loop(epoch_start=1, num_epochs=num_epochs) #200
    
    ############ Validation #############
    test_tr = ls_tr
    # outputs_val = []
    # lb_val = []
    
    # # Evaluate the model on the validation set.
    # for labeled_examples, labels, ds in tqdm(valid_loader):
    #     test_reg.model.eval()
    #     with torch.no_grad():
    #         # Preprocess inputs for validation.
    #         labeled_examples = labeled_examples.to(device)[:, :-1]
    #         labels = labels.to(device)
    #         output_val = test_reg.model(labeled_examples.squeeze().float())
            
    #         # Optionally invert scaling if a scaler is available.
    #         if scaler_list is not None:
    #             outputs_val.append(scaler_list.inverse_transform(output_val.cpu().numpy()))
    #             lb_val.append(scaler_list.inverse_transform(labels.cpu().numpy()))
            
    #         if test_reg.transformation_layer_inv is not None:  
    #             output_val = test_reg.transformation_layer_inv(output_val)
             
    #         outputs_val.append(output_val.cpu().numpy())
    #         lb_val.append(labels.cpu().numpy())
    
    # # Combine predictions and ground truth into DataFrames.
    # pred = pd.DataFrame(np.concatenate(outputs_val, axis=0), columns=test_tr)
    # obs_pf = pd.DataFrame(np.concatenate(lb_val, axis=0), columns=test_tr)
    
    # # Evaluate metrics and return selected metric values.
    # val_mertics = eval_metrics(obs_pf, pred)
    val_mertics = validation(valid_loader, test_reg.model, test_reg.transformation_layer_inv, scaler_list, test_tr) ##, sp_type = 'full', n_bands=500
    ext_mertics = validation(valid_ext_loader, test_reg.model, test_reg.transformation_layer_inv, scaler_list, test_tr, ext=True) ##, sp_type = 'full', n_bands=500
    
    return val_mertics.values[:, 0], val_mertics.values[:, 2], ext_mertics.values[:, 0], ext_mertics.values[:, 2]
