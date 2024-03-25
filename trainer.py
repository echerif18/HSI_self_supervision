
"""
MAE traning
"""
import random

import numpy as np
from scipy.stats import norm
from torch.nn import Module
import torch.nn as nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

from tqdm import tqdm

from torch.utils.data.sampler import SubsetRandomSampler
from MAE_1D import *
import time

import warnings
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device = 'cuda'#'cuda:0' cpu

class Settings:
    """Represents the settings for a given run of SRGAN."""
    def __init__(self):
        # self.steps_to_run = 200000
        self.starting_step = 0
        self.epochs = 200
        self.temporary_directory = 'temporary'
        self.logs_directory = 'logs'
        self.batch_size = 256
        self.valid_size = 0.2
        self.augmentation = False
        self.learning_rate = 1e-4
        self.weight_decay = 0

        self.load_model_path = None
        self.should_save_models = True
        self.skip_completed_experiment = True
        self.number_of_data_workers = 4

        self.mask_ratio = 0.5

        self.n_bands=1720
        self.seq_size=20
        self.in_chans=1
        self.embed_dim=128
        self.depth=4
        self.num_heads=4
        self.decoder_embed_dim=128
        self.decoder_depth=4
        self.decoder_num_heads=4
        self.mlp_ratio=4.
        self.norm_layer=nn.LayerNorm
        self.cls_token=False
        self.device = device


def seed_all(seed=None):
    """Seed every type of random used by the SRGAN."""
    random.seed(seed)
    np.random.seed(seed)
    if seed is None:
        seed = int(time.time())
    torch.manual_seed(seed)



class Trainer():
    """A class to manage an experimental traning."""
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # self.train_dataset: Dataset = None
        self.train_loader: DataLoader = None
        self.valid_loader: Dataset = None
        
        self.model: Module = None
        self.optimizer: Optimizer = None
        self.signal_quit = False
        self.starting_step = 0

        self.gradient_norm = None

        self.train_loss_list = []
        self.valid_loss_list = []        

    def train(self, fr):
        """
        Run the MAE training for the experiment.
        """
        seed_all(0)

        self.dataset_setup(fr)
        self.model_setup()
        self.prepare_optimizers()
        self.gpu_mode()
        self.train_mode()

        self.training_loop()


    def prepare_optimizers(self):
        """Prepares the optimizers of the network."""
        lr = self.settings.learning_rate
        
        weight_decay = self.settings.weight_decay
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_mode(self):
        """
        Converts the networks to train mode.
        """
        self.model.train()

    def gpu_mode(self):
        """
        Moves the networks to the GPU (if available).
        """
        self.model.to(self.settings.device)

    def eval_mode(self):
        """
        Changes the network to evaluation mode.
        """
        self.model.eval()

    def cpu_mode(self):
        """
        Moves the networks to the CPU.
        """
        self.model.to('cpu')

    def optimizer_to_gpu(self, optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    def dataset_setup(self, fr):
        """Prepares all the datasets and loaders required for the application."""        
        x_p = torch.Tensor(fr.values) #fr_clean
        
        # get training indices that wil be used for validation
        train_size = len(x_p)
        indices = list(range(train_size))
        np.random.shuffle(indices)
        split = int(np.floor(self.settings.valid_size * train_size))
        train_idx, valid_idx = indices[split:], indices[:split]
        
        # # define samplers to obtain training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # # prepare data loaders
        self.train_loader = DataLoader(x_p, batch_size=self.settings.batch_size, sampler=train_sampler)
        self.valid_loader = DataLoader(x_p, batch_size=self.settings.batch_size, sampler=valid_sampler)


    def model_setup(self):
        """Prepares all the model architectures required for the application."""
        self.model = MaskedAutoencoder(n_bands=self.settings.n_bands, seq_size=self.settings.seq_size, in_chans=self.settings.in_chans,
                 embed_dim=self.settings.embed_dim, depth=self.settings.depth, num_heads=self.settings.num_heads,
                 decoder_embed_dim=self.settings.decoder_embed_dim, decoder_depth=self.settings.decoder_depth, decoder_num_heads=self.settings.decoder_num_heads,
                 mlp_ratio=self.settings.mlp_ratio, norm_layer=self.settings.norm_layer, cls_token=self.settings.cls_token)


    def adjust_learning_rate(self, step):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.settings.learning_rate * (0.1 ** (step // self.settings.epochs)) #100000
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    
    def train_step(self, batch):
        # prepare model for training
        # model.train()
        self.train_mode()
        self.optimizer.zero_grad()
        
        x = batch
        x = x.to(self.settings.device).view(x.shape[0], x.shape[-1])
        
        loss, pred, mask, z = self.model(x, mask_ratio= self.settings.mask_ratio)
        
        if self.model.is_cls_token:
          z = z[:, 0, :]
        else:
          z = torch.mean(z[:, 1:, :], dim=1)
        
        loss.backward()
        self.optimizer.step()
        return z.detach().cpu(), loss.item()
    
    def val_step(self,batch):
        # set model to evaluation mode
        # self.model.eval()
        self.eval_mode()
        
        x = batch
        x = x.to(self.settings.device).view(x.shape[0], x.shape[-1])
        
        with torch.no_grad():
          loss, pred, mask, z = self.model(x, mask_ratio=self.settings.mask_ratio)
        
        if self.model.is_cls_token:
          z = z[:, 0, :]
        else:
          z = torch.mean(z[:, 1:, :], dim=1)
        
        return z.cpu(), loss.item()

    def training_loop(self):
        """Runs the main training loop."""
        
        curr = time.process_time()
        
        for e in range(1, self.settings.epochs):
            train_loss = 0.0
            valid_loss = 0.0
            
            for images in tqdm(self.train_loader,
                            total=len(self.train_loader),
                            desc=f'Training epoch {e}'):
                # # move to gpu if available
                # images = images.to(self.settings.device).view(images.shape[0], images.shape[-1])
                
                z, loss = self.train_step(images)
                # track loss
                train_loss += loss
            
            # validate model
            for images in self.valid_loader:
                # move to gpu if available
                # images = images.to(self.settings.device).view(images.shape[0], images.shape[-1])
                z, loss_val = self.val_step(images)
                
                valid_loss += loss_val
            
            # get average loss values
            train_loss = train_loss / len(self.train_loader)
            valid_loss = valid_loss / len(self.valid_loader)
            
            self.train_loss_list.append(train_loss)
            self.valid_loss_list.append(valid_loss)
            
            # output training statistics for epoch
            print('Training Loss: {:.6f} \t Validation Loss: {:.6f}'
                        .format( train_loss, valid_loss))
            
            self.adjust_learning_rate(e)
        
        end = time.process_time()- curr
        print('the training process took {}s >> {}h'.format(end, end/3600))