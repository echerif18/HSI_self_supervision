"""
Regression semi-supervised GAN code.
"""
import datetime
import os
import re
import select
import sys
# from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm
from torch.nn import Module
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import Tensor

from utils import *
from Gan_models import *


from tqdm import tqdm
import json

# from settings import Settings
# from utility import SummaryWriter, gpu, make_directory_name_unique, MixtureModel, seed_all, norm_squared, square_mean

class SrGAN():
    """A class to manage an experimental trial."""
    def __init__(self, settings: Settings):
        # self.dataset_class = None
        self.settings = settings
        
        self.train_dataset: Dataset = None
        self.train_dataset_loader: DataLoader = None
        self.unlabeled_dataset: Dataset = None
        self.unlabeled_dataset_loader: DataLoader = None
        self.validation_dataset: Dataset = None
        
        self.D: Module = None
        self.d_optimizer: Optimizer = None
        self.G: Module = None
        self.g_optimizer: Optimizer = None
        self.signal_quit = False
        self.starting_step = 0

        self.labeled_features = None
        self.unlabeled_features = None
        self.fake_features = None
        self.interpolates_features = None
        self.gradient_norm = None

        self.glob_gen_loss = None
        self.glob_disc_loss = None
        
        self.scaler_list = None

    def train(self):
        """
        Run the SRGAN training for the experiment.
        """
        seed_all(0)

        self.dataset_setup(val_x, val_y, fr_sup, y_sup, fr_unsup, scaler_list)
        self.model_setup(latent_dim, input_shape, n_lb)
        self.prepare_optimizers()
        # self.load_models()
        self.gpu_mode()
        self.train_mode()

        self.training_loop()

    def training_loop(self, n_epochs=200):
        """Runs the main training loop."""
        
        for epoch in range(1,n_epochs):
            step = self.starting_step
            sup_train_iterator = iter(self.train_dataset_loader)
            unsup_train_iterator = iter(self.unlabeled_dataset_loader)
            
            for unlabeled_examples in tqdm(unsup_train_iterator, total=len(self.unlabeled_dataset_loader), desc=f'Training epoch {epoch}'):
                self.adjust_learning_rate(step)
            
                # GAN.
                unlabeled_examples = unlabeled_examples.to(gpu)
                
                labeled_examples, labels  = next(sup_train_iterator)
                labeled_examples = labeled_examples.to(gpu)
                labels = labels.to(gpu)
            
                ######
                self.gan_training_step(labeled_examples, labels, unlabeled_examples, step)
                step=+1
    

    def prepare_optimizers(self):
        """Prepares the optimizers of the network."""
        g_lr = self.settings.learning_rate
        d_lr = 4 * self.settings.learning_rate
        
        weight_decay = self.settings.weight_decay
        self.d_optimizer = Adam(self.D.parameters(), lr=d_lr, weight_decay=weight_decay)
        self.g_optimizer = Adam(self.G.parameters(), lr=g_lr)

    def train_mode(self):
        """
        Converts the networks to train mode.
        """
        self.D.train()
        self.G.train()

    def gpu_mode(self):
        """
        Moves the networks to the GPU (if available).
        """
        self.D.to(gpu)
        self.G.to(gpu)

    def eval_mode(self):
        """
        Changes the network to evaluation mode.
        """
        self.D.eval()
        self.G.eval()

    def cpu_mode(self):
        """
        Moves the networks to the CPU.
        """
        self.D.to('cpu')
        self.G.to('cpu')

    def optimizer_to_gpu(self, optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    def disable_batch_norm_updates(module):
        """Turns off updating of batch norm statistics."""
        # noinspection PyProtectedMember
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


    def gan_training_step(self, labeled_examples, labels, unlabeled_examples, step):
        """Runs an individual round of GAN training."""
        # Labeled.
        # self.D.apply(disable_batch_norm_updates)  # No batch norm
        self.d_optimizer.zero_grad()
        labeled_loss = self.labeled_loss_calculation(labeled_examples, labels)
        labeled_loss.backward()
        # Unlabeled.
        # self.D.apply(disable_batch_norm_updates)  # Make sure only labeled data is used for batch norm statistics
        unlabeled_loss = self.unlabeled_loss_calculation(labeled_examples, unlabeled_examples)
        unlabeled_loss.backward()
        # Fake.
        z = torch.tensor(MixtureModel([norm(-self.settings.mean_offset, 1),
                                       norm(self.settings.mean_offset, 1)]
                                      ).rvs(size=[unlabeled_examples.size(0),
                                                  self.G.latent_dim]).astype(np.float32)).to(gpu)
        fake_examples = self.G(z)
        fake_loss = self.fake_loss_calculation(unlabeled_examples, fake_examples)
        fake_loss.backward()
        # Gradient penalty.
        gradient_penalty = self.gradient_penalty_calculation(fake_examples, unlabeled_examples)
        gradient_penalty.backward()
        # Discriminator update.
        self.d_optimizer.step()
        # Generator.
        if step % self.settings.generator_training_step_period == 0:
            self.g_optimizer.zero_grad()
            z = torch.randn(unlabeled_examples.size(0), self.G.latent_dim).to(gpu)
            fake_examples = self.G(z)
            generator_loss = self.generator_loss_calculation(fake_examples, unlabeled_examples)
            generator_loss.backward()
            self.g_optimizer.step()
        return labeled_loss, unlabeled_loss, fake_loss, gradient_penalty, generator_loss

    def labeled_loss_calculation(self, labeled_examples, labels):
        """Calculates the labeled loss."""
        predicted_labels = self.D(labeled_examples)[0]
        self.labeled_features = self.D(labeled_examples)[1]
        labeled_loss = self.labeled_loss_function(predicted_labels, labels, threshold=1)
        labeled_loss *= self.settings.labeled_loss_multiplier
        return labeled_loss

    def unlabeled_loss_calculation(self, labeled_examples: Tensor, unlabeled_examples: Tensor):
        """Calculates the unlabeled loss."""
        # _ = self.D(labeled_examples)[0]
        self.labeled_features = self.D(labeled_examples)[1]
        # _ = self.D(unlabeled_examples)[0]
        self.unlabeled_features = self.D(unlabeled_examples)[1]
        unlabeled_loss = self.feature_distance_loss(self.unlabeled_features, self.labeled_features)
        unlabeled_loss *= self.settings.matching_loss_multiplier
        unlabeled_loss *= self.settings.srgan_loss_multiplier
        return unlabeled_loss


    def fake_loss_calculation(self, unlabeled_examples: Tensor, fake_examples: Tensor):
        """Calculates the fake loss."""
        # _ = self.D(unlabeled_examples)[0]
        self.unlabeled_features = self.D(unlabeled_examples)[1]
        # _ = self.D(fake_examples.detach())[0]
        self.fake_features = self.D(fake_examples.detach())[1]
        fake_loss = self.feature_distance_loss(self.unlabeled_features, self.fake_features,
                                               distance_function=self.settings.contrasting_distance_function)
        fake_loss *= self.settings.contrasting_loss_multiplier
        fake_loss *= self.settings.srgan_loss_multiplier
        return fake_loss


    def gradient_penalty_calculation(self, fake_examples: Tensor, unlabeled_examples: Tensor) -> Tensor:
        """Calculates the gradient penalty from the given fake and real examples."""
        alpha_shape = [1] * len(unlabeled_examples.size())
        # alpha_shape[0] = self.settings.batch_size
        alpha_shape[0] = unlabeled_examples.size(0)
        alpha = torch.rand(alpha_shape, device=gpu)
        interpolates = (alpha * unlabeled_examples.detach().requires_grad_() +
                        (1 - alpha) * fake_examples.detach().requires_grad_())
        interpolates_loss = self.interpolate_loss_calculation(interpolates)
        gradients = torch.autograd.grad(outputs=interpolates_loss, inputs=interpolates,
                                        grad_outputs=torch.ones_like(interpolates_loss, device=gpu),
                                        create_graph=True)[0]
        gradient_norm = gradients.view(unlabeled_examples.size(0), -1).norm(dim=1)
        self.gradient_norm = gradient_norm
        norm_excesses = torch.max(gradient_norm - 1, torch.zeros_like(gradient_norm))
        gradient_penalty = (norm_excesses ** 2).mean() * self.settings.gradient_penalty_multiplier
        return gradient_penalty

    def interpolate_loss_calculation(self, interpolates):
        """Calculates the interpolate loss for use in the gradient penalty."""
        _ = self.D(interpolates)[0]
        self.interpolates_features = self.D(interpolates)[1]
        return self.interpolates_features.norm(dim=1)

    def generator_loss_calculation(self, fake_examples, unlabeled_examples):
        """Calculates the generator's loss."""
        # _ = self.D(fake_examples)[0]
        self.fake_features = self.D(fake_examples)[1]
        # _ = self.D(unlabeled_examples)[0]
        detached_unlabeled_features = self.D(unlabeled_examples)[1].detach()
        generator_loss = self.feature_distance_loss(detached_unlabeled_features, self.fake_features)
        generator_loss *= self.settings.matching_loss_multiplier
        return generator_loss

    # @abstractmethod
    def dataset_setup(self, val_x, val_y, fr_sup, y_sup, fr_unsup, scaler_list):
        """Prepares all the datasets and loaders required for the application."""
        self.scaler_list = scaler_list
        x_p = torch.Tensor(torch.Tensor(val_x.values)).unsqueeze(dim=1)
        lb_p = torch.Tensor(torch.Tensor(scaler_list.transform(val_y.values)))
        self.validation_dataset = TensorDataset(x_p,lb_p)  
        self.valid_loader = DataLoader(self.validation_dataset, batch_size=self.settings.batch_size)
        
        x_p = torch.Tensor(fr_sup.values).unsqueeze(dim=1) #fr_clean
        lb_p = torch.Tensor(scaler_list.transform(y_sup.values))
        self.train_dataset = TensorDataset(x_p,lb_p)
        
        # Calculate the number of samples to be drawn with replacement
        num_samples_with_replacement = len(fr_unsup) #int(len(fr_unsup) - len(sup_dataset)- len(dataset_val))
        # Create a sampler that samples with replacement
        sampler_rep = torch.utils.data.sampler.RandomSampler(self.train_dataset, replacement=True, num_samples=num_samples_with_replacement)
        
        self.train_dataset_loader = DataLoader(self.train_dataset, batch_size= self.settings.batch_size, sampler= sampler_rep, num_workers=2)
        if self.settings.augmentation:
            ### data augmentation ##
            self.train_dataset_loader.transform = transforms.Compose([
                transforms.Lambda(lambda x, y: self.train_dataset_loader.data_augmentation(x, y))
            ])
        
        x_p = torch.Tensor(fr_unsup.values).unsqueeze(dim=1)
        # self.unlabeled_dataset = TensorDataset(x_p)
        self.unlabeled_dataset_loader = DataLoader(x_p, batch_size= self.settings.batch_size, shuffle=True, num_workers=2)

    @staticmethod
    def infinite_iter(dataset):
        """Create an infinite generator from a dataset"""
        while True:
            for examples in dataset:
                yield examples

    def model_setup(self, latent_dim, input_shape, n_lb):
        """Prepares all the model architectures required for the application."""
        self.D = Discriminator(input_shape, n_classes=n_lb) 
        self.G = Generator(latent_dim, input_shape)

    @staticmethod
    def labeled_loss_function(y_pred, y_true, threshold=1):
        """Calculate the loss from the label difference prediction."""
        bool_finite = torch.isfinite(y_true)
        error = y_pred[bool_finite] - y_true[bool_finite]
        
        is_small_error = torch.abs(error) < threshold
        squared_loss = torch.square(error) / 2
        linear_loss = threshold * torch.abs(error) - threshold**2 / 2
        
        return torch.mean(torch.where(is_small_error, squared_loss, linear_loss))

    def evaluate(self):
        """Evaluates the model on the test dataset (needs to be overridden by subclass)."""
        self.model_setup()
        self.load_models()
        self.eval_mode()

    def adjust_learning_rate(self, step):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.settings.learning_rate * (0.1 ** (step // 100000))
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = lr

    def feature_distance_loss(self, base_features, other_features, distance_function=None):
        """Calculate the loss based on the distance between feature vectors."""
        if distance_function is None:
            distance_function = self.settings.matching_distance_function
        base_mean_features = base_features.mean(0)
        other_mean_features = other_features.mean(0)
        if self.settings.normalize_feature_norm:
            epsilon = 1e-5
            base_mean_features = base_mean_features / (base_mean_features.norm() + epsilon)
            other_mean_features = other_features / (other_mean_features.norm() + epsilon)
        distance_vector = distance_function(base_mean_features - other_mean_features)
        return distance_vector