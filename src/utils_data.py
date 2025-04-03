import os
from pickle import dump,load
from sklearn.preprocessing import PowerTransformer

# from torchvision import transforms
from torch.utils.data import Sampler, Dataset, DataLoader, TensorDataset
import numpy as np
import math
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import torch
import random

# from sklearn.preprocessing import StandardScaler  # Assuming scalers come from scikit-learn
from typing import Optional, List, Union, Any


# Traits = ['LMA (g/m²)', 'N content (mg/cm²)', 'LAI (m²/m²)', 'C content (mg/cm²)', 'Chl content (μg/cm²)', 'EWT (mg/cm²)', 
# 'Carotenoid content (μg/cm²)', 'Phosphorus content (mg/cm²)', 'Lignin (mg/cm²)', 'Cellulose (mg/cm²)', 
# 'Fiber (mg/cm²)',
# 'Anthocyanin content (μg/cm²)',
# 'NSC (mg/cm²)',
# 'Magnesium content (mg/cm²)',
# 'Ca content (mg/cm²)',
# 'Potassium content (mg/cm²)',
# 'Boron content (mg/cm²)',
# 'Copper content (mg/cm²)',
# 'Sulfur content (mg/cm²)',
# 'Manganese content (mg/cm²)']

# Traits = ['LMA_g_m2', 'N_area_mg_cm2', 'LAI_m2_m2', 'C_area_mg_cm2', 'Chl_area_ug_cm2', 'EWT_mg_cm2',
#           'Car_area_ug_cm2', 'Anth_area_ug_cm2', 'Protein_g_m2']
       # 'Boron_area_mg_cm2', 'Boron_mass_mg_g',
       #  'C_mass_mg_g', 'Ca_area_mg_cm2', 'Ca_mass_mg_g',
       #  'Car_mass_mg_g', 'Cellulose_mg_cm2',
       #  'Cu_area_mg_cm2',
       # 'Cu_mass_mg_g',  'Fiber_mg_cm2', 'Fiber_mg_g',
       # 'Flavonoids_area_mg_cm2', 'Flavonoids_mass_mg_g', 'Iron_area_mg_cm2',
       # 'Iron_mass_mg_g', 'LDMC_g_g', 'LWC%',
       # 'Lignin_mg_cm2', 'Lignin_mg_g', 'Mg_area_mg_cm2', 'Mg_mass_mg_g',
       # 'Mn_area_mg_cm2', 'Mn_mass_mg_g',  'N_mass_mg_g',
       # 'NSC_mg_cm2', 'NSC_mg_g', 'Phenolics_area_mg_cm2',
       # 'Phenolics_mass_mg_g', 'P_area_mg_cm2', 'P_mass_mg_g',
       # 'Potassium_area_mg_cm2', 'Potassium_mass_mg_g',  'RWC%',
       # 'Starch_area_mg_cm2', 'Starch_mass_mg_g', 'Sugar_area_mg_cm2',
       # 'Sugar_mass_mg_g', 'S_area_mg_cm2', 'S_mass_mg_g', 'Zn _area_mg_cm2',
       # 'Zn_mass_mg_g'

######### Raw data ##########

def read_db(file, sp=False, encoding=None):
    db = pd.read_csv(file, encoding=encoding, low_memory=False)
    db.drop(['Unnamed: 0'], axis=1, inplace=True)
    if (sp):
        features = db.loc[:, "400":"2500"]
        labels = db.drop(features.columns, axis=1)
        return db, features, labels
    else:
        return db

### Apply savgol filter for a wavelength filter, 
def filter_segment(features_noWtab, order=1,der= False):
    #features_noWtab: Segment of the signal
    #order: Order of the savgol filter
    #der: If with first derivative
    
#     part1 = features_noWtab.loc[:,indx]
    part1 = features_noWtab.copy()
    if (der):
        fr1 = savgol_filter(part1, 65, 1,deriv=1)
    else:
        fr1 = savgol_filter(part1, 65, order)
    fr1 = pd.DataFrame(data=fr1, columns=part1.columns)
    return fr1


def feature_preparation(features, inval = [1351,1431, 1801, 2051], frmax=2451, order=1,der= False):
    # features: The original reflectance signal
    #order: Order of the savgol filter
    #der: If with first derivative
    
    features.columns = features.columns.astype('int')
    features[features<0] = 0   
    
    #####Substitute high values with the mean of neighbour values
    other = features.copy()
    other[other>1] = np.nan
    other = (other.fillna(method='ffill') + other.fillna(method='bfill'))/2
    other=other.interpolate(method='linear', axis=1).ffill().bfill()
    
    wt_ab = [i for i in range(inval[0],inval[1])]+[i for i in range(inval[2],inval[3])]+[i for i in range(2451,2501)] 

    features_Wtab = other.loc[:,wt_ab]
    features_noWtab=other.drop(wt_ab,axis=1)
    
    fr1 = filter_segment(features_noWtab.loc[:,:inval[0]-1], order = order, der = der)
    fr2 = filter_segment(features_noWtab.loc[:,inval[1]:inval[2]-1], order = order,der = der)
    fr3 = filter_segment(features_noWtab.loc[:,inval[3]:frmax], order = order,der = der)    
    
    
    inter = pd.concat([fr1,fr2,fr3], axis=1, join='inner')
    inter[inter<0]=0
    
    return inter


######## calculate sample weights from meta data #########
def samp_w(w_train, train_x):
    wstr = 100 - 100 * (w_train.loc[train_x.index, :].groupby(['dataset'])['numSamples'].count() /
                        w_train.loc[train_x.index, :].shape[0])
    samp_w_tr = np.array(w_train.loc[train_x.index, 'dataset'].map(dict(wstr)), dtype='float')
    return samp_w_tr


def data_prep_db(db_val_lb, ls_tr, weight_sample=False):
    val_x = feature_preparation(db_val_lb.loc[:, '400':'2500']).loc[:, 400:2450]
    val_x.index = db_val_lb.index
    
    val_y = db_val_lb[ls_tr]
    
    if(weight_sample):
        w_val = samp_w(db_val_lb.iloc[:,:8], db_val_lb)
        return val_x, val_y, w_val
    else:
        return val_x, val_y

# def data_prep(minl, gap_fil, Traits, i = len(Traits)-1, w_train=None, multi=False):

#     ##########Testing/validation data preparation (only for the last added trait)#######
#     if (multi):
#         train_x = gap_fil.loc[:, minl:]
#         train_y = gap_fil.loc[train_x.index, Traits[:i + 1]]
#     else:
#         train_x = gap_fil.loc[gap_fil[gap_fil[Traits[i]].notnull()].index, minl:]
#         train_y = gap_fil.loc[train_x.index, Traits[i:i + 1]]

#     if(w_train is not None):
#         samp_w_tr = samp_w(w_train, train_x)  # >>>>>>samples weights calculation
#         return train_x, train_y, samp_w_tr
#     else:
#         return train_x, train_y


# def balanceData(db_train, w_train, Traits, random_state=300,percentage=1):
#         ### The maximum number of samples within a dataset ##
#         mx = pd.concat([w_train.reset_index(drop=True),db_train.reset_index(drop=True)], axis=1).groupby('dataset')[Traits].count().max().max()*percentage
#         fill = pd.concat([w_train.reset_index(drop=True),db_train.reset_index(drop=True)], axis=1).groupby('dataset').sample(n=int(mx), random_state=random_state, replace=True)#.reset_index(drop=True)
#         return fill


def balanceData(db_train, w_train, Traits, random_state=300,percentage=1):
        ### The maximum number of samples within a dataset ##
        # mx = pd.concat([w_train.reset_index(drop=True),db_train.reset_index(drop=True)], axis=1).groupby('dataset')[Traits].count().max().max()*percentage
        mx = pd.concat([w_train.reset_index(drop=True),db_train.reset_index(drop=True)], axis=1).groupby('dataset').numSamples.count().max().max()*percentage
        # fill = pd.concat([w_train.reset_index(drop=True),db_train.reset_index(drop=True)], axis=1).groupby('dataset').sample(n=int(mx),random_state = random_state,replace=True)#.reset_index(drop=True)
        fill = pd.concat([w_train, db_train], axis=1).groupby('dataset').sample(n=int(mx),random_state = random_state,replace=True)#.reset_index(drop=True)
        return fill

# def save_scaler(train_y, save=False, dir_n=None, k=None, standardize=False):
#     scaler = PowerTransformer(method='box-cox', standardize=standardize).fit(np.array(train_y)) # method='yeo-johnson' box-cox
#     if save:
#         if not os.path.exists(dir_n):
#             os.mkdir(dir_n)
#         dump(scaler, open(dir_n + '/scaler_{}.pkl'.format(k), 'wb')) 
#     return scaler


def save_scaler(train_y, save=False, dir_n=None, k=None, standardize=False, scale=False):
    from sklearn.preprocessing import FunctionTransformer, PowerTransformer
    
    if(scale):
        scaler = PowerTransformer(method='box-cox', standardize=standardize).fit(np.array(train_y)) # method='yeo-johnson' box-cox
    else:
        identity = FunctionTransformer(None)
        scaler = identity.fit(np.array(train_y))
    if save:
        if not os.path.exists(dir_n):
            os.mkdir(dir_n)
        dump(scaler, open(dir_n + '/scaler_{}.pkl'.format(k), 'wb')) 
    return scaler


def split_data(path_data, ls_tr, percentage_tr=1, random_state=300):
    db_lb_all = pd.read_csv(path_data, low_memory=False).drop(['Unnamed: 0'], axis=1)   
    
    ### external
    groups = db_lb_all.groupby('dataset')
    
    val_ext_idx = list(groups.get_group(32).index)+list(groups.get_group(3).index)
    samples_val_ext = db_lb_all.loc[val_ext_idx,:]
    
    db_lb_all.drop(val_ext_idx, inplace=True)
    
    ##### validation ä##
    w_train = db_lb_all.iloc[:,:8]
    db_train = db_lb_all.copy().drop(w_train.columns,axis=1)
    
    db_val_lb = balanceData(db_train, w_train, ls_tr, random_state=random_state, percentage=0.1)
    val_idx = list(set(db_val_lb.index))
    
    db_lb_all.drop(val_idx, inplace=True)
    
    ## train
    db_lb = db_lb_all.sample(n = int(percentage_tr*db_lb_all.shape[0]), random_state=random_state)#.reset_index(drop=True)
    return db_lb, db_val_lb, samples_val_ext




############# Dtaa loader + balancing data samples  ##
def dataaugment(x, betashift=0.05, slopeshift=0.05, multishift=0.05, kind='shift', std_dev = 0.02):
    
    if kind == 'shift':
        beta = (torch.rand(1) * 2 * betashift - betashift)#.to(device)
        slope = (torch.rand(1) * 2 * slopeshift - slopeshift + 1)#.to(device)
    
        if len(x.shape) == 1:
            axis = torch.arange(x.shape[0], dtype=torch.float) / float(x.shape[0])
        else:
            axis = torch.arange(x.shape[1], dtype=torch.float) / float(x.shape[1])
    
        offset = (slope * (axis) + beta - axis - slope / 2. + 0.5)#.to(device)
        multi = (torch.rand(1) * 2 * multishift - multishift + 1)#.to(device)
        return multi * x + offset
        
    if kind == 'noise':
        # signal = x.clone().detach() #torch.tensor(x)#.to(device)

        # Define the standard deviation of the Gaussian noise
        # std_dev = std_dev #torch.std(torch.tensor(features.values),0) # Adjust this value according to the desired amount of noise
        
        # Generate Gaussian noise with the same shape as the signal
        noise = (torch.randn_like(x) * std_dev)#.to(device)
        
        # Add the noise to the signal
        return (x.clone() + noise)#.detach()


# # Custom transformation class
# class AugmentationTransform(object):
#     def __init__(self, augmentation_ratio=0.3, std_dev = 0.02,  kind='noise'):
#         self.augmentation_ratio = augmentation_ratio
#         self.std_dev = std_dev
#         self.kind = kind

#     def __call__(self, sample):
#         x, y = sample
#         if random.random() < self.augmentation_ratio:
#             x = dataaugment(x, kind = self.kind, std_dev = self.std_dev)
#         return x, y

# Custom transformation class
class AugmentationTransform(object):
    def __init__(self, augmentation_ratio=0.3, std_dev = 0.02,  kind='noise'):
        self.augmentation_ratio = augmentation_ratio
        self.std_dev = std_dev
        self.kind = kind

    def __call__(self, sample):
        x = sample
        if random.random() < self.augmentation_ratio:
            x = dataaugment(x, kind = self.kind, std_dev = self.std_dev)
        return x



class BalancedSampler(Sampler):
    def __init__(self, db_train, w_train, traits, percentage=1.0, random_state=300):
        self.db_train = db_train
        self.w_train = w_train
        self.traits = traits
        self.percentage = percentage
        self.random_state = random_state

        # Combine the dataframes and reset indices
        self.data = pd.concat([w_train.reset_index(drop=True), db_train.reset_index(drop=True)], axis=1)

        # Calculate the maximum number of samples
        mx = int(self.data.groupby('dataset')[self.traits].count().max().max() * self.percentage)

        # Sample data to balance
        self.balanced_data = self.data.groupby('dataset').sample(n=mx, random_state=self.random_state, replace=True)

        # Original indices of the balanced data
        self.indices = self.balanced_data.index.tolist()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


### base data set ####
class DatasetLoader:
    def __init__(self, val_x, val_y, fr_sup, y_sup, fr_unsup, batch_size, scaler_list=None):
        self.scaler_list = scaler_list
        self.batch_size = batch_size
        
        # Validation dataset
        x_p_val = torch.tensor(val_x.values, dtype=torch.float).unsqueeze(dim=1)
        if scaler_list is not None:
            lb_p_val = torch.tensor(scaler_list.transform(val_y.values),dtype=torch.float)
        else:
            lb_p_val = torch.tensor(val_y.values,dtype=torch.float)
        self.validation_dataset = TensorDataset(x_p_val, lb_p_val)
        
        # Validation DataLoader
        self.valid_loader = DataLoader(self.validation_dataset, batch_size=batch_size)

        # Unlabeled DataLoader
        x_p_unsup = torch.tensor(fr_unsup.values).unsqueeze(dim=1)
        self.unlabeled_dataset_loader = DataLoader(x_p_unsup, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Training dataset
        x_p_train = torch.tensor(fr_sup.values, dtype=torch.float).unsqueeze(dim=1)
        if scaler_list is not None:
            lb_p_train = torch.tensor(scaler_list.transform(y_sup.values), dtype=torch.float) #dtype=torch.float
        else:
            lb_p_train = torch.tensor(y_sup.values, dtype=torch.float)
        self.train_dataset = TensorDataset(x_p_train, lb_p_train)
        
        # Calculate the number of samples to be drawn with replacement
        num_samples_with_replacement = len(fr_unsup)
        # num_samples_with_replacement = len(self.unlabeled_dataset_loader.dataset)
        # Create a sampler that samples with replacement
        sampler_rep = torch.utils.data.sampler.RandomSampler(self.train_dataset, replacement=True, num_samples=num_samples_with_replacement)
        
        # Training DataLoader
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, sampler=sampler_rep, num_workers=0)


# class DatasetLoader:
#     def __init__(
#         self, 
#         val_x: Any, 
#         val_y: Any, 
#         fr_sup: Any, 
#         y_sup: Any, 
#         fr_unsup: Any, 
#         batch_size: int, 
#         scaler_list: Optional[PowerTransformer] = None, 
#         num_workers: int = 0,
#         sample_with_replacement: bool = True
#     ):
#         """
#         DatasetLoader class to create DataLoaders for validation, training, and unsupervised datasets.

#         Args:
#             val_x (Any): Validation input features (e.g., DataFrame or NumPy array).
#             val_y (Any): Validation target labels.
#             fr_sup (Any): Supervised training input features.
#             y_sup (Any): Supervised training target labels.
#             fr_unsup (Any): Unsupervised input features (unlabeled data).
#             batch_size (int): Batch size for the DataLoader.
#             scaler_list (Optional[StandardScaler]): Optional scaler for normalizing labels.
#             num_workers (int): Number of worker threads for data loading (0 for no parallelism).
#             sample_with_replacement (bool): Whether to sample training data with replacement.
#         """
#         self.scaler_list = scaler_list
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.sample_with_replacement = sample_with_replacement
        
#         # Create validation DataLoader
#         self.valid_loader = self._create_loader(val_x, val_y, batch_size, is_train=False)

#         # Create unsupervised DataLoader (without labels)
#         self.unlabeled_loader = self._create_unlabeled_loader(fr_unsup, batch_size)

#         # Create supervised training DataLoader
#         self.train_loader = self._create_loader(fr_sup, y_sup, batch_size, is_train=True, fr_unsup=fr_unsup)

#     def _create_loader(
#         self, 
#         x_data: Any, 
#         y_data: Any, 
#         batch_size: int, 
#         is_train: bool = False, 
#         fr_unsup: Optional[Any] = None
#     ) -> DataLoader:
#         """
#         Helper function to create DataLoader for supervised datasets.

#         Args:
#             x_data (Any): Input features.
#             y_data (Any): Target labels.
#             batch_size (int): Batch size.
#             is_train (bool): Whether this is the training dataset.
#             fr_unsup (Optional[Any]): Unsupervised data (used for matching samples if sampling with replacement).

#         Returns:
#             DataLoader: The DataLoader object for the dataset.
#         """
#         # Convert data to tensors
#         x_tensor = torch.tensor(x_data.values, dtype=torch.float).unsqueeze(dim=1)
#         y_tensor = self._transform_labels(y_data)
        
#         # Create the dataset
#         dataset = TensorDataset(x_tensor, y_tensor)
        
#         # Apply sampling strategy for training data if requested
#         if is_train and self.sample_with_replacement:
#             num_samples = len(fr_unsup) if fr_unsup is not None else len(dataset)
#             sampler = torch.utils.data.sampler.RandomSampler(dataset, replacement=True, num_samples=num_samples)
#             return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=self.num_workers)
#         else:
#             return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=self.num_workers)

#     def _create_unlabeled_loader(self, x_data: Any, batch_size: int) -> DataLoader:
#         """
#         Helper function to create DataLoader for unlabeled datasets.

#         Args:
#             x_data (Any): Unlabeled input features.
#             batch_size (int): Batch size.

#         Returns:
#             DataLoader: The DataLoader object for the unlabeled dataset.
#         """
#         x_tensor = torch.tensor(x_data.values, dtype=torch.float).unsqueeze(dim=1)
#         return DataLoader(x_tensor, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)

#     def _transform_labels(self, y_data: Any) -> torch.Tensor:
#         """
#         Helper function to scale target labels if a scaler is provided.

#         Args:
#             y_data (Any): Target labels.

#         Returns:
#             torch.Tensor: Scaled or unscaled target labels as a tensor.
#         """
#         if self.scaler_list is not None:
#             return torch.tensor(self.scaler_list.transform(y_data.values), dtype=torch.float)
#         else:
#             return torch.tensor(y_data.values, dtype=torch.float)


#### for AE_RTM###
# def infinite_iter(train_dataset_loader, unlabeled_dataset_loader, augmentation=False):
#     data_loader_itr = iter(train_dataset_loader)
#     data_loader_un_itr = iter(unlabeled_dataset_loader)
    
#     while True:
#         try:
#             # Attempt to fetch the next batch from each iterator
#             unlabeled_examples = next(data_loader_un_itr)
#             labeled_examples, labels = next(data_loader_itr)
            
#             #### in the training ###
#             if augmentation:
#                 kind_option = ['shift', 'noise']
#                 kind = np.random.choice(kind_option, 1)[0]
                
#                 transform = AugmentationTransform(augmentation_ratio=0.3, std_dev=0.01, kind=kind)
#                 labeled_examples = transform((labeled_examples))
            
#             shape = (len(unlabeled_examples), labels.shape[1])
#             # Create a tensor filled with NaN values
#             nan_tensor = torch.full(shape, float('nan'))
            
#             samples = torch.cat([labeled_examples, unlabeled_examples])
#             samples_lb = torch.cat([labels, nan_tensor])
            
#             yield samples, samples_lb
            
#         except StopIteration:
#             # Reset the iterator if it's exhausted
#             break
#             # data_loader_itr = iter(train_dataset_loader)
#             # data_loader_un_itr = iter(unlabeled_dataset_loader)



def infinite_iter(train_dataset_loader, unlabeled_dataset_loader):
    data_loader_itr = iter(train_dataset_loader)
    data_loader_un_itr = iter(unlabeled_dataset_loader)
    
    while True:
        try:
            # Attempt to fetch the next batch from the unlabeled dataset iterator
            unlabeled_examples = next(data_loader_un_itr)
        except StopIteration:
            # If the unlabeled dataset iterator is exhausted, stop the infinite loop
            # print("Unlabeled dataset fully consumed. Stopping.")
            break
        try:
            # Attempt to fetch the next batch from the labeled dataset iterator
            labeled_examples, labels, _ = next(data_loader_itr)
        except StopIteration:
            # If the labeled dataset iterator is exhausted, reset it
            data_loader_itr = iter(train_dataset_loader)
            labeled_examples, labels, _ = next(data_loader_itr)

        # Create a tensor filled with NaN values for unlabeled data labels
        shape = (len(unlabeled_examples), labels.shape[1])
        nan_tensor = torch.full(shape, float('nan'))
        
        # Concatenate labeled and unlabeled data and labels
        samples = torch.cat([labeled_examples, unlabeled_examples])
        samples_lb = torch.cat([labels, nan_tensor])
        
        # Yield the combined batch
        yield samples, samples_lb



# https://www.kaggle.com/code/alejopaullier/hms-efficientnetb0-pytorch-inference : have a look 
#### for supervised ####
class CustomDataset(Dataset):
    def __init__(
        self, 
        data_x: torch.Tensor, 
        data_y: torch.Tensor, 
        scaler_list: Optional[List[PowerTransformer]] = None, 
        augment: bool = False, 
        augmentation_ratio: float = 0., 
        w_tr: Optional[torch.Tensor] = None
    ):
        """
        Custom dataset class for supervised learning with optional augmentation and scaling.
        
        Args:
            data_x (torch.Tensor): Input data (features).
            data_y (torch.Tensor): Output data (labels).
            scaler_list (List[PowerTransformer], optional): List of scalers for normalization.
            augment (bool): Flag to enable data augmentation.
            augmentation_ratio (float): Probability of applying augmentation to each sample.
            w_tr (torch.Tensor, optional): Weights for weighted sampling.
        """
        self.data_x = data_x  # Input data (features)
        self.data_y = data_y  # Output data (labels)
        self.scaler_list = scaler_list  # Scaler list for normalization
        self.augment = augment  # Flag for data augmentation
        self.augmentation_ratio = augmentation_ratio
        self.w_tr = w_tr if w_tr is not None else torch.ones(len(data_x))  # Default weights are 1 if not provided
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data_x)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Get a sample from the dataset, along with its label and weight.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of (input data, label, weight)
        """
        x = self.data_x[idx].squeeze()  # Squeeze to remove extra dimensions if necessary
        y = self.data_y[idx]
        
        # Retrieve the sample weight (if not provided, default weight is 1)
        z = self.w_tr[idx].float() if self.w_tr is not None else torch.tensor(1.0)

        # Apply data augmentation if enabled and chosen based on augmentation_ratio
        if self.augment and torch.rand(1).item() < self.augmentation_ratio:
            x = self.apply_augmentation(x)

        # Normalize data using scaler_list, if available
        if self.scaler_list:
            y = self.scaler_list.transform(y.unsqueeze(0)).squeeze()

        return x, y, z

    def apply_augmentation(self, x: torch.Tensor, std_dev: float = 0.003) -> torch.Tensor:
        """
        Apply augmentation to the input data.
        
        Args:
            x (torch.Tensor): Input data to augment.
            std_dev (float): Standard deviation for noise augmentation.
        
        Returns:
            torch.Tensor: Augmented input data.
        """
        augmentation_type = 'noise' #self.select_augmentation_type()
        
        if augmentation_type == 'shift':
            return self.shift_augmentation(x)
        elif augmentation_type == 'noise':
            return self.noise_augmentation(x, std_dev)
        else:
            return x  # No augmentation if an invalid type is selected

    def select_augmentation_type(self) -> str:
        """
        Randomly select an augmentation type ('shift' or 'noise').
        
        Returns:
            str: The selected augmentation type.
        """
        return 'shift' if torch.rand(1).item() < 0.5 else 'noise'
    
    def shift_augmentation(self, x: torch.Tensor, betashift: float = 0.05, slopeshift: float = 0.05, multishift: float = 0.05) -> torch.Tensor:
        """
        Apply shift augmentation to the input data by modifying its slope and offset.
        
        Args:
            x (torch.Tensor): Input data.
            betashift (float): Range for shifting the offset.
            slopeshift (float): Range for modifying the slope.
            multishift (float): Range for multiplicative scaling of the data.
        
        Returns:
            torch.Tensor: Augmented data.
        """
        axis = torch.arange(x.size(-1)) / float(x.size(-1))  # Normalize axis between 0 and 1

        beta = (torch.rand(1) * 2 - 1) * betashift
        slope = 1 + (torch.rand(1) * 2 - 1) * slopeshift
        offset = slope * axis + beta - axis - slope / 2 + 0.5
        multi = 1 + (torch.rand(1) * 2 - 1) * multishift

        return multi * x + offset

    def noise_augmentation(self, x: torch.Tensor, std_dev: float) -> torch.Tensor:
        """
        Add Gaussian noise to the input data.
        
        Args:
            x (torch.Tensor): Input data.
            std_dev (float): Standard deviation for the noise.
        
        Returns:
            torch.Tensor: Data with noise added.
        """
        noise = torch.randn_like(x) * std_dev
        return x + noise


class SpectraDataset(Dataset):
    def __init__(self, X_train, y_train=None, meta_train=None, augmentation=False, aug_prob=0.5, betashift=0.01, slopeshift=0.01, multishift=0.1):
        """
        Args:
            X_train: Input features (spectra).
            y_train: Labels (None if unlabeled).
            meta_train: Metadata (optional).
            augmentation: Whether to apply augmentation.
            aug_prob: Probability of applying augmentation per sample.
            betashift, slopeshift, multishift: Parameters for shift augmentation.
        """
        self.X_train = np.array(X_train)  # Ensure these are NumPy arrays
        self.y_train = None if y_train is None else np.array(y_train)
        self.meta_train = None if meta_train is None else np.array(meta_train.dataset)
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.betashift = betashift  # Reduced parameter for minimal shift
        self.slopeshift = slopeshift
        self.multishift = multishift

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        # Retrieve the corresponding spectra
        x = torch.tensor(self.X_train[idx], dtype=torch.float32)
        
        # Optionally retrieve the label if available
        y = None if self.y_train is None else torch.tensor(self.y_train[idx], dtype=torch.float32)
        
        # Optionally retrieve metadata if available
        meta = None if self.meta_train is None else self.meta_train[idx]
        
        # Optionally apply augmentation with batch-level and sample-level randomness
        if self.augmentation and random.random() < self.aug_prob:
            x = self._apply_augmentation(x)

        # If the dataset is unlabeled, return only the spectra (no labels or metadata)
        if y is None:
            return x  # Unlabeled data, return only spectra
        
        # If labeled, return spectra, labels, and metadata (if available)
        return x, y, meta

    def _apply_augmentation(self, x_tensor):
        """
        Apply one of the augmentation methods to the input spectra.
        We randomly select an augmentation method at batch-level
        and apply it with sample-level probability.
        """
        # Define the augmentation methods to choose from
        augmentation_methods = [self._add_noise, self._shift]
        
        # Randomly pick an augmentation method for the current batch
        aug_method = random.choice(augmentation_methods)
        
        # Apply the chosen augmentation method
        return aug_method(x_tensor)

    def _add_noise(self, x_tensor, std=0.01):
        """Add Gaussian noise to the input tensor."""
        noise = torch.randn_like(x_tensor) * std
        return x_tensor + noise

    def _shift(self, x_tensor):
        """Apply a custom shift to the input tensor based on provided parameters, ensuring positivity."""
        # Calculate the standard deviation of the spectra
        std = torch.std(x_tensor)  # Use the std of this particular sample

        # Generate random shift parameters with a very small beta and slope
        beta = (torch.rand(1) * 2 * self.betashift - self.betashift) * std  # Ensure the shift is minimal
        slope = (torch.rand(1) * 2 * self.slopeshift - self.slopeshift + 1)  # Small variation around 1

        # Calculate axis for shifting based on the size of the input tensor
        axis = torch.arange(x_tensor.shape[0], dtype=torch.float32) / float(x_tensor.shape[0])
        
        # Calculate the offset based on beta and slope
        offset = (slope * axis + beta - axis - slope / 2.0 + 0.5)
        multi = (torch.rand(1) * 2 * self.multishift - self.multishift + 1)

        # Apply the shift by multiplying and adding the offset
        augmented_x = multi * x_tensor + offset * std
        
        # Ensure the augmented signal remains positive (clamp to a minimum of 0)
        augmented_x = torch.clamp(augmented_x, min=0)

        return augmented_x



# ############### unlabeled from multzi csv files ##
# class MultiFileAugmentedCSVDataset(Dataset):
#     def __init__(self, file_paths, chunk_size=1000, augmentation=False, aug_prob=0.5,
#                  betashift=0.01, slopeshift=0.01, multishift=0.1, transform=None):
#         self.file_paths = file_paths
#         self.chunk_size = chunk_size
#         self.augmentation = augmentation
#         self.aug_prob = aug_prob
#         self.betashift = betashift
#         self.slopeshift = slopeshift
#         self.multishift = multishift
#         self.transform = transform
#         self.current_chunk = None
#         self.current_index = 0
#         self.file_index = 0
#         self.chunk_iter = None
#         self.load_next_file()  # Initialize with the first file

#     def __len__(self):
#         total_rows = 0
#         for file_path in self.file_paths:
#             total_rows += sum(1 for _ in open(file_path)) - 1  # Exclude header
#         return total_rows

#     def reset(self):
#         """Reset file and chunk iterators to start from the beginning."""
#         self.current_chunk = None
#         self.current_index = 0
#         self.file_index = 0
#         self.chunk_iter = None
#         self.load_next_file()

#     def load_next_file(self):
#         if self.file_index < len(self.file_paths):
#             self.chunk_iter = pd.read_csv(self.file_paths[self.file_index], chunksize=self.chunk_size, low_memory=False)
#             self.file_index += 1
#         else:
#             # Reset to the first file to allow reiteration
#             self.file_index = 0
#             self.load_next_file()

#     def load_next_chunk(self):
#         if self.chunk_iter is None:
#             return False

#         try:
#             self.current_chunk = next(self.chunk_iter)

#             # Clean and prepare the chunk
#             if 'Unnamed: 0' in self.current_chunk.columns:
#                 self.current_chunk = self.current_chunk.drop(['Unnamed: 0'], axis=1).reset_index(drop=True)

#             target_columns = [str(i) for i in range(400, 2451)]
#             existing_columns = [col for col in target_columns if col in self.current_chunk.columns]
#             if existing_columns:
#                 self.current_chunk = self.current_chunk[existing_columns]
#                 self.current_chunk.columns = self.current_chunk.columns.astype(int)
#             else:
#                 raise ValueError("None of the target columns found in the current chunk.")

#             self.current_index = 0

#             if self.transform:
#                 self.current_chunk = self.current_chunk.apply(self.transform, axis=1)

#             return True
#         except StopIteration:
#             self.load_next_file()
#             return self.load_next_chunk()

#     def __getitem__(self, idx):
#         while self.current_chunk is None or self.current_index >= len(self.current_chunk):
#             if not self.load_next_chunk():
#                 # Instead of raising an error, reset to allow reiteration
#                 self.reset()

#         row = self.current_chunk.iloc[self.current_index]
#         self.current_index += 1

#         x = torch.tensor(row.values, dtype=torch.float32)

#         if self.augmentation and random.random() < self.aug_prob:
#             x = self._apply_augmentation(x)

#         return x

#     def _apply_augmentation(self, x_tensor):
#         augmentation_methods = [self._add_noise] #self._shift
#         aug_method = random.choice(augmentation_methods)
#         return aug_method(x_tensor)

#     def _add_noise(self, x_tensor, std=0.01):
#         noise = torch.randn_like(x_tensor) * std
#         return x_tensor + noise

#     def _shift(self, x_tensor):
#         std = torch.std(x_tensor)
#         beta = (torch.rand(1) * 2 * self.betashift - self.betashift) * std
#         slope = (torch.rand(1) * 2 * self.slopeshift - self.slopeshift + 1)
#         axis = torch.arange(x_tensor.shape[0], dtype=torch.float32) / float(x_tensor.shape[0])
#         offset = (slope * axis + beta - axis - slope / 2.0 + 0.5)
#         multi = (torch.rand(1) * 2 * self.multishift - self.multishift + 1)
#         augmented_x = multi * x_tensor + offset * std
#         augmented_x = torch.clamp(augmented_x, min=0)
#         return augmented_x


# ############### unlabeled from multzi csv files ##
########### working with scaler##
### with scaling option ##

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class MultiFileAugmentedCSVDataset(Dataset):
    def __init__(self, file_paths, chunk_size=1000, augmentation=False, aug_prob=0.,
                 betashift=0.01, slopeshift=0.01, multishift=0.1, transform=None, scale=False):
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.betashift = betashift
        self.slopeshift = slopeshift
        self.multishift = multishift
        self.transform = transform
        self.current_chunk = None
        self.current_index = 0
        self.file_index = 0
        self.chunk_iter = None
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.load_next_file()  # Initialize with the first file
        self.scale = scale
        
        if(self.scale):
            self.fit_scaler()      # Fit scaler on the data

    def fit_scaler(self):
        # Load the full dataset or chunks to compute scaler statistics
        data = []
        for file_path in self.file_paths:
            chunk_iter = pd.read_csv(file_path, chunksize=self.chunk_size)
            for chunk in chunk_iter:
                spectra = chunk.drop(['Unnamed: 0'], axis=1).values  # Assuming all columns are part of the spectra
                data.append(spectra)
        full_data = np.vstack(data)
        self.scaler.fit(full_data) #[:,:-1] if needed!!! # Fit scaler on all spectra data

    def scale_data(self, spectra):
        return self.scaler.transform(spectra)

    def __len__(self):
        total_rows = 0
        for file_path in self.file_paths:
            total_rows += sum(1 for _ in open(file_path)) - 1  # Exclude header
        return total_rows

    def reset(self):
        """Reset file and chunk iterators to start from the beginning."""
        self.current_chunk = None
        self.current_index = 0
        self.file_index = 0
        self.chunk_iter = None
        self.load_next_file()

    def load_next_file(self):
        if self.file_index < len(self.file_paths):
            self.chunk_iter = pd.read_csv(self.file_paths[self.file_index], chunksize=self.chunk_size, low_memory=False)
            self.file_index += 1
        else:
            self.file_index = 0
            self.load_next_file()

    def load_next_chunk(self):
        if self.chunk_iter is None:
            return False

        try:
            self.current_chunk = next(self.chunk_iter)

            # Prepare the chunk
            if 'Unnamed: 0' in self.current_chunk.columns:
                self.current_chunk = self.current_chunk.drop(['Unnamed: 0'], axis=1).reset_index(drop=True)

            spectra = self.current_chunk.values
            # spectra = spectra[:,:-1]
            
            if(self.scale):
                spectra = self.scale_data(spectra)  # Apply scaling here
            self.current_chunk = pd.DataFrame(spectra)

            self.current_index = 0

            if self.transform:
                self.current_chunk = self.current_chunk.apply(self.transform, axis=1)

            return True
        except StopIteration:
            self.load_next_file()
            return self.load_next_chunk()

    # def __getitem__(self, idx):
    #     while self.current_chunk is None or self.current_index >= len(self.current_chunk):
    #         if not self.load_next_chunk():
    #             self.reset()

    #     row = self.current_chunk.iloc[self.current_index]
    #     self.current_index += 1

    #     x = torch.tensor(row.values, dtype=torch.float32)

    #     # if self.augmentation and random.random() < self.aug_prob:
    #     #     x = self._apply_augmentation(x)

    #     if self.augmentation : 
    #         if random.random() < self.aug_prob:
    #             x = self._apply_augmentation(x)

    #     return x

    def __getitem__(self, idx):
        while self.current_chunk is None or self.current_index >= len(self.current_chunk):
            if not self.load_next_chunk():
                self.reset()
    
        row = self.current_chunk.iloc[self.current_index]
        self.current_index += 1
    
        x = torch.tensor(row.values, dtype=torch.float32)
    
        if self.augmentation and self.aug_prob > 0:
            rand_val = random.random()
            if rand_val < self.aug_prob:
                # print(f"🚀 Applying augmentation at index {idx}, rand_val: {rand_val:.4f}, aug_prob: {self.aug_prob:.4f}")
                x = self._apply_augmentation(x)
            # else:
            #     print(f"❌ No augmentation at index {idx}, rand_val: {rand_val:.4f}, aug_prob: {self.aug_prob:.4f}")
    
        return x

    def _apply_augmentation(self, x_tensor):
        augmentation_methods = [self._add_noise]  # Add other augmentation methods if needed
        aug_method = random.choice(augmentation_methods)
        return aug_method(x_tensor)

    def _add_noise(self, x_tensor, std=0.01):
        noise = torch.randn_like(x_tensor) * std
        return x_tensor + noise