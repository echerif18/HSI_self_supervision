### util modul ########

"""
General settings.
"""
import platform
import random
from copy import deepcopy
from enum import Enum
from scipy.stats import rv_continuous
import math
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import torch


class MixtureModel(rv_continuous):
    """Creates a combination distribution of multiple scipy.stats model distributions."""
    def __init__(self, submodels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels

    def _pdf(self, x, **kwargs):
        pdf = self.submodels[0].pdf(x)
        for submodel in self.submodels[1:]:
            pdf += submodel.pdf(x)
        pdf /= len(self.submodels)
        return pdf

    def rvs(self, size):
        """Random variates of the mixture model."""
        submodel_choices = np.random.randint(len(self.submodels), size=size)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs


def seed_all(seed=None):
    """Seed every type of random used by the SRGAN."""
    random.seed(seed)
    np.random.seed(seed)
    if seed is None:
        seed = int(time.time())
    torch.manual_seed(seed)

def norm_squared(tensor, axis=1):
    """Calculates the norm squared along an axis. The default axis is 1 (the feature axis), with 0 being the batch."""
    return tensor.pow(2).sum(dim=axis)

def square_mean(tensor):
    """Calculates the element-wise square, then the mean of a tensor."""
    return tensor.pow(2).mean()

def abs_plus_one_log_mean_neg(tensor):
    """Takes the absolute value, then adds 1, then takes the log, then mean, then negates."""
    return tensor.abs().add(1).log().mean().neg()


def abs_mean(tensor):
    """Takes the absolute value, then mean."""
    return tensor.abs().mean()

def abs_plus_one_sqrt_mean_neg(tensor):
    """Takes the absolute value, then adds 1, then takes the log, then mean, then negates."""
    return tensor.abs().add(1).sqrt().mean().neg()

class Settings:
    """Represents the settings for a given run of SRGAN."""
    def __init__(self):
        self.trial_name = 'base'
        self.steps_to_run = 200000
        self.temporary_directory = 'temporary'
        self.logs_directory = 'logs'
        self.batch_size = 256
        self.augmentation = False
        self.summary_step_period = 2000
        self.labeled_dataset_size = 50
        self.unlabeled_dataset_size = 50000
        self.validation_dataset_size = 1000
        self.learning_rate = 0.0001 #1e-4
        self.weight_decay = 0

        self.labeled_loss_multiplier = 1e0
        self.matching_loss_multiplier = 1e0
        self.contrasting_loss_multiplier = 1e0
        self.srgan_loss_multiplier = 1e0
        self.dggan_loss_multiplier = 1e1
        self.gradient_penalty_on = True
        self.gradient_penalty_multiplier = 1e1
        self.mean_offset = 0
        self.labeled_loss_order = 2
        self.generator_training_step_period = 1
        self.labeled_dataset_seed = 0
        self.normalize_fake_loss = False
        self.normalize_feature_norm = False
        self.contrasting_distance_function = abs_plus_one_sqrt_mean_neg
        self.matching_distance_function = abs_mean

        self.load_model_path = None
        self.should_save_models = True
        self.skip_completed_experiment = True
        self.number_of_data_workers = 4
        self.pin_memory = True
        self.continue_from_previous_trial = False
        self.continue_existing_experiments = False
        self.save_step_period = None

        # Coefficient application only.
        self.hidden_size = 10

        # Crowd application only.
        self.crowd_dataset = 'World Expo'
        self.number_of_cameras = 5  # World Expo data only
        self.number_of_images_per_camera = 5  # World Expo data only
        self.test_summary_size = None
        self.test_sliding_window_size = 128
        self.image_patch_size = 224
        self.label_patch_size = 224
        self.map_multiplier = 1e-6
        self.map_directory_name = 'i1nn_maps'

        # SGAN models only.
        self.number_of_bins = 10

    def local_setup(self):
        """Code to override some settings when debugging on the local (low power) machine."""
        if 'Carbon' in platform.node():
            self.labeled_dataset_seed = 0
            self.batch_size = min(10, self.batch_size)
            self.summary_step_period = 10
            self.labeled_dataset_size = 10
            self.unlabeled_dataset_size = 10
            self.validation_dataset_size = 10
            self.skip_completed_experiment = False
            self.number_of_data_workers = 0


def convert_to_settings_list(settings, shuffle=True):
    """
    Creates permutations of settings for any setting that is a list.
    (e.g. if `learning_rate = [1e-4, 1e-5]` and `batch_size = [10, 100]`, a list of 4 settings objects will return)
    This function is black magic. Beware.
    """
    settings_list = [settings]
    next_settings_list = []
    any_contains_list = True
    while any_contains_list:
        any_contains_list = False
        for settings in settings_list:
            contains_list = False
            for attribute_name, attribute_value in vars(settings).items():
                if isinstance(attribute_value, (list, tuple)):
                    for value in attribute_value:
                        settings_copy = deepcopy(settings)
                        setattr(settings_copy, attribute_name, value)
                        next_settings_list.append(settings_copy)
                    contains_list = True
                    any_contains_list = True
                    break
            if not contains_list:
                next_settings_list.append(settings)
        settings_list = next_settings_list
        next_settings_list = []
    if shuffle:
        random.seed()
        random.shuffle(settings_list)
    return settings_list


def r_squared(y_true, y_pred):
    # Calculate the mean of the true values
    bool_finite = torch.isfinite(y_true)
    y_mean = torch.mean(y_true[bool_finite])

    # Calculate the total sum of squares
    total_sum_of_squares = torch.sum((y_true[bool_finite] - y_mean)**2)

    # Calculate the residual sum of squares
    residual_sum_of_squares = torch.sum((y_true[bool_finite] - y_pred[bool_finite])**2)

    # Calculate R-squared
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)

    return torch.mean(r2)
    
    
def unit_vector(vector):
    """Gets the unit vector version of a vector."""
    return vector.div(vector.norm() + 1e-10)


def angle_between(vector0, vector1):
    """Calculates the angle between two vectors."""
    unit_vector0 = unit_vector(vector0)
    unit_vector1 = unit_vector(vector1)
    epsilon = 1e-6
    return unit_vector0.dot(unit_vector1).clamp(-1.0 + epsilon, 1.0 - epsilon).acos()


def square(tensor):
    """Squares the tensor value."""
    return tensor.pow(2)


def feature_distance_loss_unmeaned(base_features, other_features, distance_function=square):
    """Calculate the loss based on the distance between feature vectors."""
    base_mean_features = base_features.mean(0, keepdim=True)
    distance_vector = distance_function(base_mean_features - other_features)
    return distance_vector.mean()


def feature_distance_loss_both_unmeaned(base_features, other_features, distance_function=norm_squared):
    """Calculate the loss based on the distance between feature vectors."""
    distance_vector = distance_function(base_features - other_features)
    return distance_vector.mean()


def feature_angle_loss(base_features, other_features, target=0, summary_writer=None):
    """Calculate the loss based on the angle between feature vectors."""
    angle = angle_between(base_features.mean(0), other_features.mean(0))
    if summary_writer:
        summary_writer.add_scalar('Feature Vector/Angle', angle.item(), )
    return (angle - target).abs().pow(2)


def feature_corrcoef(x):
    """Calculate the feature vector's correlation coefficients."""
    transposed_x = x.transpose(0, 1)
    return corrcoef(transposed_x)


def corrcoef(x):
    """Calculate the correlation coefficients."""
    mean_x = x.mean(1, keepdim=True)
    xm = x.sub(mean_x)
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c


def feature_covariance_loss(base_features, other_features):
    """Calculate the loss between feature vector correlation coefficient distances."""
    base_corrcoef = feature_corrcoef(base_features)
    other_corrcoef = feature_corrcoef(other_features)
    return (base_corrcoef - other_corrcoef).abs().sum()


def disable_batch_norm_updates(module):
    """Turns off updating of batch norm statistics."""
    # noinspection PyProtectedMember
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def enable_batch_norm_updates(module):
    """Turns on updating of batch norm statistics."""
    # noinspection PyProtectedMember
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.train()


def para_sampling(rtm_paras, num_samples=100):
    pi_tensor = torch.tensor(math.pi)
    
    # run uniform sampling for learnable parameters
    para_dict = {}
    for para_name in rtm_paras.keys():
        min = rtm_paras[para_name]['min']
        max = rtm_paras[para_name]['max']
        para_dict[para_name] = torch.rand(num_samples) * (max - min) + min
    SD = 500
    para_dict['cd'] = torch.sqrt(
        (para_dict['fc']*10000)/(pi_tensor*SD))*2 #pi_tensor torch.pi
    para_dict['h'] = torch.exp(
        2.117 + 0.507*torch.log(para_dict['cd']))

    return para_dict



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

Traits = ['LMA_g_m2', 'N_area_mg_cm2', 'LAI_m2_m2', 'C_area_mg_cm2', 'Chl_area_ug_cm2', 'EWT_mg_cm2',
          'Car_area_ug_cm2', 'Anth_area_ug_cm2', 'Protein_g_m2']
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


def read_db(file, sp=False, encoding=None):
    db = pd.read_csv(file, encoding=encoding, low_memory=False)
    db.drop(['Unnamed: 0'], axis=1, inplace=True)
    if (sp):
        features = db.loc[:, "400":]
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


from sklearn.preprocessing import PowerTransformer
from torchvision import transforms

def data_prep(minl, gap_fil, Traits, i = len(Traits)-1, w_train=None, multi=False):

    ##########Testing/validation data preparation (only for the last added trait)#######
    if (multi):
        train_x = gap_fil.loc[:, minl:]
        train_y = gap_fil.loc[train_x.index, Traits[:i + 1]]
    else:
        train_x = gap_fil.loc[gap_fil[gap_fil[Traits[i]].notnull()].index, minl:]
        train_y = gap_fil.loc[train_x.index, Traits[i:i + 1]]

    if(w_train is not None):
        samp_w_tr = samp_w(w_train, train_x)  # >>>>>>samples weights calculation
        return train_x, train_y, samp_w_tr
    else:
        return train_x, train_y

def dataaugment(x, betashift=0.05, slopeshift=0.05, multishift=0.05, kind='shift'):
    
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
        signal = torch.tensor(sample_sp.values)#.to(device)

        # Define the standard deviation of the Gaussian noise
        std_dev = 0.03 #torch.std(torch.tensor(features.values),0) # Adjust this value according to the desired amount of noise
        
        # Generate Gaussian noise with the same shape as the signal
        noise = (torch.randn_like(signal) * std_dev)#.to(device)
        
        # Add the noise to the signal
        return signal + noise


def data_augmentation(x, y):
    # x = x.to(device).view(x.shape[0], x.shape[-1])
    # y = y.to(device).view(y.shape[0], y.shape[-1])
    
    data_std = torch.std(x, 0)
    
    if torch.rand(1) < 0.15:
        x = dataaugment(x, betashift=data_std, slopeshift=data_std, multishift=data_std, kind='shift')
    if torch.rand(1) < 0.15:
        x = dataaugment(x, kind='noise')
    return x, y


def save_scaler(train_y, save=False, dir_n=None, k=None):
    scaler = PowerTransformer(method='box-cox').fit(np.array(train_y))
    if save:
        if not os.path.exists(dir_n):
            os.mkdir(dir_n)
        dump(scaler, open(dir_n + '/scaler_{}.pkl'.format(k), 'wb')) 
    return scaler