from utils_data import *
from transformation_utils import *
from utils_all import *
# 
from GAN.SrGAN_RTM_trainer import *
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

from datetime import datetime

import glob
import math
import gc

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

seed = 155 # 4 155 140 #random.randint(0, 500)
seed_all(seed=seed) ###155

path_save = '/home/mila/e/eya.cherif/scratch/Gans_models/'
project = 'Gan_wandb_test'

lr = 1e-4
n_epochs = 500
ls_tr = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
batch_size = 128  # This should match the batch size for unlabeled data


percentage_tr = 1

######## Data ########
directory_path = '/home/mila/e/eya.cherif/scratch/Datasets/csv_files'
file_paths = glob.glob(os.path.join(directory_path, "*.csv"))

file_paths = file_paths[:int(percentage_tr*len(file_paths))]

################ Lbeled ###############
# path_data_lb = '/home/mila/e/eya.cherif/Gan_project_test/49_all_lb_prosailPro.csv'
path_data_lb = '/home/mila/e/eya.cherif/Gan_project_test/50SHIFT_all_lb_prosailPro.csv'

db_lb_all = pd.read_csv(path_data_lb, low_memory=False).drop(['Unnamed: 0'], axis=1)   


start = 1
end = 20 #50

# for gp in range(start,end, 10):
# for i in range(5):
for gp in [2]: #2, 32, 50, 47, 6, 38
    # gp = random.randint(1, 50)
    run = 'Gan_NoRTM_{}_gp{}UNlabels_{}'.format(formatted_datetime, gp, seed)
    checkpoint_dir = os.path.join(path_save, "checkpoints_{}".format(run)) #'./checkpoints'
    
    # Optional: Summarize GPU memory usage
    print(torch.cuda.memory_summary())
    print(gp)

    ### external
    db_lb_all = pd.read_csv(path_data_lb, low_memory=False).drop(['Unnamed: 0'], axis=1)   
    groups = db_lb_all.groupby('dataset')
    
    # val_ext_idx = list(np.concatenate([groups.get_group(i).index for i in range(gp, gp+10)]))
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
    

    # db_tr = balanceData(pd.concat([X_train, y_train], axis=1), meta_train, ls_tr, random_state=300,percentage=1)
    # X_train = db_tr.loc[:,400:2450]
    # y_train = db_tr.loc[:,'cab':]
    # meta_train = db_tr.iloc[:,:8]    
    
    
    ######### scaler ######
    ### transformation in the model 
    
    scaler_list = None
    scaler_model = save_scaler(y_train, standardize=True, scale=True, save=True, dir_n=checkpoint_dir, k='all_{}'.format(100*percentage_tr))
    
    
    # Create the dataset
    train_dataset = SpectraDataset(X_train, y_train, meta_train, augmentation=True, aug_prob=0.8)
    # Define DataLoader with the custom collate function for fair upsampling
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = SpectraDataset(X_train=X_val, y_train=y_val, meta_train=meta_val, augmentation=False)
    # Create DataLoader for the test dataset
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create the dataset
    untrain_dataset = MultiFileAugmentedCSVDataset(file_paths, chunk_size=1000, augmentation=True, aug_prob=0.5, scale=False) ## No scaling of specra!!!
    unlabeled_dataset_loader = DataLoader(untrain_dataset, batch_size=batch_size, 
                            shuffle=True
                           )
    
    
    ################### Model 
    
    settings_dict = {
        'checkpoint_dir': checkpoint_dir,
        'train_loader': train_dataset_loader,
        'valid_loader': valid_loader,
        'unlabeled_loader': unlabeled_dataset_loader,
        
        'scaler_model': scaler_model,
        'n_lb': y_train.shape[1], #8,
        'input_shape': 1720, #1720, 500
        'latent_dim': 100,
        'learning_rate': lr,
        'weight_decay': 1e-4,
        
        'n_epochs': n_epochs,
        
        'rtm_D': False,
        'rtm_G': False,
        
        'lambda_fk': 1.0,
        'lambda_un': 10.0,
        
        'labeled_loss_multiplier': 1.0,
        'matching_loss_multiplier': 1.0,
        'contrasting_loss_multiplier': 1.0,
        
        'gradient_penalty_on': True,
        'gradient_penalty_multiplier': 10.0,
        'srgan_loss_multiplier': 1.0,
        
        'early_stop': True,
        'early_stopping': None,
        'patience': 10,
        'logger': None,
        'log_epoch': 10,
        
        'mean_offset': 0,
        'normalize_fake_loss': False,
        'normalize_feature_norm': False,
        
        'contrasting_distance_function': nn.CosineEmbeddingLoss(),
        'matching_distance_function': nn.CosineEmbeddingLoss(),
        'labeled_loss_function': HuberCustomLoss(threshold=1.0),
        
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    ### with wandb #####
    wandb.init(
        # Set the project where this run will be logged
        project=project,
        # We pass a run name (otherwise it’ll be randomly assigned)
        name=f"experiment_{run}",
        # Track hyperparameters and run metadata
        config=settings_dict,
        dir= checkpoint_dir
        )
    
    #>>> Model input 1720 !!!
    settings = Settings() ## set the settings first
    # Update settings using the dictionary
    settings.update_from_dict(settings_dict)
    
    
    test = SrGAN_RTM(settings) #SrGAN SrGAN_RTM Experiment
    test.settings.logger = wandb
    
    test.dataset_setup()
    # test.model_setup(test.settings.latent_dim, test.settings.input_shape, test.settings.n_lb)
    test.model_setup()
    # test.prepare_optimizers()
    test.prepare_optimizers(test.settings.n_epochs) ##
    test.gpu_mode()
    test.train_mode()
    test.transformation_setup()
    test.early_stopping_setup()
    
    test.train_loop(n_epochs=test.settings.n_epochs) #self, start_epoch=1, n_epochs=200
    
    test.settings.logger.finish()
    
    #########
    preds = torch.empty(0,8).to(device)
    ori = torch.empty(0,8).to(device)
    
    val_train_iterator = iter(test.valid_loader) 
    
    with torch.no_grad():
        for val_examples, val_labels, _ in val_train_iterator:
            val_examples = val_examples.unsqueeze(dim=1)[:,:,:-1].float().to(device)
            
            val_labels = val_labels.float().to(device)
        
            test.eval_mode()
    
            if(test.transformation_layer_inv is not None): 
                preds_D = test.transformation_layer_inv(test.D(val_examples)[0]) ### shoud keep the sam eorder of labels !!!
                
            elif(test.settings.scaler_list is not None):
                preds_D = torch.tensor(test.settings.scaler_list.inverse_transform(test.D(val_examples)[0].cpu().detach().numpy()), dtype=torch.float32)#.requires_grad_(True)
            else:
                preds_D = test.D(val_examples)[0]
            
            ori = torch.cat((ori.data, val_labels.data), dim=0)
            preds = torch.cat((preds.data, preds_D.data), dim=0)
    
    ori_lb = pd.DataFrame(ori.cpu(), columns=ls_tr[:])    
    df_tr_val = pd.DataFrame(preds.cpu(), columns=ls_tr[:])

    ori_lb.to_csv(os.path.join(checkpoint_dir, "Obs_CV{}".format(gp)))
    df_tr_val.to_csv(os.path.join(checkpoint_dir, "Preds_CV{}".format(gp)))
    
    val_mertics = eval_metrics(ori_lb, df_tr_val)
    val_mertics.to_csv(os.path.join(checkpoint_dir, "ValidationMetrics_CV{}".format(gp)))

    # Clean up after training
    del test
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Completed training model {percentage_tr}. GPU memory cleared.\n")