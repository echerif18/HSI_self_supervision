import sys
import os
import warnings
from src.utils_data import *


warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


directory_path = os.path.join(project_root, "Splits")
directory_path_Ds = os.path.join(project_root, "Datasets")

# # Install Git LFS (only needed once)
# !git lfs install

# # Clone the entire dataset repo
# !git clone https://huggingface.co/datasets/avatar5/DiverseSpecLib {directory_path_Ds}


import subprocess

# Run Git LFS install
subprocess.run(["git", "lfs", "install"], check=True)

# Clone the repo into your desired directory
subprocess.run([
    "git", "clone",
    "https://huggingface.co/datasets/avatar5/DiverseSpecLib",
    directory_path_Ds
], check=True)


num_splits = 20  # Number of output splits
chunk_size = 5000  # Tune based on your memory constraints

os.makedirs(directory_path, exist_ok=True)  # Create the output folder if it doesn't exist
split_csvs_with_proportions_sequential(directory_path_Ds, directory_path, num_splits, chunk_size)