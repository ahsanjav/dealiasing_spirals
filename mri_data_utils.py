"""
Data utilities for MRI data.
Provides the torch dataset class for traind and test and functions to load from multiple h5files
"""

import sys
import logging

from tqdm import tqdm
import numpy as np
from pathlib import Path
from colorama import Fore, Style
import h5py
Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from trainer import get_bar_format

# -------------------------------------------------------------------------------------------------
def load_images_from_h5file_cart_recon(h5file, keys, max_load=100000):
    """
    Load images from h5 file objects
    @args:
        - h5file (h5File list): list of h5files to load images from
        - keys (key list list): list of list of keys. One for each h5file
        
    @outputs:
        - images : list of image and gmap pairs as a list
    """
    images = []

    num_loaded = 0
    for i in range(len(h5file)):
        
        if max_load<=0:
            logging.info(f"{h5file[i]}, data will not be pre-read ...")
        
        with tqdm(total=len(keys[i]), bar_format=get_bar_format()) as pbar:
            for n, key in enumerate(keys[i]):
                if num_loaded < max_load:
                    with h5py.File(h5file[i]) as f:
# The csm will not be found its stored in a tricky way to save memory need to write the logic for csm access based on the data gen updates.
                        images.append([np.array(f[key+"/input"]), np.array(f[key[:-4]+"/1.0"+"/output"]),
                                   np.array(f[key[:-4]+"/1.0"+"/csm"]),i])
                    num_loaded += 1
                else:
                    #images.append([key+"/in_image", key+"/out_image", i])
                    with h5py.File(h5file[i]) as f:
                        if len(f[key[:-4]].keys())>1:    
                            images.append([key+"/input", key+"/output" ,key+"/csm", i])
                    
                if n>0 and n%100 == 0:
                    pbar.update(100)
                    pbar.set_description_str(f"{h5file}, {n} in {len(keys[i])}, total {len(images)}")

    return images



# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    pass