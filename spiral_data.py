"""
Data utilities for MRI data.
Provides the torch dataset class for traind and test and functions to load from multiple h5files
"""
# -------------------------------------------------------------------------------------------------
# Spiral train dataset class
# -------------------------------------------------------------------------------------------------
import os
import sys
import h5py
import torch
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path
from colorama import Fore, Style

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from mri_data_utils import *

class MRIReconSpiralDatasetTrain():
    def __init__(self, h5file, keys, data_type, conf,max_load=10000,
                    time_cutout=30, 
                    use_complex=True, 
                    matrix_size_adjust_ratio=[0.5, 0.75, 1.0, 1.25, 1.5],
                    kspace_filter_sigma=[0.8, 1.0, 1.5, 2.0, 2.25],
                    kspace_filter_T_sigma=[0.25, 0.5, 0.65, 0.85, 1.0, 1.5, 2.0, 2.25],
                    cutout_shuffle_time=True,
                    num_patches_cutout=8,
                    patches_shuffle=False,
                    ):
        """
        Initilize the denoising dataset
        Loads and store all images and gmaps
        h5files should have the following strucutre
        file --> <key> --> "image"+"gmap"
        @args:
            - h5file (h5File list): list of h5files to load images from
            - keys (key list list): list of list of keys. One for each h5file
            - data_type ("2d"|"2dt"|"3d"): types of mri data
            - max_load (int): number of loaded samples when instantiating the dataset
            - time_cutout (int): cutout size in time dimension
            - cutout_shape (int list): 2 values for patch cutout shape
            - use_complex (bool): whether to return complex image
        """
        assert data_type=="2d" or data_type=="2dt" or data_type=="3d",\
            f"Data type not implemented: {data_type}"
        self.data_type = data_type
        self.config = conf
        self.h5file = h5file
        self.keys = keys
        self.max_load = max_load
        
        self.time_cutout = time_cutout
        if self.data_type=="2d": self.time_cutout = 1
        
        self.use_complex = use_complex
        

    
        self.matrix_size_adjust_ratio = matrix_size_adjust_ratio
        self.kspace_filter_sigma = kspace_filter_sigma
        self.kspace_filter_T_sigma = kspace_filter_T_sigma
        self.cutout_shuffle_time = cutout_shuffle_time

        self.num_patches_cutout = num_patches_cutout
        self.patches_shuffle = patches_shuffle

        self.images = load_images_from_h5file_cart_recon(h5file, keys, max_load=self.max_load)
        self.rng = np.random.Generator(np.random.PCG64(85442365))

    def load_one_sample(self, i):
        """
        Loads one sample from the saved images
        @args:
            - i (int): index of the file to load
        @rets:
            input_ims (corrupted/undersampled) and output_ims (reference)
        """
        # get the image
        in_image    = self.images[i][0]
        out_image   = self.images[i][1]
        
        iR = self.config.usample
        
        if not isinstance(in_image, np.ndarray):
            ind = self.images[i][3]
            key_image       = self.images[i][0]
            key_image_out   = self.images[i][1]
            
            key_image_out_parts = key_image_out.split('/')
            key_image_out = key_image_out_parts[0]+'/1.0/'+key_image_out_parts[-1]
            data_set_failed = 0
            try:
                in_image    = np.array(self.h5file[ind][key_image])
                out_image   = np.array(self.h5file[ind][key_image_out])
            except:
                print('Dataset not working')
                data_set_failed = 1
                pass
        if not data_set_failed:    
            if in_image.ndim == 2: 
                in_image = in_image[np.newaxis,:,:]
                out_image = out_image[np.newaxis,:,:]
            
            
            out_image   = out_image.astype(np.complex64)
            in_image    = in_image.astype(np.complex64)
        # in_image, out_image = self.random_flip(in_image, out_image)
            
    #         T, CHA, RO, E1 = in_image.shape
    #         s_x, s_y, s_t = self.get_cutout_range(in_image)

    #         if(RO>=self.cutout_shape[0] and E1>=self.cutout_shape[1]):
    # #                if(self.use_complex): # Commented out because we will always use complex for recon - I love complex numbers :D
    #             patch_data_in  = self.do_cutout(in_image, s_x, s_y, s_t)
    #             patch_data_out = self.do_cutout(out_image, s_x, s_y, s_t)
    #             cutout_in  = np.concatenate((patch_data_in.real, patch_data_in.imag),axis=1)
    #             cutout_out = np.concatenate((patch_data_out.real, patch_data_out.imag),axis=1)
                
    #             t_indexes = np.arange(cutout_in.shape[0])
    #             np.random.shuffle(t_indexes)

    #             np.take(cutout_in, t_indexes, axis=0, out=cutout_in)
    #             np.take(cutout_out, t_indexes, axis=0, out=cutout_out)
            
            #input_ims  = torch.from_numpy(cutout_in.astype(np.float32))
            #output_ims = torch.from_numpy(cutout_out.astype(np.float32))
            in_image = in_image[:,:,:,0]
            out_image = out_image[:,:,:,0]
            in_image = in_image.transpose(2,1,0)
            out_image = out_image.transpose(2,1,0)
            if(self.config.useLastImage):
                out_image = np.tile(out_image[-1,:,:],[in_image.shape[0],1,1])

            in_image  = in_image[-1*self.config.time:,:,:]                     
            # if(in_image.ndim <5):
            #     in_image  = in_image[np.newaxis,:,:,:]
            #     if(out_image.ndim<3):
            #         out_image = out_image[np.newaxis,np.newaxis,:,:]
            #     else:    
            #         out_image = out_image[np.newaxis,:,:,:]
                    
            if(self.config.normalize_images):
                
                normalize = lambda x: x/np.tile(np.max(np.abs(x),axis=(1,2)),[x.shape[1],1,1]).transpose(2,0,1)
                
                
                #in_image = in_image / np.amax(np.abs(in_image),axis=1)
                if(not (np.count_nonzero(np.max(np.abs(in_image),axis=(1,2))==0)>0 and np.count_nonzero(np.max(np.abs(out_image),axis=(1,2))==0))>0):
                    in_image = normalize(in_image)
                    out_image = normalize(out_image)
                else:
                    pass
            else:
                in_image = in_image/1000.0
                out_image = out_image/1000.0

            if(self.config.model_type ==  'FASTVDNET'):
                in_image  = np.abs(in_image)
                out_image = np.abs(out_image)

            if(self.config.complex_i): 
                in_image  = np.concatenate((in_image.real, in_image.imag),axis=0)/1.0#.transpose([1,0,2,3])
                out_image = np.concatenate((out_image.real, out_image.imag),axis=0)/1.0#.transpose([1,0,2,3])
            else:
                in_image = abs(in_image)
                out_image = abs(out_image)
            
            #if(out_image.shape != in_image.shape):
            #    out_image = np.tile(out_image[:,-1,:,:],[1,in_image.shape[1],1,1])

            if(self.config.model_type ==  'FASTVDNET'):
                out_image = out_image[-1,:,:]
                out_image = out_image[np.newaxis,:,:]

            

                #out_image = out_image / np.max(np.abs(out_image.ravel()))
                
            out_image[np.isnan(out_image)]=0
            in_image [np.isnan(in_image)]=0

                


            input_ims   = torch.from_numpy(in_image.astype(np.float32))
            output_ims  = torch.from_numpy(out_image.astype(np.float32))

                
            if( not self.config.model_type ==  'FASTVDNET' and out_image.shape != in_image.shape):
                pass
            else:
                return input_ims, output_ims

    def random_flip(self, data, data2):
            """
            Randomly flips the input image and gmap
            """
            flip1 = np.random.randint(0, 2) > 0
            flip2 = np.random.randint(0, 2) > 0
            
            def flip(image):
                if flip1:
                    image = image[:,:,::-1,:].copy()
                if flip2:
                    image = image[:,:,:,::-1].copy()
                return image

            return flip(data), flip(data2)
        
    def get_stat(self):
        stat = load_images_for_statistics_recon(self.h5file, self.keys)
        return stat

    def __len__(self):
        """
        Length of dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Gets the item given index
        """
        return self.load_one_sample(index)
