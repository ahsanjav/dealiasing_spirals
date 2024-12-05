#!/bin/bash
iter=200
epochs=200
lr=0.001

cd /home/javeda2/code/dealiasing_spirals ; python  train_model.py --train_files /data_drive/_0_retro_cines_kspace_spiral_walsh_itt_x.h5 /data_drive/_1_retro_cines_kspace_spiral_walsh_itt_x.h5 /data_drive/_2_retro_cines_kspace_spiral_walsh_itt_x.h5 /data_drive/_3_retro_cines_kspace_spiral_walsh_itt_x.h5 --batch_size 128 --epochs $iter --exp_name fastvdnet_unet_mag_exp_final_r6_complex_ssim_itt --cuda 1 --num_workers 6 --usample 6.0 --time 5 --complex_i 1 --no_in_channel 2  --use_non_appended_keys 0 --ssim_only 1 --lr $lr --epochs $epochs
