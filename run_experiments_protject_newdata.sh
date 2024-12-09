#!/bin/bash
export OMP_NUM_THREADS=24
epochs=200
lr=0.0001
batch_size=96
num_workers=48
num_gpus=4
entity="gadgetron"
data_dir="/project/labcampbell/Javed/AI_data/new_samp_itt/"
data_dir="/data_drive/"

#cd /home/javeda2/code/dealiasing_spirals ; python  train_model.py --train_files ${data_dir}_0_retro_cines_kspace_spiral_walsh_itt.h5  ${data_dir}_1_retro_cines_kspace_spiral_walsh_itt.h5 ${data_dir}_2_retro_cines_kspace_spiral_walsh_itt.h5 ${data_dir}_3_retro_cines_kspace_spiral_walsh_itt.h5 --batch_size $batch_size --exp_name fastvdnet_unet_mag_exp_final_r6_complex_ssim_itt_GA1 --cuda 1 --num_workers $num_workers --usample 6.0 --time 5 --complex_i 1 --no_in_channel 2  --use_non_appended_keys 0 --ssim_only 1 --lr $lr --epochs $epochs --gpus $num_gpus --entity $entity

cd /home/javeda2/code/dealiasing_spirals ; python  train_model.py --train_files ${data_dir}_0_retro_cines_kspace_spiral_walsh_itt_nonGA.h5  ${data_dir}_1_retro_cines_kspace_spiral_walsh_itt_nonGA.h5 ${data_dir}_2_retro_cines_kspace_spiral_walsh_itt_nonGA.h5 ${data_dir}_3_retro_cines_kspace_spiral_walsh_itt_nonGA.h5 --batch_size $batch_size --exp_name fastvdnet_unet_mag_exp_final_r6_complex_ssim_itt_nonGA --cuda 1 --num_workers $num_workers --usample 6.0 --time 5 --complex_i 1 --no_in_channel 2  --use_non_appended_keys 0 --ssim_only 1 --lr $lr --epochs $epochs --gpus $num_gpus --entity $entity
#/data_drive/_1_retro_cines_kspace_spiral_walsh_itt_nonGA.h5 /data_drive/_2_retro_cines_kspace_spiral_walsh_itt_nonGA.h5 /data_drive/_3_retro_cines_kspace_spiral_walsh_itt_nonGA.h5
