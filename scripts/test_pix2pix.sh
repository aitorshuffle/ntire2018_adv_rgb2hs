# original
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --dataset_mode aligned --norm batch

# rgb2hs
python test.py --dataroot ./datasets/icvl_ntire2018 --name 2 --challenge Clean --phase Validate --output_nc 31 --model pix2pix --which_model_netG unet_256_noBN --which_direction AtoB --dataset_mode rgb2hs --norm batch --gpu_ids 1
#--use_envi True --overwrite_envi True
#--phase Validate Test
