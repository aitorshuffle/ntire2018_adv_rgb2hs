#!/usr/bin/env bash

python test.py --dataroot ./datasets/icvl_ntire2018 --name 29 --challenge Clean --phase Test --output_nc 31 --model pix2pix --which_model_netG pbdl2017PrunedTo3 --which_direction AtoB --dataset_mode rgb2hs --norm batch --gpu_ids 0