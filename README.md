# Execution instructions

* Download the code

```
$ git clone https://github.com/aitorshuffle/ntire2018_adv_rgb2hs.git
$ cd ntire2018_adv_rgb2hs
```

* Place the input RGB image in the ```datasets/icvl_ntire2018/NTIRE2018_Test_Clean/``` or ```datasets/icvl_ntire2018/NTIRE2018_Test_RealWorld``` directory

* Run the rgb to hyperspectral conversion:
```
ntire2018_adv_rgb2hs$ python test.py --dataroot ./datasets/icvl_ntire2018 --name 29 --challenge Clean --phase Test --output_nc 31 --model pix2pix --which_model_netG pbdl2017PrunedTo3 --which_direction AtoB --dataset_mode rgb2hs --norm batch --gpu_ids 0
```