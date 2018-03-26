# Requirements

* python 3.6
* Python packages: pytorch (torch, torchvision), skimage, spectral, colour, numpy, h5py, PIL, dominate, scipy, hdf5storage, tqdm, joblib 

# Execution instructions

* Download the code

```
$ git clone https://github.com/aitorshuffle/ntire2018_adv_rgb2hs.git
$ cd ntire2018_adv_rgb2hs
```

* Place the input RGB images to be processed in the ```datasets/icvl_ntire2018/NTIRE2018_Test_Clean/``` and/or ```datasets/icvl_ntire2018/NTIRE2018_Test_RealWorld``` directory

* Run the rgb to hyperspectral conversion:

	* Make the execution scripts executable: 
	```
	ntire2018_adv_rgb2hs$ chmod 777 ./scripts/test_ntire2018_adv_rgb2hs_Clean.sh
	ntire2018_adv_rgb2hs$ chmod 777 ./scripts/test_ntire2018_adv_rgb2hs_RealWorld.sh
	```
	
	* Run the execution script for each track: 
        * Clean track:
        ```
        ntire2018_adv_rgb2hs$ ./scripts/test_ntire2018_adv_rgb2hs_clean.sh 
        ```

        * RealWorld track:
        ```
        ntire2018_adv_rgb2hs$ ./scripts/test_ntire2018_adv_rgb2hs_clean.sh
        ```
* Output results will be generated in:
    * Clean track: ```results/29```
    * RealWorld track: ```results/34```
    Each of these contain an images directory, with the predicted hyperspectral mat file in the required format and one RGB image triplet per test image:
    	 * ```TEST_IMG_NAME.mat```: predicted hyperspectral mat file 
         * ```TEST_IMG_NAME_real_A.png```: input RGB image
         * ```TEST_IMG_NAME_fake_B.png```: predicted hyperspectral image rendered as sRGB
         * ```TEST_IMG_NAME_real_B.png```: Ground truth hyperspectral image rendered as sRGB. Only makes sense for validation. At test time
	There will also be a ```index.html``` web page rendering all the mentioned rgb triplets.          
    