# Requirements

* python 3.6
* Python packages: pytorch

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