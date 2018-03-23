# Evaluation script for the NTIRE 2018 Spectral Reconstruction Challenge
#
# * Provide input and output directories as arguments
# * Validation files should be found in the '/ref' subdirectory of the input dir
# * Input validation files are expected in the v7.3 .mat format


import h5py as h5py
import numpy as np
import sys
import os


MRAEs = {}
RMSEs = {}


def get_ref_from_file(filename):
    matfile = h5py.File(filename, 'r')
    mat={}
    for k, v in matfile.items():
        mat[k] = np.array(v)
    return mat['rad']


#input and output directories given as arguments
[_, input_dir, output_dir] = sys.argv

validation_files = os.listdir(input_dir +'/ref')

for f in validation_files:
    # Read ground truth data
    if not(os.path.splitext(f)[1] in '.mat'):
        print('skipping '+f)
        continue
    gt = get_ref_from_file(input_dir + '/ref/' + f)
    # Read user submission
    rc = get_ref_from_file(input_dir + '/res/' + f)

    # compute MRAE
    diff = gt-rc
    abs_diff = np.abs(diff)
    relative_abs_diff = np.divide(abs_diff,gt+np.finfo(float).eps) # added epsilon to avoid division by zero.
    MRAEs[f] = np.mean(relative_abs_diff)

    # compute RMSE
    square_diff = np.power(diff,2)
    RMSEs[f] = np.sqrt(np.mean(square_diff))


    print(f)
    print(MRAEs[f])
    print(RMSEs[f])


MRAE = np.mean(MRAEs.values())
print("MRAE:\n"+MRAE.astype(str))
RMSE = np.mean(RMSEs.values())
print("\nRMSE:\n"+RMSE.astype(str))


with open(output_dir + '/scores.txt', 'w') as output_file:
    output_file.write("MRAE:"+MRAE.astype(str))
    output_file.write("\nRMSE:"+RMSE.astype(str))