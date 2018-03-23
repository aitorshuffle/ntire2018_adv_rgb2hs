import os.path
import random
import torchvision.transforms as transforms
import torch
# import torch.nn.functional as F
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset_from_dir_list
from PIL import Image, ImageOps
import h5py
import numpy as np
import spectral
from tqdm import tqdm
from joblib import Parallel, delayed
from util.spectral_color import dim_ordering_tf2th, dim_ordering_th2tf

class IcvlNtire2018Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.challenge = opt.challenge  # 'Clean' or 'RealWorld'
        self.root = opt.dataroot  # e.g. icvl_ntire2018
        assert (opt.phase in ['train', 'Validate', 'Test'])
        self.dirlist_rgb = [os.path.join(self.root, 'NTIRE2018_Train1_' + self.challenge), os.path.join(self.root, 'NTIRE2018_Train2_' + self.challenge)] if opt.phase == 'train' else [os.path.join(self.root, 'NTIRE2018_' + opt.phase + '_' + self.challenge)]  # A
        self.dirlist_hs = [os.path.join(self.root, 'NTIRE2018_Train1_Spectral'), os.path.join(self.root, 'NTIRE2018_Train2_Spectral')] if opt.phase == 'train' else [os.path.join(self.root, 'NTIRE2018_' + opt.phase + '_Spectral')]  # B

        self.paths_rgb = sorted(make_dataset_from_dir_list(self.dirlist_rgb))
        self.paths_hs = sorted(make_dataset_from_dir_list(self.dirlist_hs))
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        # self.AB_paths = sorted(make_dataset(self.dir_AB))

        # print('RETURN TO FULL SIZE PATHS_hs and RGB')  #fixme
        # self.paths_rgb = self.paths_rgb[:5]
        # self.paths_hs = self.paths_hs[:5]

        # to handle envi files, so that we can do partial loads
        self.use_envi = opt.use_envi
        if self.use_envi:
            # update self.dirlist_hs
            self.dirlist_hs_mat = self.dirlist_hs
            self.dirlist_hs = [os.path.join(self.root, 'NTIRE2018_Train_Spectral_envi')]

            print(spectral.io.envi.get_supported_dtypes())
            if opt.generate_envi_files:
                self.generate_envi_files(overwrite_envi=opt.overwrite_envi)
            # update self.paths_hs with the hdr files
            self.paths_hs = sorted(make_dataset_from_dir_list(self.dirlist_hs))
            # for dir_hs in self.dirlist_hs:
            #     if not os.path.exists(dir_hs):

        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        # AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        # AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        # AB = transforms.ToTensor()(AB)

        # load rgb image
        path_rgb = self.paths_rgb[index]
        rgb = Image.open(path_rgb)#.convert('RGB')
        # fixme set it between 0,1?
        # rgb = transforms.ToTensor()(rgb)  # rgb.shape: torch.Size([3, 1392, 1300])

        # sample crop locations
        # w = rgb.shape[2]  # over the tensor already
        # h = rgb.shape[1]  # over the tensor already
        w = rgb.width  #store them in self so as to accesswhile testing for cropping final result
        h = rgb.height
        
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        # actually crop rgb image
        if self.opt.phase.lower() == 'train':
            if self.opt.challenge.lower() == 'realworld':
                # print('realworld<----------------------------------jitter')
                rgb = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01)(rgb)

            rgb = transforms.ToTensor()(rgb)  # rgb.shape: torch.Size([3, 1392, 1300])

            # train on random crops
            rgb_crop = rgb[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]  # rgb_crop is created as a tensor already

        else:
            topdown_pad = (1536 - h) // 2
            leftright_pad = (1536 - w) // 2
            full_img_padding = (leftright_pad, topdown_pad, leftright_pad, topdown_pad)
            rgb_crop = ImageOps.expand(rgb, full_img_padding)
            rgb_crop = transforms.ToTensor()(rgb_crop)

        ## load hs image
        path_hs = self.paths_hs[index]
        if self.use_envi:
            hs = spectral.io.envi.open(path_hs) # https://github.com/spectralpython/spectral/blob/master/spectral/io/envi.py#L282  not loaded yet until read_subregion
            # hs.shape: Out[3]: (1392, 1300, 31) (nrows, ncols, nbands)
            # check dimensions and crop hs image (actually read only that one
            # print(rgb.shape)
            # print(hs.shape)
            assert (rgb.shape[1] == hs.shape[0] and rgb.shape[2] == hs.shape[1])
            hs_crop = (hs.read_subregion(row_bounds=(h_offset, h_offset + self.opt.fineSize), col_bounds=(w_offset, w_offset + self.opt.fineSize))).astype(float)
            # hs_crop.shape = (h,w,c)=(256,256,31) here
            hs_crop = hs_crop / 4095. * 255  # 4096: db max. totensor expects in [0, 255]
            hs_crop = transforms.ToTensor()(hs_crop)  # convert ndarray (h,w,c) [0,255]-> torch tensor (c,h,w) [0.0, 1.0]  #move to GPU only the 256,256 crop!good!
        else:
            mat = h5py.File(path_hs)  # b[{'rgb', 'bands', 'rad'}]  # Shape: (Bands, Cols, Rows) <-> (bands, samples, lines)
            hs = mat['rad'].value  #  ndarray (c,w,h)
            hs = np.transpose(hs)  # reverse axis order. ndarray (h,w,c). totensor expects this shape
            hs = hs / 4095. * 255  #4096: db max. totensor expects in [0, 255]

            hs = transforms.ToTensor()(hs)  # convert ndarray (h,w,c) [0,255] -> torch tensor (c,h,w) [0.0, 1.0]  #fixme why move everything and not only the crop to the gpu?

            # check dimensions and crop hs image
            # assert(rgb.shape[1] == hs.shape[1] and rgb.shape[2] == hs.shape[2])
            if self.opt.phase == 'train':
                # train on random crops
                hs_crop = hs[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            else:
                # Validate or Test
                hs_crop = hs #will pad on the net
                # topdown_pad = (1536 - 1392) // 2
                # leftright_pad = (1536 - 1300) // 2
                # hs_crop = F.pad(hs, (leftright_pad, leftright_pad, topdown_pad, topdown_pad))


        rgb_crop = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(rgb_crop)  #fixme still valid in icvl?
        hs_crop = transforms.Normalize(tuple([0.5] * 31), tuple([0.5] * 31))(hs_crop)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(rgb_crop.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            rgb_crop = rgb_crop.index_select(2, idx)
            hs_crop = hs_crop.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = rgb_crop[0, ...] * 0.299 + rgb_crop[1, ...] * 0.587 + rgb_crop[2, ...] * 0.114
            rgb_crop = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = hs_crop[0, ...] * 0.299 + hs_crop[1, ...] * 0.587 + hs_crop[2, ...] * 0.114
            hs_crop = tmp.unsqueeze(0)

        return_dict =  {'A': rgb_crop, 'B': hs_crop,
                        'A_paths': path_rgb, 'B_paths': path_hs}
        if self.opt.phase == 'Validate' or self.opt.phase == 'Test':
            return_dict['full_img_padding'] = full_img_padding

        return return_dict


    def generate_single_envi_file(self, fpath_hs_mat, overwrite_envi=False):
        dir_hs = self.dirlist_hs[0]  # for brevity
        hsmat = h5py.File(fpath_hs_mat)  # b[{'rgb', 'bands', 'rad'}]  # Shape: (Bands, Cols, Rows) <-> (bands, samples, lines)
        hsnp = hsmat['rad'].value  # hs image numpy array #  ndarray (c,w,h)spec
        # hdr = io.envi.read_envi_header(file='data/envi_template.hdr')
        # hdr = self.update_hs_metadata(metadata=hdr, wl=hsmat['bands'].value.flatten())
        hdr_file = os.path.join(dir_hs, os.path.splitext(os.path.basename(fpath_hs_mat))[0] + '.hdr')
        spectral.io.envi.save_image(hdr_file=hdr_file, image=np.transpose(hsnp).astype(np.int16), force=overwrite_envi,
                                    dtype=np.int16)  # dtype int16 range: [-32000, 32000]

    def generate_envi_files(self, overwrite_envi=False):

        if not os.path.exists(self.dirlist_hs[0]):
            os.makedirs(self.dirlist_hs[0])

        nb_free_cores=1
        Parallel(n_jobs=-1 - nb_free_cores)(
            delayed(self.generate_single_envi_file)(fpath_hs_mat=fpath_hs_mat, overwrite_envi=overwrite_envi) for fpath_hs_mat in tqdm(self.paths_hs))

    def create_base_hdr(self):
        hdr=[]
        """
        http://www.harrisgeospatial.com/docs/ENVIHeaderFiles.html#Example
        data_Type: The type of data representation:

1 = Byte: 8-bit unsigned integer
2 = Integer: 16-bit signed integer
3 = Long: 32-bit signed integer
4 = Floating-point: 32-bit single-precision
5 = Double-precision: 64-bit double-precision floating-point
6 = Complex: Real-imaginary pair of single-precision floating-point
9 = Double-precision complex: Real-imaginary pair of double precision floating-point
12 = Unsigned integer: 16-bit
13 = Unsigned long integer: 32-bit
14 = 64-bit long integer (signed)
15 = 64-bit unsigned long integer (unsigned)"""
        return hdr

    def update_hs_metadata(self, metadata, wl):

        metadata['interleave'] = 'bsq'  # (Rows, Cols, Bands) <->(lines, samples, bands)
        # metadata['lines'] = int(metadata['lines']) - 4  # lines = rows. Lines <= 1300
        # metadata['samples'] = 1392  # samples = cols. Samples are 1392 for the whole dataset
        # metadata['bands'] = len(wl)
        metadata['data type'] = 4  #5 = Double-precision: 64-bit double-precision floating-point         http://www.harrisgeospatial.com/docs/ENVIHeaderFiles.html#Example
        metadata['wavelength'] = wl
        metadata['default bands'] = [5, 15, 25]
        metadata['fwhm'] = np.diff(wl)
        metadata['vroi'] = [1, len(wl)]
        return metadata

    def __len__(self):
        return len(self.paths_rgb)

    def name(self):
        return 'icvl_ntire2018_dataset'
