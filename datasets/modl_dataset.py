import torch
from torch.utils.data import Dataset
import h5py as h5
import numpy as np

from models.select_mask import define_Mask
from utils import c2r
from utils import pad_kspace

from utils_cdiff.utils_fourier import *

class modl_dataset(Dataset):
    def __init__(self, mode, dataset_path, opt, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        self.prefix = 'trn' if mode == 'train' else 'tst'
        self.dataset_path = dataset_path
        self.opt = opt
        self.sigma = sigma

        with h5.File(self.dataset_path, 'r') as f:
            self.num_data = len(f[self.prefix + 'Org'])  # Total number of samples

    def __getitem__(self, index):
        """
        :x0: zero-filled reconstruction (2 x nrow x ncol) - float32
        :gt: fully-sampled image (2 x nrow x ncol) - float32
        :csm: coil sensitivity map (ncoil x nrow x ncol) - complex64
        :mask: undersample mask (nrow x ncol) - int8
        """
        # with h5.File(self.dataset_path, 'r') as f:
        #     gt = f['reconstruction_rss'][index]
        with h5.File(self.dataset_path, 'r') as f:
            gt, csm= f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index]

        gt = pad_kspace(gt)
        nrow, ncol = gt.shape
        if 'fMRI' in self.opt['mask']:
            mask_1d = define_Mask(self.opt)
            mask_1d = mask_1d[:, np.newaxis]
            mask = np.repeat(mask_1d, ncol, axis=1).transpose((1, 0))
            self.mask = mask  # (H, W)

        x0 = undersample_kspace(gt,self.mask)

        return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(csm), torch.from_numpy(self.mask.astype(np.float32))

    def __len__(self):

        return self.num_data

def preprocess_normalisation(img):

    img = img / abs(img).max()

    return img

def creat_mask(content):
    mask_1d = define_Mask(content)
    mask_1d = mask_1d[:, np.newaxis]
    mask = np.repeat(mask_1d, 256, axis=1).transpose((1, 0))

    return mask

def undersample_kspace(x, mask):

    # d.1.0.complex --> d.1.1.complex
    # WARNING: This function only take x (H, W), not x (H, W, 1)
    # x (H, W) & x (H, W, 1) return different results
    # x (H, W): after fftshift, the low frequency is at the center.
    # x (H, W, 1): after fftshift, the low frequency is NOT at the center.
    # use abd(fft) to visualise the difference

    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    fft = fft2(x)
    fft = fftshift(fft)
    fft = fft * mask

    # if is_noise:
    #     raise NotImplementedError
    #     fft = fft + generate_gaussian_noise(fft, noise_level, noise_var)

    fft = ifftshift(fft)
    x = ifft2(fft)

    return x

# def undersample(gt, csm, mask, sigma):
#     """
#     :get fully-sampled image, undersample in k-space and convert back to image domain
#     """
#     ncoil, nrow, ncol = csm.shape
#
#     # mask = create_undersampling_mask((nrow, ncol), accel_factor)
#
#     sample_idx = np.where(mask.flatten()!=0)[0]
#     noise = np.random.randn(len(sample_idx)*ncoil) + 1j*np.random.randn(len(sample_idx)*ncoil)
#     noise = noise * (sigma / np.sqrt(2.))
#     b = piA(gt, csm, mask, nrow, ncol, ncoil) + noise #forward model
#     atb = piAt(b, csm, mask, nrow, ncol, ncoil)
#     return atb
#
# def piA(im, csm, mask, nrow, ncol, ncoil):
#     """
#     fully-sampled image -> undersampled k-space
#     """
#     im = np.reshape(im, (nrow, ncol))
#     im_coil = np.tile(im, [ncoil, 1, 1]) * csm #split coil images
#     k_full = np.fft.fft2(im_coil, norm='ortho') #fft
#     if len(mask.shape) == 2:
#         mask = np.tile(mask, (ncoil, 1, 1))
#     k_u = k_full[mask!=0]
#     return k_u
#
# def piAt(b, csm, mask, nrow, ncol, ncoil):
#     """
#     k-space -> zero-filled reconstruction
#     """
#     if len(mask.shape) == 2:
#         mask = np.tile(mask, (ncoil, 1, 1))
#     zero_filled = np.zeros((ncoil, nrow, ncol), dtype=np.complex64)
#     zero_filled[mask!=0] = b #zero-filling
#     img = np.fft.ifft2(zero_filled, norm='ortho') #ifft
#     coil_combine = np.sum(img*csm.conj(), axis=0).astype(np.complex64) #coil combine
#     return coil_combine