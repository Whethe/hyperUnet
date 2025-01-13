import torch
from torch.utils.data import Dataset
import h5py as h5
import numpy as np

from utils import c2r

class modl_dataset(Dataset):
    def __init__(self, mode, dataset_path, samplerate, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        self.prefix = 'trn' if mode == 'train' else 'tst'
        self.dataset_path = dataset_path
        self.samplerate = samplerate
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
        with h5.File(self.dataset_path, 'r') as f:
            gt, csm= f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index]
        ncoil, nrow, ncol = csm.shape
        mask = create_undersampling_mask((nrow, ncol), self.samplerate)
        x0 = undersample(gt, csm, mask, self.sigma)

        return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(csm), torch.from_numpy(mask.astype(np.float32))

    def __len__(self):

        return self.num_data

def create_undersampling_mask(shape, accel_factor, center_fraction=0.1):
    """
        Create a variable density Cartesian pseudo-random undersampling mask.

        :param shape: Tuple of k-space dimensions (nrow, ncol).
        :param accel_factor: Acceleration factor (e.g., 10 for 10x undersampling).
        :param center_fraction: Fraction of the k-space to densely sample around the center.
        :return: Binary mask where 1 indicates sampled points.
    """
    nrow, ncol = shape
    num_low_freqs = int(round(center_fraction * min(nrow, ncol)))

    # Create radial distance map from the center
    y, x = np.ogrid[-nrow // 2:nrow // 2, -ncol // 2:ncol // 2]
    distance_from_center = np.sqrt(x ** 2 + y ** 2)
    distance_from_center /= distance_from_center.max()  # Normalize to [0, 1]

    # Create probability density map (higher near the center, lower at edges)
    prob_map = np.exp(-distance_from_center * 5)  # Exponential decay for smooth density

    # Shift the mask to match FFT frequency arrangement
    prob_map = np.fft.fftshift(prob_map)

    # Normalize and adjust for sampling rate
    prob_map /= prob_map.max()
    total_points = nrow * ncol
    num_samples = total_points // accel_factor

    # Generate the mask using the adjusted probabilities
    mask = np.random.rand(nrow, ncol) < prob_map * (num_samples / total_points)

    # Ensure center is fully sampled
    center_start_row = (nrow - num_low_freqs) // 2
    center_end_row = center_start_row + num_low_freqs
    center_start_col = (ncol - num_low_freqs) // 2
    center_end_col = center_start_col + num_low_freqs

    # Apply center mask in FFT-shifted coordinates
    mask = np.fft.fftshift(mask)
    mask[center_start_row:center_end_row, center_start_col:center_end_col] = 1
    mask = np.fft.ifftshift(mask)

    return mask.astype(np.int8)


def undersample(gt, csm, mask, sigma):
    """
    :get fully-sampled image, undersample in k-space and convert back to image domain
    """
    ncoil, nrow, ncol = csm.shape

    # mask = create_undersampling_mask((nrow, ncol), accel_factor)

    sample_idx = np.where(mask.flatten()!=0)[0]
    noise = np.random.randn(len(sample_idx)*ncoil) + 1j*np.random.randn(len(sample_idx)*ncoil)
    noise = noise * (sigma / np.sqrt(2.))
    b = piA(gt, csm, mask, nrow, ncol, ncoil) + noise #forward model
    atb = piAt(b, csm, mask, nrow, ncol, ncoil)
    return atb

def piA(im, csm, mask, nrow, ncol, ncoil):
    """
    fully-sampled image -> undersampled k-space
    """
    im = np.reshape(im, (nrow, ncol))
    im_coil = np.tile(im, [ncoil, 1, 1]) * csm #split coil images
    k_full = np.fft.fft2(im_coil, norm='ortho') #fft
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    k_u = k_full[mask!=0]
    return k_u

def piAt(b, csm, mask, nrow, ncol, ncoil):
    """
    k-space -> zero-filled reconstruction
    """
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    zero_filled = np.zeros((ncoil, nrow, ncol), dtype=np.complex64)
    zero_filled[mask!=0] = b #zero-filling
    img = np.fft.ifft2(zero_filled, norm='ortho') #ifft
    coil_combine = np.sum(img*csm.conj(), axis=0).astype(np.complex64) #coil combine
    return coil_combine