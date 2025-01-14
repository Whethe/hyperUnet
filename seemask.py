from PIL import Image
from pathlib import Path

from utils_cdiff.utils_fourier import *

from utils import pad_kspace

import h5py
import numpy as np

from models.select_mask import define_Mask

import json


def save_mask_as_image(mask, output_path):
    """Save the binary mask as a black and white image."""
    output_dir = Path(output_path).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    if mask_img.mode != 'L':
        mask_img = mask_img.convert('L')

    mask_img.save(output_path)

    print(f"Mask saved to {output_path}")

def preprocess_normalisation(img):

    img = img / abs(img).max()

    return img

def save_image(image, output_path):
    """Save the real part of the image as a grayscale image using PIL."""
    image_magnitude = np.abs(image)
    image_magnitude = (image_magnitude / np.max(image_magnitude) * 255).astype(np.uint8)

    image_pil = Image.fromarray(image_magnitude)
    image_pil.save(output_path)
    print(f"Image saved to {output_path}")

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

shape = (256, 256)
acceleration = 6
mask_output_path = './masks/mask.png'
image_output_path = './masks/reconstructed_image.png'
h5_file_path = 'data/demoImage.hdf5'

with open("./configs/mask.json", "r") as f:
    content = json.load(f)

mask_1d = define_Mask(content)
mask_1d = mask_1d[:, np.newaxis]
mask = np.repeat(mask_1d, 256, axis=1).transpose((1, 0))

with h5py.File(h5_file_path, 'r') as f:
    gt = f['tstOrg'][()]

print(gt.shape)
gt = pad_kspace(gt[0], shape)
gtimage = preprocess_normalisation(gt)
img_L = undersample_kspace(gt, mask, False, 1, 1)

save_mask_as_image(mask, mask_output_path)
save_image(img_L, image_output_path)
