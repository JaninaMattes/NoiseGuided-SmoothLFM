
# Code adapted from DCTdiff Repo
# https://github.com/forever208/DCTdiff/blob/DCTdiff/utils.py
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from absl import logging
import cv2
from PIL import Image


""" Utils for DCTdiffusion """

def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = unpreprocess_fn(sample_fn(mini_batch_size))
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1


def combine_blocks(blocks, height, width, block_sz):
    image = np.zeros((height, width), np.float32)
    index = 0
    for i in range(0, height, block_sz):
        for j in range(0, width, block_sz):
            image[i:i + block_sz, j:j + block_sz] = blocks[index]
            index += 1
    return image



""" DCT functions """

def split_into_blocks(img, block_sz):
    blocks = []
    for i in range(0, img.shape[0], block_sz):
        for j in range(0, img.shape[1], block_sz):
            blocks.append(img[i:i + block_sz, j:j + block_sz])  # first row, then column
    return np.array(blocks)

def combine_blocks(blocks, height, width, block_sz):
    img = np.zeros((height, width), np.float32)
    index = 0
    for i in range(0, height, block_sz):
        for j in range(0, width, block_sz):
            img[i:i + block_sz, j:j + block_sz] = blocks[index]
            index += 1
    return img

def dct_transform(blocks):
    dct_blocks = []
    for block in blocks:
        dct_block = np.float32(block) - 128  # Shift to center around 0
        dct_block = cv2.dct(dct_block)
        dct_blocks.append(dct_block)
    return np.array(dct_blocks)

def idct_transform(blocks):
    idct_blocks = []
    for block in blocks:
        idct_block = cv2.idct(block)
        idct_block = idct_block + 128  # Shift back
        idct_blocks.append(idct_block)
    return np.array(idct_blocks)



""" DCT to Image """

def DCT_to_RGB(sample, tokens=0, low_freqs=0, block_sz=0, reverse_order=None, resolution=0, Y_bound=None):
    num_y_blocks = tokens * 4
    num_cb_blocks = tokens
    cb_blocks_per_row = int((resolution / block_sz) / 2)
    Y_blocks_per_row = int(resolution / block_sz)

    assert sample.shape == (tokens, low_freqs*6)
    sample = np.clip(sample, -2, 2)  # clamp into [-1, 1]
    sample = sample.reshape(tokens, 6, low_freqs)  # (tokens, 6, low_freqs)

    # fill up DCT coes
    DCT = np.zeros((tokens, 6, block_sz * block_sz))  # (tokens, 6, 16)
    DCT[:, :, :low_freqs] = sample
    DCT = DCT[..., reverse_order]  # convert the low to high freq order back to 8*8 order

    Y_bound = np.array(Y_bound)
    DCT_Y = DCT[:, :4, :] * Y_bound  # (64, 4, 16)
    DCT_Cb = DCT[:, 4, :] * Y_bound  # (64, 16)
    DCT_Cr = DCT[:, 5, :] * Y_bound  # (64, 16)

    DCT_Cb = DCT_Cb.reshape(num_cb_blocks, block_sz, block_sz)  # (64, 16) --> (64, 4, 4)
    DCT_Cr = DCT_Cr.reshape(num_cb_blocks, block_sz, block_sz)  # (64, 16) --> (64, 4, 4)

    y_blocks = []
    for row in range(cb_blocks_per_row):  # 16 cb/cr blocks, so 4*4 spatial blocks
        tem_ls = []
        for col in range(cb_blocks_per_row):
            ind = row * cb_blocks_per_row + col
            y_blocks.append(DCT_Y[ind, 0, :])
            y_blocks.append(DCT_Y[ind, 1, :])
            tem_ls.append(DCT_Y[ind, 2, :])
            tem_ls.append(DCT_Y[ind, 3, :])
        for ele in tem_ls:
            y_blocks.append(ele)
    DCT_Y = np.array(y_blocks).reshape(num_y_blocks, block_sz, block_sz)  # (256, 4, 4)

    # Apply Inverse DCT on each block
    idct_y_blocks = idct_transform(DCT_Y)
    idct_cb_blocks = idct_transform(DCT_Cb)
    idct_cr_blocks = idct_transform(DCT_Cr)

    # Combine blocks back into images
    height, width = resolution, resolution
    y_reconstructed = combine_blocks(idct_y_blocks, height, width, block_sz)
    cb_reconstructed = combine_blocks(idct_cb_blocks, int(height / 2), int(width / 2), block_sz)
    cr_reconstructed = combine_blocks(idct_cr_blocks, int(height / 2), int(width / 2), block_sz)

    # Upsample Cb and Cr to original size
    cb_upsampled = cv2.resize(cb_reconstructed, (width, height), interpolation=cv2.INTER_LINEAR)
    cr_upsampled = cv2.resize(cr_reconstructed, (width, height), interpolation=cv2.INTER_LINEAR)

    # Step 5: Convert YCbCr back to RGB
    R = y_reconstructed + 1.402 * (cr_upsampled - 128)
    G = y_reconstructed - 0.344136 * (cb_upsampled - 128) - 0.714136 * (cr_upsampled - 128)
    B = y_reconstructed + 1.772 * (cb_upsampled - 128)

    rgb_reconstructed = np.zeros((height, width, 3))
    rgb_reconstructed[:, :, 0] = np.clip(R, 0, 255)
    rgb_reconstructed[:, :, 1] = np.clip(G, 0, 255)
    rgb_reconstructed[:, :, 2] = np.clip(B, 0, 255)

    # Convert to uint8
    rgb_reconstructed = np.uint8(rgb_reconstructed)  # (h, w, 3), RGB channels

    return rgb_reconstructed


def DCTsamples_to_grid_image(samples, tokens=0, low_freqs=0, block_sz=0,
                             reverse_order=None, resolution=0, grid_sz=0, path=None, Y_bound=None):
    samples = samples.detach().cpu().numpy()
    rgb_imgs = []
    for sample in samples:
        rgb_reconstructed = DCT_to_RGB(sample, tokens, low_freqs, block_sz, reverse_order, resolution, Y_bound)
        rgb_imgs.append(rgb_reconstructed)
    rgb_imgs = np.array(rgb_imgs)
    img_sz = rgb_imgs.shape[1]

    # Fill the grid image with the 36 smaller images
    grid_image = np.zeros((grid_sz * img_sz, grid_sz * img_sz, 3), dtype=np.uint8)
    for i in range(grid_sz):
        for j in range(grid_sz):
            idx = i * grid_sz + j
            if idx < rgb_imgs.shape[0]:
                grid_image[i * img_sz:(i + 1) * img_sz, j * img_sz:(j + 1) * img_sz, :] = rgb_imgs[idx]

    # Convert the NumPy array to an image and save or show it
    final_image = Image.fromarray(grid_image)
    final_image.save(path)


def DCTsample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn,
                  tokens=0, low_freqs=0, reverse_order=None, resolution=0, block_sz=8, Y_bound=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes
    print(f'using Y_bound {Y_bound} for sampling')

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = sample_fn(mini_batch_size)
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        samples = samples.detach().cpu().numpy()
        if accelerator.is_main_process:
            for sample in samples:
                rgb_reconstructed = DCT_to_RGB(sample, tokens, low_freqs, block_sz, reverse_order, resolution, Y_bound)

                cv2.imwrite(os.path.join(path, f"{idx}.jpg"), cv2.cvtColor(rgb_reconstructed, cv2.COLOR_RGB2BGR))
                idx += 1

