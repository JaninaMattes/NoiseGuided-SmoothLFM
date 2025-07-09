# Code adpated from:
# - https://github.com/forever208/DCTdiff/blob/DCTdiff/datasets.py#L392
import numpy as np
import torchvision.transforms.functional as F
import cv2

""" Discrete Cosine Transform, Type II (a.k.a. the DCT) """

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
