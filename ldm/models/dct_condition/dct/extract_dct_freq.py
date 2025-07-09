import numpy as np
import torch
import cv2


######################################
#            DCT-extraction          #
######################################

# Code taken from Torch-DCT Repo
# https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py#L12
# 
# 
def dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))

def idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


""" Discrete Cosine Transform, Type II (a.k.a. the DCT) """
    
    
def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)    
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)



def split_low_high_dct(x, low_ratio=0.25, norm=None):
    """
    Splits input into low and high frequency components using 2D DCT.

    Args:
        x (Tensor): Input tensor, which can be either:
                    - (B, C, H, W) for image data
                    - (B, 1024) for 1D vector data
        low_ratio (float): Ratio for low frequency cutoff (0 < low_ratio <= 1)
        norm (str): DCT normalization mode (None or 'ortho')

    Returns:
        low_freq (Tensor): Low frequency image (B, C, H, W) or (B, 1024)
        high_freq (Tensor): High frequency image (B, C, H, W) or (B, 1024)
    """

    if len(x.shape) == 4:  # When the input is an image (B, C, H, W)
        B, C, H, W = x.shape
        dct_out = dct_2d(x, norm=norm)  # Apply 2D DCT

        # Create low-pass mask
        cutoff_h = int(H * low_ratio)
        cutoff_w = int(W * low_ratio)
        mask = torch.zeros_like(dct_out)
        mask[:, :, :cutoff_h, :cutoff_w] = 1.0

        # Apply masks
        low_dct = dct_out * mask
        high_dct = dct_out * (1 - mask)

        # Inverse DCT
        low_freq = idct_2d(low_dct, norm=norm)
        high_freq = idct_2d(high_dct, norm=norm)

    elif len(x.shape) == 2:  # When the input is a 1D vector (B, 1024)
        B, D = x.shape
        side_len = int(D ** 0.5)
        if side_len * side_len != D:
            raise ValueError(f"Vector length {D} cannot be reshaped into a square.")
        
        x_reshaped = x.view(B, 1, side_len, side_len) 
        
        # Apply the 2D DCT as before
        low_freq, high_freq = split_low_high_dct(x_reshaped, low_ratio, norm) 

        # Flatten the result back to a 1D vector 
        low_freq = low_freq.reshape(B, -1)  # Flatten to (B, 1024)
        high_freq = high_freq.reshape(B, -1)  # Flatten to (B, 1024)

    else:
        raise ValueError("Input tensor must have 2 or 4 dimensions")

    return low_freq, high_freq





######################################
#            DCT-extraction          #
######################################

def dct_transform(blocks):
    dct_blocks = []
    for block in blocks:
        block = np.float32(block) - 128
        dct_block = cv2.dct(block)
        dct_blocks.append(dct_block)
    return np.array(dct_blocks)

def idct_transform(blocks):
    idct_blocks = []
    for block in blocks:
        idct_block = cv2.idct(block)
        idct_block = idct_block + 128
        idct_blocks.append(idct_block)
    return np.array(idct_blocks)




######################################
#            DCT-extraction          #
######################################

""" High-frequency extraction using DCT """
def extract_high_frequencies_cv2(x, block_size=8, high_freqs=8):
    B, C, H, W = x.shape
    high_freqs_output = torch.zeros_like(x)
    
    for b in range(B):
        for c in range(C):
            img = x[b, c].cpu().numpy()
            blocks = [img[i:i+block_size, j:j+block_size] for i in range(0, H, block_size) for j in range(0, W, block_size)]
            dct_blocks = dct_transform(blocks)
            
            for i in range(len(dct_blocks)):
                block = dct_blocks[i]
                block[0:high_freqs, 0:high_freqs] = 0
                dct_blocks[i] = block
            
            idct_blocks = idct_transform(dct_blocks)
            reconstructed_img = np.zeros_like(img)
            idx = 0
            for i in range(0, H, block_size):
                for j in range(0, W, block_size):
                    reconstructed_img[i:i+block_size, j:j+block_size] = idct_blocks[idx]
                    idx += 1
            
            high_freqs_output[b, c] = torch.tensor(reconstructed_img).float()
    
    return high_freqs_output



""" Low-frequency extraction using DCT """
def extract_low_frequencies_cv2(x, block_size=8, low_freqs=8):
    B, C, H, W = x.shape
    low_freqs_output = torch.zeros_like(x)
    
    for b in range(B):
        for c in range(C):
            img = x[b, c].cpu().numpy()
            blocks = [img[i:i+block_size, j:j+block_size] for i in range(0, H, block_size) for j in range(0, W, block_size)]
            dct_blocks = dct_transform(blocks)
            
            for i in range(len(dct_blocks)):
                block = dct_blocks[i]
                block[low_freqs:, low_freqs:] = 0
                dct_blocks[i] = block
            
            idct_blocks = idct_transform(dct_blocks)
            reconstructed_img = np.zeros_like(img)
            idx = 0
            for i in range(0, H, block_size):
                for j in range(0, W, block_size):
                    reconstructed_img[i:i+block_size, j:j+block_size] = idct_blocks[idx]
                    idx += 1
            
            low_freqs_output[b, c] = torch.tensor(reconstructed_img).float()
    
    return low_freqs_output
