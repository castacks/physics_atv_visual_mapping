import torch
import torch.nn.functional as F

def setup_kernel(metadata, kernel_type='box', kernel_radius=1., kernel_sharpness=1.):
    """
    Setup terrain estimation kernel
    Args:
        kernel_type: one of {"gaussian", "box"} the type of kernel to use
        kernel_radius: half-length of the kernel in m
        kernel_sharpness: for gaussian, the sharpness of the gaussian
        metadata: corresponding metadata for the kernel
    """
    kernel_dx = (kernel_radius / metadata.resolution[:2]).round().long()

    if kernel_type == 'box':
        return box_kernel(kernel_dx)
    elif kernel_type == 'gaussian':
        return gaussian_kernel(kernel_dx, kernel_sharpness)
    elif kernel_type == 'circle':
        return circle_kernel(kernel_dx)
    elif kernel_type == 'neighbors':
        return torch.tensor([
            [0., 1., 0.],
            [1., -4., 1.],
            [0., 1., 0.]
        ])
    elif kernel_type=='sobel_x':
        return sobel_x_kernel(kernel_dx)
    elif kernel_type=='sobel_y':
        return sobel_y_kernel(kernel_dx)
    elif kernel_type=='scharr_x':
        return torch.tensor([
            [-3., -10., -3.],
            [0., 0., 0.],
            [3., 10., 3.]
        ])/32
    elif kernel_type=='scharr_y':
        return torch.tensor([
            [-3., 0., 3.],
            [-10., 0., 10.],
            [-3., 0., 3.]
        ])/32

def box_kernel(rad):
    return torch.ones(2*rad[0]+1, 2*rad[1]+1)

def circle_kernel(rad):
    xs = torch.linspace(-1., 1., 2*rad[0] + 1)
    ys = torch.linspace(-1., 1., 2*rad[1] + 1)
    xs, ys = torch.meshgrid(xs, ys)

    return (torch.hypot(xs, ys) <= 1.).float()

def gaussian_kernel(rad, sharp):
    xs = torch.linspace(-1., 1., 2*rad[0] + 1) * sharp
    ys = torch.linspace(-1., 1., 2*rad[1] + 1) * sharp
    xs, ys = torch.meshgrid(xs, ys) 

    return torch.exp(-0.5 * torch.hypot(xs, ys)**2)

#https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size
def sobel_x_kernel(rad):
    """
    Since we care about metric stuff being right, divide by correction factor
    https://www.researchgate.net/publication/239398674_An_Isotropic_3x3_Image_Gradient_Operator
    """
    dxs = torch.arange(-rad[0], rad[0]+1)
    dys = torch.arange(-rad[1], rad[1]+1)

    dxs, dys = torch.meshgrid(dxs, dys, indexing='ij')
    ds = dxs**2 + dys**2

    kernel = torch.arange(-rad[0], rad[0]+1).view(-1, 1).tile(1, 2*rad[0]+1)
    kernel = kernel / (ds + 1e-6)
    kernel = kernel / kernel.abs().sum()

    return kernel

def sobel_y_kernel(rad):
    """
    Since we care about metric stuff being right, divide by correction factor
    https://www.researchgate.net/publication/239398674_An_Isotropic_3x3_Image_Gradient_Operator
    """
    dxs = torch.arange(-rad[0], rad[0]+1)
    dys = torch.arange(-rad[1], rad[1]+1)

    dxs, dys = torch.meshgrid(dxs, dys, indexing='ij')
    ds = dxs**2 + dys**2

    kernel = torch.arange(-rad[0], rad[0]+1).view(1, -1).tile(2*rad[1]+1, 1)
    kernel = kernel / (ds + 1e-6)
    kernel = kernel / kernel.abs().sum()

    return kernel

def apply_kernel(kernel, data, pad_mode='constant', pad_value=0.):
    """
    apply kernel to data
    """
    #torch wants pad in img coordinates
    kernel_pad = [
        kernel.shape[1]//2,
        kernel.shape[1]//2,
        kernel.shape[0]//2,
        kernel.shape[0]//2,
    ]
    _data = F.pad(data, pad=kernel_pad, mode=pad_mode, value=pad_value)

    _kernel_shape = kernel.shape
    _data_shape = _data.shape
    data_shape = data.shape
    res = F.conv2d(_data.view(1, 1, *_data_shape), kernel.view(1, 1, *_kernel_shape))

    return res.view(data_shape)

def get_adjacencies(data):
    """
    Args:
        data: A WxH tensor of values
    Returns:
        adj: A WxHx4 Tensor of rolled values (copy edges)
    """
    left = torch.roll(data, -1, 0)
    left[-1] = left[-2]

    right = torch.roll(data, 1, 0)
    right[0] = right[1]

    up = torch.roll(data, -1, 1)
    up[:, -1] = up[:, -2]

    down = torch.roll(data, 1, 1)
    down[:, 0] = down[:, 1]

    return torch.stack([left, right, up, down], axis=0)