#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim_masked(img1, img2, mask, window_size=11, size_average=True):
    """
    Compute SSIM only in masked regions.
    
    Args:
        img1, img2: Input images [B, C, H, W]
        mask: Binary mask [B, 1, H, W] or [B, C, H, W] where 1=include, 0=exclude
        window_size: Size of Gaussian window
        size_average: Whether to average the result
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_masked(img1, img2, mask, window, window_size, channel, size_average)


def _ssim_masked(img1, img2, mask, window, window_size, channel, size_average=True):
    """
    Compute SSIM only in masked regions with proper masking of local neighborhoods.
    """
    # Handle mask shape - ensure it has the channel dimension
    if len(mask.shape) == 3:  # [B, H, W] -> [B, 1, H, W]
        mask = mask.unsqueeze(1)

    # Ensure mask has same number of channels as images
    if mask.size(1) == 1 and channel > 1:
        mask = mask.expand(-1, channel, -1, -1)
    
    # Apply mask to images first
    masked_img1 = img1 * mask
    masked_img2 = img2 * mask
    
    # Compute local means with proper normalization for masked regions
    mu1_sum = F.conv2d(masked_img1, window, padding=window_size // 2, groups=channel)
    mu2_sum = F.conv2d(masked_img2, window, padding=window_size // 2, groups=channel)
    
    # Compute local mask weights (how many valid pixels in each window)
    # FIX: Use the expanded mask, not the original mask
    mask_weights = F.conv2d(mask.float(), window, padding=window_size // 2, groups=channel)
    
    # Avoid division by zero and compute proper local means
    valid_windows = mask_weights > 1e-6
    mu1 = torch.zeros_like(mu1_sum)
    mu2 = torch.zeros_like(mu2_sum)
    mu1[valid_windows] = mu1_sum[valid_windows] / mask_weights[valid_windows]
    mu2[valid_windows] = mu2_sum[valid_windows] / mask_weights[valid_windows]
    
    # Compute local variances and covariances with proper normalization
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Compute second moments
    sigma1_sum = F.conv2d(masked_img1 * masked_img1, window, padding=window_size // 2, groups=channel)
    sigma2_sum = F.conv2d(masked_img2 * masked_img2, window, padding=window_size // 2, groups=channel)
    sigma12_sum = F.conv2d(masked_img1 * masked_img2, window, padding=window_size // 2, groups=channel)
    
    # Compute variances and covariance
    sigma1_sq = torch.zeros_like(sigma1_sum)
    sigma2_sq = torch.zeros_like(sigma2_sum)
    sigma12 = torch.zeros_like(sigma12_sum)
    
    sigma1_sq[valid_windows] = sigma1_sum[valid_windows] / mask_weights[valid_windows] - mu1_sq[valid_windows]
    sigma2_sq[valid_windows] = sigma2_sum[valid_windows] / mask_weights[valid_windows] - mu2_sq[valid_windows]
    sigma12[valid_windows] = sigma12_sum[valid_windows] / mask_weights[valid_windows] - mu1_mu2[valid_windows]
    
    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Compute SSIM only for valid windows
    ssim_map = torch.zeros_like(mu1)
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    valid_ssim = (denominator > 1e-10) & valid_windows
    ssim_map[valid_ssim] = numerator[valid_ssim] / denominator[valid_ssim]
    
    # Apply original mask to ensure we only consider originally masked regions
    ssim_map = ssim_map * mask
    
    if size_average:
        # Average only over regions that were both originally masked and had valid windows
        valid_mask = mask * valid_windows.float()
        return ssim_map.sum() / valid_mask.sum().clamp(min=1)
    else:
        return ssim_map