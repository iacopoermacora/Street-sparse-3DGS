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

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnr_masked(img1, img2, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=img1.device)
    
    # Calculate MSE per channel, only for masked pixels
    squared_error = (img1 - img2) ** 2
    mask_expanded = mask.expand_as(img1)
    
    # Apply mask and reshape to [channels, -1] to get MSE per channel
    masked_error = squared_error * mask_expanded
    masked_error_flat = masked_error.view(img1.shape[0], -1)
    mask_flat = mask_expanded.view(img1.shape[0], -1)
    
    # Calculate MSE per channel (sum of errors / number of valid pixels per channel)
    mse = masked_error_flat.sum(1, keepdim=True) / mask_flat.sum(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# def psnr_masked(img1, img2, mask):
#     '''
#     Calculates PSNR between two images, considering only the masked region.
    
#     Args:
#         img1 (torch.Tensor): The first image tensor (e.g., rendered).
#         img2 (torch.Tensor): The second image tensor (e.g., ground truth).
#         mask (torch.Tensor): A binary mask tensor (1 for ROI, 0 for ignore).
#     '''
#     if mask.sum() == 0:
#         # If mask is empty, return 0
#         return torch.tensor(0.0, device=img1.device)
    
#     # Calculate mean squared error over the masked region
#     squared_error = (img1 - img2) ** 2
#     masked_squared_error = squared_error * mask
#     mse = masked_squared_error.sum() / mask.sum()
#     # Retuns PSNR value
#     psnr = 20 * torch.log10(1.0 / torch.sqrt(squared_error.mean()))
#     print(f"PSNR full image: {psnr.item()}")
#     print(f"PSNR masked: {20 * torch.log10(1.0 / torch.sqrt(mse))}")
#     return 20 * torch.log10(1.0 / torch.sqrt(mse))

# NOTE: Something is wrong in the calculation of PSNR, check the formula and explore the values.