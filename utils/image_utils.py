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
    '''
    Calculates PSNR between two images, considering only the masked region.
    
    Args:
        img1 (torch.Tensor): The first image tensor (e.g., rendered).
        img2 (torch.Tensor): The second image tensor (e.g., ground truth).
        mask (torch.Tensor): A binary mask tensor (1 for ROI, 0 for ignore).
    '''
    if sum(mask) == 0:
        # If mask is empty, return 0
        return torch.tensor(0.0, device=img1.device)
    
    # Calculate mean squared error over the masked region
    squared_error = (img1 - img2) ** 2
    masked_squared_error = squared_error * mask
    mse = masked_squared_error.sum() / mask.sum()
    # Retuns PSNR value
    return 20 * torch.log10(1.0 / torch.sqrt(mse))