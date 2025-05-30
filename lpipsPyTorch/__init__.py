import torch

from .modules.lpips import LPIPS
from typing import Optional, Sequence


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          mask: Optional[torch.Tensor] = None, # PACOMMENT: Added mask parameter
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        mask (torch.Tensor, optional): binary mask tensor with same spatial dims as x, y.
                                     Shape: (B, 1, H, W) or (B, H, W) or (1, H, W) or (H, W)
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y, mask) # PACOMMENT: Pass mask to the LPIPS criterion
