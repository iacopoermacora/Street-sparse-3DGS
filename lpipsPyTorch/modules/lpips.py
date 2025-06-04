import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import get_network, LinLayers
from .utils import get_state_dict
from typing import Optional, Sequence


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None):

        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        
        if mask is None:
            res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]
        else:
            # Apply masked computation
            res = []
            for d, l in zip(diff, self.lin):
                # Get feature map dimensions
                _, _, h_feat, w_feat = d.shape
                
                # Resize mask to match feature dimensions
                mask_resized = self._resize_mask_to_features(mask, h_feat, w_feat)
                
                # Apply linear layer
                weighted_diff = l(d)  # Shape: (B, 1, H_feat, W_feat)
                
                # Apply spatial masking and weighted averaging
                masked_weighted = weighted_diff * mask_resized
                
                # Sum over spatial dimensions and normalize by valid area
                spatial_sum = torch.sum(masked_weighted, dim=(2, 3), keepdim=True)  # (B, 1, 1, 1)
                valid_area = torch.sum(mask_resized, dim=(2, 3), keepdim=True)  # (B, 1, 1, 1)
                
                # Avoid division by zero
                valid_area = torch.clamp(valid_area, min=1e-8)
                normalized_result = spatial_sum / valid_area
                
                res.append(normalized_result)

        return torch.sum(torch.cat(res, 0), 0, True)
    
    def _resize_mask_to_features(self, mask: torch.Tensor, h_feat: int, w_feat: int) -> torch.Tensor:
        """Resize mask to match feature map dimensions."""
        # Handle different mask input shapes
        if mask.dim() == 2:  # (H, W)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif mask.dim() == 3:  # (B, H, W) or (1, H, W)
            if mask.shape[0] == 1:  # (1, H, W)
                mask = mask.unsqueeze(1)  # (1, 1, H, W)
            else:  # (B, H, W)
                mask = mask.unsqueeze(1)  # (B, 1, H, W)
        # mask should now be (B, 1, H, W)
        
        # Resize to feature dimensions using nearest neighbor interpolation
        mask_resized = F.interpolate(
            mask.float(), 
            size=(h_feat, w_feat), 
            mode='nearest'
        )
        
        return mask_resized
