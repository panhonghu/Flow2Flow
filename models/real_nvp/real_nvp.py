import torch
import torch.nn as nn
import torch.nn.functional as F
from models.real_nvp.coupling_layer import CouplingLayer, MaskType
from models.real_nvp.conv_layer import ActivationLayer, InvConv2d, InvConv2dLU
from util import squeeze_2x2


class RealNVP(nn.Module):
    """RealNVP Model
    Based on the paper:
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio
    (https://arxiv.org/abs/1605.08803).
    Args:
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
        `Coupling` layers.
        un_normalize_x (bool): Un-normalize inputs `x`: shift (-1, 1) to (0, 1)
            assuming we used `transforms.Normalize` with mean 0.5 and std 0.5.
        no_latent (bool): If True, assume both `x` and `z` are image distributions.
            So we should pre-process the same in both directions. E.g., True in CycleFlow.
    """
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, un_normalize_x=True, \
                       no_latent=False):
        super(RealNVP, self).__init__()
        # Register data_constraint to pre-process images, not learnable
        self.register_buffer('data_constraint', torch.tensor([0.9], dtype=torch.float32))
        self.un_normalize_x = un_normalize_x
        self.no_latent = no_latent
        # Get inner layers
        self.flows = _RealNVP(0, num_scales, in_channels, mid_channels, num_blocks)

    def forward(self, x, reverse=False):
        sldj = None
        if self.no_latent or not reverse:
            # De-quantize and convert to logits
            x, sldj = self._pre_process(x)
        x, sldj = self.flows(x, sldj, reverse)
        return x, sldj

    def _pre_process(self, x):
        """De-quantize and convert the input image `x` to logits.
        Args:
            x (torch.Tensor): Input image.
        Returns:
            y (torch.Tensor): Logits of `x`.
            ldj (torch.Tensor): Log-determinant of the Jacobian of the transform.
        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        if self.un_normalize_x:
            x = x * 0.5 + 0.5
        # Expect inputs in [0, 1]
        if x.min() < 0 or x.max() > 1:
            raise ValueError('Expected x in [0, 1], got x with min/max {}/{}'.format(x.min(), x.max()))
        # De-quantize
        x = (x * 255. + torch.rand_like(x)) / 256.
        # Convert to logits
        y = (2 * x - 1) * self.data_constraint  # [-0.9, 0.9]
        y = (y + 1) / 2                         # [0.05, 0.95]
        y = y.log() - (1. - y).log()            # logit
        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.data_constraint).log() - self.data_constraint.log())
        ldj = ldj.view(ldj.size(0), -1).sum(-1)
        return y, ldj


class _RealNVP(nn.Module):
    """Recursive builder for a `RealNVP` model.
    Each `_RealNVPBuilder` corresponds to a single scale in `RealNVP`,
    and the constructor is recursively called to build a full `RealNVP` model.
    Args:
        scale_idx (int): Index of current scale.
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
    """
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super(_RealNVP, self).__init__()
        self.is_last_block = scale_idx==num_scales-1
        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)
        ])
        if self.is_last_block:
            ### only one CouplingLayer for last block
            self.in_couplings.append(CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True))
        else:
            self.out_couplings = nn.ModuleList([
                CouplingLayer(4*in_channels, 2*mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
                CouplingLayer(4*in_channels, 2*mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
                # ActivationLayer(),
                CouplingLayer(4*in_channels, 2*mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)
            ])
            self.next_block = _RealNVP(scale_idx+1, num_scales, 2*in_channels, 2*mid_channels, num_blocks)

    def forward(self, x, sldj, reverse=False):
        if reverse:
            if not self.is_last_block:
                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)
                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, reverse=False)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)
            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)
            if not self.is_last_block:
                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, reverse=False)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)
                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)
        return x, sldj

