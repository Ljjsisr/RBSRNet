import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def sh_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def ll_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@LOSS_REGISTRY.register()
class OURLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(OURLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, ll, lr, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # return self.loss_weight * (sh_loss(pred, target, weight, reduction=self.reduction) + ((0.8 * sh_loss(pred, target, weight, reduction=self.reduction) * (0.2 * ll_loss(ll, lr, weight, reduction=self.reduction)))))

        return self.loss_weight * (sh_loss(pred, target, weight, reduction=self.reduction) + ((0.8 * sh_loss(pred, target, weight, reduction=self.reduction) * (0.2 * ll_loss(ll, lr, weight, reduction=self.reduction)))))
