# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: minimal Dice and Dice+CE composite loss

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss
from .utils import get_class_weight, weight_reduce_loss


def _dice_loss(pred,
               target,
               valid_mask,
               smooth=1.0,
               exponent=2,
               class_weight=None,
               eps=1e-6):
    # pred, target: N,C,H,W
    assert pred.shape == target.shape
    pred = pred * valid_mask
    target = target * valid_mask

    dims = (0, 2, 3)
    numerator = 2 * torch.sum(pred * target, dims) + smooth
    denominator = torch.sum(
        pred.pow(exponent) + target.pow(exponent), dims) + smooth
    loss = 1 - numerator / (denominator + eps)

    if class_weight is not None:
        loss = loss * class_weight
    return loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    """Dice loss for semantic segmentation.

    Args:
        use_sigmoid (bool): Whether to apply sigmoid instead of softmax.
        activate (bool): If True, apply activation to predictions.
        reduction (str): Reduction type, 'none' | 'mean' | 'sum'.
        class_weight (list[float] | str | None): Class weights.
        loss_weight (float): Overall loss weight.
        smooth (float): Smoothing term to avoid division by zero.
        exponent (int): Exponent for denominator.
        ignore_index (int): Label to ignore.
        eps (float): Numerical stability epsilon.
    """

    def __init__(self,
                 use_sigmoid=False,
                 activate=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 smooth=1.0,
                 exponent=2,
                 ignore_index=255,
                 eps=1e-6):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.activate = activate
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.smooth = smooth
        self.exponent = exponent
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                pred = pred.softmax(dim=1)

        target = target.long()
        if ignore_index is None:
            ignore_index = self.ignore_index
        valid_mask = (target != ignore_index).float()
        if weight is not None:
            valid_mask = valid_mask * weight.float()
        valid_mask = valid_mask.unsqueeze(1)

        num_classes = pred.shape[1]
        target_clamped = torch.clamp(target, min=0)
        target_onehot = F.one_hot(target_clamped, num_classes=num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        class_weight = None
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)

        loss = _dice_loss(
            pred,
            target_onehot,
            valid_mask,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            eps=self.eps,
        )

        loss = weight_reduce_loss(loss, reduction=reduction,
                                  avg_factor=avg_factor)
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceCELoss(nn.Module):
    """Combined Dice + Cross Entropy loss."""

    def __init__(self,
                 ce_weight=1.0,
                 dice_weight=1.0,
                 use_sigmoid=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 smooth=1.0,
                 exponent=2,
                 ignore_index=255,
                 eps=1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.ce = CrossEntropyLoss(
            use_sigmoid=use_sigmoid,
            reduction=reduction,
            class_weight=class_weight,
            loss_weight=ce_weight,
        )
        self.dice = DiceLoss(
            use_sigmoid=use_sigmoid,
            reduction=reduction,
            class_weight=class_weight,
            loss_weight=dice_weight,
            smooth=smooth,
            exponent=exponent,
            ignore_index=ignore_index,
            eps=eps,
        )

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        ce_loss = self.ce(
            pred,
            target,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            ignore_index=ignore_index,
        )
        dice_loss = self.dice(
            pred,
            target,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            ignore_index=ignore_index,
        )
        return self.loss_weight * (ce_loss + dice_loss)


@LOSSES.register_module()
class DiceCEConfidenceLoss(DiceCELoss):
    """Dice + Cross Entropy loss for confidence-weighted pseudo labels.

    Behavior matches DiceCELoss, but is kept separate so mix-loss configs
    can explicitly opt into confidence-weighted training.
    """
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        ce_weight = weight
        dice_weight = weight
        if isinstance(weight, dict):
            ce_weight = weight.get('ce', None)
            dice_weight = weight.get('dice', None)
        elif isinstance(weight, (tuple, list)) and len(weight) == 2:
            ce_weight, dice_weight = weight
        ce_loss = self.ce(
            pred,
            target,
            weight=ce_weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            ignore_index=ignore_index,
        )
        dice_loss = self.dice(
            pred,
            target,
            weight=dice_weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            ignore_index=ignore_index,
        )
        return self.loss_weight * (ce_loss + dice_loss)
