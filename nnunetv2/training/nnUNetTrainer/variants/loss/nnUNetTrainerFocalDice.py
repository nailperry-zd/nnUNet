import torch
from torch import nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, SoftDiceLoss
import numpy as np

from nnunet.training.network_training.nnUNet_variants.loss_function.nnUNetTrainerV2_focalLoss import \
    FocalLoss

from nnunetv2.utilities.helpers import softmax_helper_dim1


class Focal_DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, fl_kwargs=None, weight_ce=1.0, weight_dice=1.0, weight_fl=1.0,
                 ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Focal_DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_fl = weight_fl
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        if fl_kwargs is None:
            fl_kwargs = {}
        self.fl = FocalLoss(apply_nonlin=nn.Softmax(), **fl_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(Focal_DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        fl_loss = self.fl(net_output, target)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_fl * fl_loss
        return result


class nnUNetTrainerFocalDice(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = Focal_DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                     'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1,
                                    weight_dice=1,
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=MemoryEfficientSoftDiceLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerFocalDice05(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = Focal_DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                     'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                    weight_ce=1,
                                    weight_dice=0.5,
                                    weight_fl=1.0,
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=MemoryEfficientSoftDiceLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerFocal05Dice(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = Focal_DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                     'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                    weight_ce=0.5,
                                    weight_dice=1,
                                    weight_fl=0.5,
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=MemoryEfficientSoftDiceLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss