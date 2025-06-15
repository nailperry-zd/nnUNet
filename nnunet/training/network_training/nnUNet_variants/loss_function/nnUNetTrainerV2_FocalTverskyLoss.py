#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
import torch
from torch import nn
import numpy as np
import math
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss

class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth=1e-5, alpha=0.7, beta=0.3, gamma=0.75, do_bg=True, batch_dice=True):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.do_bg = do_bg
        self.batch_dice = batch_dice

    def forward(self, y_pred, y_true):
        print(f"y_pred.shape={y_pred.shape}, y_true.shape={y_true.shape}")
        # Ensure the predictions are in the same dimension as y_true
        # y_pred = torch.sigmoid(y_pred)
        y_pred = torch.softmax(y_pred, dim=1)

        if not self.do_bg:
            y_pred = y_pred[:, 1:]


        # Calculate the Tversky loss
        tp = (y_true * y_pred).sum(dim=(2, 3, 4))
        fn = (y_true * (1 - y_pred)).sum(dim=(2, 3, 4))
        fp = ((1 - y_true) * y_pred).sum(dim=(2, 3, 4))

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)

        # Calculate the Focal Tversky loss
        loss = (1 - tversky_index).pow(self.gamma)
        loss_ret = loss.mean()
        print(f"loss_ret={loss_ret}")
        return loss_ret

class FocalTversky_DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        self.dc = FocalTverskyLoss(**soft_dice_kwargs)


    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class nnUNetTrainerV2_FocalTverskyDiceLoss(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = FocalTverskyDiceLoss")
        self.loss = FocalTversky_DC_and_CE_loss({"alpha": 0.4, "beta": 0.3, "gamma": 1, 'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

class nnUNetTrainerV2_FocalTverskyDiceLoss_FN9_FP1(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = FocalTverskyDiceLoss_FN9_FP1")
        self.loss = FocalTversky_DC_and_CE_loss({"alpha": 0.9, "beta": 0.1, "gamma": 1, 'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

