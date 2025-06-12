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

class Adaptive_Region_Specific_TverskyLoss(nn.Module):
    def __init__(self, smooth=1e-5, target_patch_hw=64, do_bg=True, batch_dice=True, A=0.3, B=0.4, apply_nonlin=True):
        """
        num_region_per_axis: the number of boxes of each axis in (z, x, y)
        3D num_region_per_axis's axis in (z, x, y)
        2D num_region_per_axis's axis in (x, y)
        """
        super(Adaptive_Region_Specific_TverskyLoss, self).__init__()
        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.A = A
        self.B = B
        self.apply_nonlin = apply_nonlin
        self.target_patch_hw = target_patch_hw

    def get_dynamic_output_size(self, input_shape):
        """
        Dynamically compute output size for AdaptiveAvgPool based on input shape.

        Args:
            input_shape: Shape of x tensor [batch, c, z, x, y] for 3D or [batch, c, x, y] for 2D.

        Returns:
            Output size list for AdaptiveAvgPool3d/2d.
        """
        if len(input_shape) == 5:  # 3D case
            D = 1  # Do not split along depth (z-axis)
            H = max(1, math.ceil(input_shape[3] / self.target_patch_hw))
            W = max(1, math.ceil(input_shape[4] / self.target_patch_hw))
            return [D, H, W]
        elif len(input_shape) == 4:  # 2D case
            H = max(1, math.ceil(input_shape[2] / self.target_patch_hw))
            W = max(1, math.ceil(input_shape[3] / self.target_patch_hw))
            return [H, W]
        else:
            raise ValueError("Invalid input shape. Expected 4D or 5D tensor.")

    def forward(self, x, y):
        print(f"y_pred.shape={x.shape}, y_true.shape={y.shape}")
        # 2D/3D: [batchsize, c, (z,) x, y]
        if self.apply_nonlin:
            x = torch.softmax(x, dim=1)

        shp_x, shp_y = x.shape, y.shape
        dim = len(shp_x) - 2  # 2D or 3D check
        assert dim in [2, 3], "Input tensor must be 2D or 3D."

        if not self.do_bg:
            x = x[:, 1:]

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        # the three in [batchsize, class_num, (z,) x, y]
        tp = x * y_onehot
        fp = x * (1 - y_onehot)
        fn = (1 - x) * y_onehot

        # Dynamically calculate output size and create pool
        output_size = self.get_dynamic_output_size(shp_x)
        print(f"dynamic output_size = {output_size} for input_size = {shp_x}")
        if dim == 3:
            pool = nn.AdaptiveAvgPool3d(output_size)
        elif dim == 2:
            pool = nn.AdaptiveAvgPool2d(output_size)

        # the three in [batchsize, class_num, (num_region_per_axis_z,) num_region_per_axis_x, num_region_per_axis_y]
        region_tp = pool(tp)
        region_fp = pool(fp)
        region_fn = pool(fn)

        if self.batch_dice:
            region_tp = region_tp.sum(0)
            region_fp = region_fp.sum(0)
            region_fn = region_fn.sum(0)

        # [(batchsize,) class_num, (num_region_per_axis_z,) num_region_per_axis_x, num_region_per_axis_y]
        alpha = self.A + self.B * (region_fp + self.smooth) / (region_fp + region_fn + self.smooth)
        beta = self.A + self.B * (region_fn + self.smooth) / (region_fp + region_fn + self.smooth)

        # [(batchsize,) class_num, (num_region_per_axis_z,) num_region_per_axis_x, num_region_per_axis_y]
        region_tversky = (region_tp + self.smooth) / (region_tp + alpha * region_fp + beta * region_fn + self.smooth)
        region_tversky = 1 - region_tversky

        # [(batchsize,) class_num]
        # if self.batch_dice:
        #     region_tversky = region_tversky.sum(list(range(1, len(shp_x)-1)))
        # else:
        #     region_tversky = region_tversky.sum(list(range(2, len(shp_x))))

        region_tversky = region_tversky.mean()

        print(f"region_tversky loss is {region_tversky}")

        return region_tversky

class Region_DC_and_CE_loss(nn.Module):
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

        self.dc = Adaptive_Region_Specific_TverskyLoss(**soft_dice_kwargs)


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

class nnUNetTrainerV2_RegionalDiceLoss(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = RegionalDiceLoss")
        self.loss = Region_DC_and_CE_loss({'target_patch_hw': 20, 'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

class nnUNetTrainerV2_RegionalDiceLoss_patchhw320(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = RegionalDiceLoss")
        self.loss = Region_DC_and_CE_loss({'target_patch_hw': 320, 'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})


