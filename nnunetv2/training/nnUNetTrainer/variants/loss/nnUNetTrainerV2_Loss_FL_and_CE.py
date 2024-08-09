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

from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerV2_focalLoss import FocalLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
from torch import nn

# TODO: replace FocalLoss by fixed implemetation (and set smooth=0 in that one?)


class FL_and_CE_loss(nn.Module):
    def __init__(self, fl_kwargs=None, ce_kwargs=None, alpha=0.5, aggregate="sum"):
        super(FL_and_CE_loss, self).__init__()
        if fl_kwargs is None:
            fl_kwargs = {}
        if ce_kwargs is None:
            ce_kwargs = {}

        self.aggregate = aggregate
        self.fl = FocalLoss(apply_nonlin=nn.Softmax(), **fl_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.alpha = alpha

    def forward(self, net_output, target):
        fl_loss = self.fl(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = self.alpha*fl_loss + (1-self.alpha)*ce_loss
        else:
            raise NotImplementedError("nah son")
        return result


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints(nnUNetTrainer):
    """
    Set loss to FL + CE and set checkpoints
    """

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = FL_and_CE_loss(alpha=0.5)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints2(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints3(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints4(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints5(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints6(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints7(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints8(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints9(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)


class nnUNetTrainerV2_Loss_FL_and_CE_checkpoints10(nnUNetTrainerV2_Loss_FL_and_CE_checkpoints):
    """
    Each run is stored in a folder with the training class name in it. This simply creates a new folder,
    to allow investigating the variability between restarts.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
