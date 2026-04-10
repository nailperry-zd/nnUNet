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
import numpy as np
import torch
import torch.nn as nn
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.loss_functions.focal_loss import FocalLossPerSample, FocalLoss
from nnunet.training.network_training.spl import FocalLossNonBatch_SPL
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.network_training.spl import FocalLossNonBatch_SPL, FLCENonBatch_SPL


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


class nnUNetTrainerV2_zonal(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True

    def process_plans(self, plans):
        super().process_plans(plans)
        # self.num_input_channels += 3  # for seg from zonal mask

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["move_last_seg_chanel_to_data"] = True
        self.data_aug_params["all_segmentation_labels"] = [0, 1, 2]
        self.data_aug_params['selected_seg_channels'] = None


class nnUNetTrainerV2_zonal_FL(nnUNetTrainerV2_zonal):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = FL_and_CE_loss(alpha=0.5)

class nnUNet_Zonal_FocalLossNonBatch_SPL_HardFirst(nnUNetTrainerV2_zonal):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = FocalLossNonBatch_SPL_HardFirst")
        self.loss = FocalLossNonBatch_SPL({}, {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})
        self.save_latest_only = False

class nnUNet_Zonal_FocalLossNonBatch_SPL_Baseline(nnUNetTrainerV2_zonal):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = FocalLossNonBatch_SPL_Baseline")
        self.loss = FocalLossNonBatch_SPL({}, {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False}, epoch_for_weighting=1000)
        self.save_latest_only = False

class nnUNet_Zonal_CELossNonBatch_SPL_Baseline(nnUNetTrainerV2_zonal):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = CELossNonBatch_SPL_Baseline")
        self.loss = FocalLossNonBatch_SPL({'gamma':0}, {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False}, epoch_for_weighting=1000)
        self.save_latest_only = False

class nnUNet_Zonal_CELossNonBatch_SPL_HardFirst(nnUNetTrainerV2_zonal):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = CELossNonBatch_SPL")
        self.loss = FocalLossNonBatch_SPL({'gamma':0}, {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})
        self.save_latest_only = False

class nnUNet_Zonal_FLCELossNonBatch_SPL_HardFirst(nnUNetTrainerV2_zonal):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 1000
        print("Setting up self.loss = FLCELossNonBatch_SPL")
        self.loss = FLCENonBatch_SPL({'gamma':2}, {'gamma':0}, max_num_epochs=self.max_num_epochs, max_weight=2)
        self.save_latest_only = False

class nnUNet_Zonal_FLCELossNonBatch_SPL_HardFirst_MW4(nnUNetTrainerV2_zonal):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 1000
        print("Setting up self.loss = FLCELossNonBatch_SPL_MW4")
        self.loss = FLCENonBatch_SPL({'gamma':2}, {'gamma':0}, max_num_epochs=self.max_num_epochs, max_weight=4)
        self.save_latest_only = False

class nnUNet_Zonal_FLCELossNonBatch_SPL_HardFirst_EW(nnUNetTrainerV2_zonal):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 1000
        print("Setting up self.loss = FLCELossNonBatch_SPL_EW")
        self.loss = FLCENonBatch_SPL({'gamma':2}, {'gamma':0}, max_num_epochs=self.max_num_epochs, epoch_for_weighting=self.max_num_epochs)
        self.save_latest_only = False

class nnUNet_Zonal_KDPZ_FLCELossNonBatch_EW500(nnUNetTrainerV2_zonal):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 500
        print("Setting up self.loss = FLCELossNonBatch_EW")
        self.loss = FLCENonBatch_SPL({'gamma':2}, {'gamma':0}, max_num_epochs=self.max_num_epochs, epoch_for_weighting=self.max_num_epochs)
        # self.save_latest_only = False