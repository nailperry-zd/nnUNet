import torch
import torch.nn as nn

import torch.nn.functional as F
from nnunet.training.loss_functions.dice_loss import SoftDice
from nnunet.training.loss_functions.focal_loss import FocalLossNonBatch
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
import os

softmax_helper = lambda x: F.softmax(x, 1)


class SymmetricSelfPacedLearning(nn.Module):
    def __init__(self, current_epoch, gradients_map, max_num_epochs=1000, max_weight=2, reverse=False):
        super().__init__()
        self.eta = 1
        self.reverse = reverse
        self.current_epoch = current_epoch + 1
        # self.epoch_step_size = 2 / (1000 - 1)
        self.epoch_step_size = max_weight / max_num_epochs
        self.weight_first = max_weight - self.current_epoch * self.epoch_step_size
        self.weight_last = max_weight - self.weight_first
        print(f"reverse={self.reverse}, weight_first = {self.weight_first}, weight_last={self.weight_last}")
        self.weight_map = self.compute_weight_map(gradients_map)

    def forward(self, loss, keys):
        weight_matrix = torch.ones(len(keys))
        for i, key in enumerate(keys):
            weight_matrix[i] = self.weight_map.get(key, 1)
        print(f"weight_matrix={weight_matrix}, keys={keys}")
        weight_matrix = weight_matrix.to(loss.device).detach()
        loss = loss * weight_matrix
        return loss

    def compute_weight_map(self, difficulty_map):
        # Sort example_difficulty based on values
        sorted_items = sorted(difficulty_map.items(), key=lambda item: item[1], reverse=self.reverse)
        sorted_indices = [item[0] for item in sorted_items]  # Get indices based on sorted keys

        # Calculate the batch step size
        batch_step_size = (self.weight_first - self.weight_last) / (len(difficulty_map) - 1)

        # Initialize weight_matrix
        weight_map = {}

        # Compute weights based on sorted order
        for i, index in enumerate(sorted_indices):
            weight = self.weight_first - batch_step_size * i
            weight_map[index] = weight

        return weight_map

class FocalLossNonBatch_SPL(nn.Module):

    def __init__(self, fl_kwargs, epoch_for_weighting=0, max_num_epochs=1000, max_weight=2, reverse=False):
        super().__init__()
        self.fl = FocalLossNonBatch(apply_nonlin=softmax_helper, **fl_kwargs)
        self.epoch_for_weighting = epoch_for_weighting
        self.max_num_epochs = max_num_epochs
        self.max_weight = max_weight
        self.reverse = reverse

    def forward(self, logit, target, current_epoch, do_backprop, keys, gradients_map):
        result_fl = self.fl(logit, target)
        print(f"FocalLoss is {result_fl}, device={result_fl.device}")
        if do_backprop and current_epoch > self.epoch_for_weighting:
            spl = SymmetricSelfPacedLearning(current_epoch, gradients_map, self.max_num_epochs, self.max_weight, self.reverse)
            weighted_loss = spl(result_fl, keys)
            print(f"dynamically weighted, do_backprop={do_backprop}, current_epoch={current_epoch}")
            return weighted_loss.mean()
        else:
            print(f"equally weighted, do_backprop={do_backprop}, current_epoch={current_epoch}")
            return result_fl.mean()


class FLCENonBatch_SPL(nn.Module):

    def __init__(self, fl_kwargs, ce_kwargs, epoch_for_weighting=0, max_num_epochs=1000, max_weight=2, reverse=False):
        super().__init__()
        self.fl = FocalLossNonBatch(apply_nonlin=softmax_helper, **fl_kwargs)
        self.ce = FocalLossNonBatch(apply_nonlin=softmax_helper, **ce_kwargs)
        self.epoch_for_weighting = epoch_for_weighting
        self.max_num_epochs = max_num_epochs
        self.max_weight = max_weight
        self.reverse = reverse

    def forward(self, logit, target, current_epoch, do_backprop, keys, gradients_map):
        result_fl = self.fl(logit, target)
        result_ce = self.ce(logit, target)
        ls = 0.5 * result_fl + 0.5 * result_ce
        if do_backprop and current_epoch > self.epoch_for_weighting:
            spl = SymmetricSelfPacedLearning(current_epoch, gradients_map, self.max_num_epochs, self.max_weight, self.reverse)
            weighted_loss = spl(ls, keys)
            print(f"dynamically weighted, do_backprop={do_backprop}, current_epoch={current_epoch}")
            return weighted_loss.mean()
        else:
            print(f"equally weighted, do_backprop={do_backprop}, current_epoch={current_epoch}")
            return ls.mean()

class FLCENonBatch_TZ_HigherW(nn.Module):

    def __init__(self, fl_kwargs, ce_kwargs, epoch_for_weighting=0, max_num_epochs=1000, max_weight=2, reverse=False):
        super().__init__()
        self.fl = FocalLossNonBatch(apply_nonlin=softmax_helper, **fl_kwargs)
        self.ce = FocalLossNonBatch(apply_nonlin=softmax_helper, **ce_kwargs)
        self.epoch_for_weighting = epoch_for_weighting
        self.max_num_epochs = max_num_epochs
        self.max_weight = max_weight
        self.reverse = reverse
        gt_dir = r"/eresearch/ai-multiparametric-mri-pc/dzha937/Archive/dzha937/picai/workdir/nnUNet_preprocessed/Task128_TZOnly/gt_segmentations"
        self.tz_case_stems = {
            fname.replace(".nii.gz", "")
            for fname in os.listdir(gt_dir)
            if fname.endswith(".nii.gz")
        }

    def forward(self, logit, target, current_epoch, do_backprop, keys, gradients_map):
        result_fl = self.fl(logit, target)
        result_ce = self.ce(logit, target)
        ls = 0.5 * result_fl + 0.5 * result_ce
        if do_backprop and current_epoch > self.epoch_for_weighting:
            spl = SymmetricSelfPacedLearning(current_epoch, gradients_map, self.max_num_epochs, self.max_weight, self.reverse)
            weighted_loss = spl(ls, keys)
            print(f"dynamically weighted, do_backprop={do_backprop}, current_epoch={current_epoch}")
            return weighted_loss.mean()
        else:
            print(f"TZ higher weighted, do_backprop={do_backprop}, current_epoch={current_epoch}")
            weight_matrix = torch.ones(len(keys))
            for i, key in enumerate(keys):
                if key in self.tz_case_stems:
                    weight_matrix[i] = 2
            print(f"weight_matrix={weight_matrix}, keys={keys}")
            weight_matrix = weight_matrix.to(ls.device).detach()
            loss = (ls * weight_matrix).sum() / weight_matrix.sum().clamp_min(1e-8)
            return loss

class nnUNetTrainerV2_FocalLossNonBatch_SPL_EasyFirst(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = FocalLossNonBatch_SPL_EasyFirst")
        self.loss = FocalLossNonBatch_SPL({})
        self.save_latest_only = False

class nnUNetTrainerV2_FocalLossNonBatch_SPL_HardFirst(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = FocalLossNonBatch_SPL_HardFirst")
        self.loss = FocalLossNonBatch_SPL({}, reverse=True)
        self.save_latest_only = False

class nnUNetTrainerV2_FocalLossNonBatch_SPL_Baseline(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = FocalLossNonBatch_SPL_Baseline")
        self.loss = FocalLossNonBatch_SPL({}, epoch_for_weighting=1000)
        self.save_latest_only = False

class nnUNetTrainerV2_CELossNonBatch_SPL_Baseline(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = CELossNonBatch_SPL_Baseline")
        self.loss = FocalLossNonBatch_SPL({'gamma':0}, epoch_for_weighting=1000)
        self.save_latest_only = False

class nnUNetTrainerV2_CELossNonBatch_EW_500(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 500
        print("Setting up self.loss = CELossNonBatch_SPL_Baseline")
        self.loss = FocalLossNonBatch_SPL({'gamma':0}, epoch_for_weighting=self.max_num_epochs)
        self.save_latest_only = True

class nnUNetTrainerV2_CELossNonBatch_SPL_HardFirst(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = CELossNonBatch_RSSPL")
        self.loss = FocalLossNonBatch_SPL({'gamma':0}, reverse=True)
        self.save_latest_only = False

class nnUNetTrainerV2_CELossNonBatch_SPL_HardFirst_500(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 500
        print("Setting up self.loss = CELossNonBatch_RSSPL")
        self.loss = FocalLossNonBatch_SPL({'gamma':0}, reverse=True)
        self.save_latest_only = True

class nnUNetTrainerV2_CELossNonBatch_SPL_HardFirst_MW4_500(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 500
        print("Setting up self.loss = CELossNonBatch_RSSPL")
        self.loss = FocalLossNonBatch_SPL({'gamma':0}, max_weight = 4, reverse=True)
        self.save_latest_only = False

class nnUNetTrainerV2_CELossNonBatch_SPL_EasyFirst(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = CELossNonBatch_SSPL")
        self.loss = FocalLossNonBatch_SPL({'gamma':0})
        self.save_latest_only = False

class nnUNetTrainerV2_CELossNonBatch_SPL_EasyFirst_500(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 500
        print("Setting up self.loss = CELossNonBatch_SSPL")
        self.loss = FocalLossNonBatch_SPL({'gamma':0})
        self.save_latest_only = False

class nnUNet_302_FLCELossNonBatch_SPL_HardFirst(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 1000
        print("Setting up self.loss = FLCELossNonBatch_RSSPL")
        self.loss = FLCENonBatch_SPL({'gamma':2}, {'gamma':0}, max_num_epochs=self.max_num_epochs, max_weight=2, reverse=True)
        self.save_latest_only = False

class nnUNet_302_FLCELossNonBatch_SPL_HardFirst_MW4(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 1000
        print("Setting up self.loss = FLCELossNonBatch_RSSPL_MW4")
        self.loss = FLCENonBatch_SPL({'gamma':2}, {'gamma':0}, max_num_epochs=self.max_num_epochs, max_weight=4, reverse=True)
        self.save_latest_only = False

class nnUNet_302_FLCELossNonBatch_SPL_HardFirst_MW4_500(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 500
        print("Setting up self.loss = FLCELossNonBatch_RSSPL_MW4")
        self.loss = FLCENonBatch_SPL({'gamma':2}, {'gamma':0}, max_num_epochs=self.max_num_epochs, max_weight=4, reverse=True)
        self.save_latest_only = False

class nnUNet_302_FLCELossNonBatch_SPL_HardFirst_EW(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 1000
        print("Setting up self.loss = FLCELossNonBatch_SPL_EW")
        self.loss = FLCENonBatch_SPL({'gamma':2}, {'gamma':0}, max_num_epochs=self.max_num_epochs, epoch_for_weighting=self.max_num_epochs)
        self.save_latest_only = False

class nnUNet_302_FLCELossNonBatch_SPL_HardFirst_EW_500(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 500
        print("Setting up self.loss = FLCELossNonBatch_SPL_EW")
        self.loss = FLCENonBatch_SPL({'gamma':2}, {'gamma':0}, max_num_epochs=self.max_num_epochs, epoch_for_weighting=self.max_num_epochs)
        self.save_latest_only = False

class nnUNet_302_FLCELossNonBatch_TZ_HigherW_500(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 500
        print("Setting up self.loss = FLCELossNonBatch_TZ_HigherW")
        self.loss = FLCENonBatch_TZ_HigherW({'gamma':2}, {'gamma':0}, max_num_epochs=self.max_num_epochs, epoch_for_weighting=self.max_num_epochs)
        self.save_latest_only = False

class nnUNet_302_CEDice_500(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.max_num_epochs = 500
        print("Setting up self.loss = default")

if __name__ == "__main__":
    # loss = torch.tensor([0, 0.8, 0.9, 0.1, 0.5, 1])  # loss = 1 - Dice
    # example_difficulty = loss
    # print(f"difficulty={example_difficulty}")
    # current_epoch = 100
    # spl = SymmetricSelfPacedLearning(current_epoch)
    # weight_matrix = spl.compute_weight_matrix(example_difficulty)
    # print(f"epoch={current_epoch}, weight_matrix={weight_matrix}")
    #
    # current_epoch = 500
    # spl = SymmetricSelfPacedLearning(current_epoch)
    # weight_matrix = spl.compute_weight_matrix(example_difficulty)
    # print(f"epoch={current_epoch}, weight_matrix={weight_matrix}")
    #
    # current_epoch = 800
    # spl = SymmetricSelfPacedLearning(current_epoch)
    # weight_matrix = spl.compute_weight_matrix(example_difficulty)
    # print(f"epoch={current_epoch}, weight_matrix={weight_matrix}")

    size = (2, 2, 16, 320, 320)
    size_label = (2, 1, 16, 320, 320)
    pre = torch.softmax(torch.rand(size), dim=1)
    label = torch.randint(0, 2, size_label)

    epoch = 900
    dice_loss_spl = SoftDiceLoss_SPL({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})
    # focal_loss_spl = FocalLossNonBatch_SPL({}, {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})
    tmp = dice_loss_spl(pre, label, epoch, True, ['123', '789'], {'123':2, '345':100, '789':39})
    print(tmp)
