import torch
import torch.nn as nn

import torch.nn.functional as F
from nnunet.training.loss_functions.dice_loss import SoftDice
from nnunet.training.loss_functions.focal_loss import FocalLossNonBatch
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


softmax_helper = lambda x: F.softmax(x, 1)

class SymmetricSelfPacedLearning(nn.Module):
    def __init__(self, current_epoch):
        super().__init__()
        self.eta = 1
        self.current_epoch = current_epoch + 1
        # self.epoch_step_size = 2 / (1000 - 1)
        self.epoch_step_size = 2 / 1000
        self.weight_first = 2 - self.current_epoch * self.epoch_step_size
        self.weight_last = 2 - self.weight_first
        print(f"weight_first = {self.weight_first}, weight_last={self.weight_last}")

    def forward(self, loss, difficulty):
        weight_matrix = self.compute_weight_matrix(difficulty)
        loss = loss * weight_matrix
        print(f"weight_matrix={weight_matrix}")
        print(f"difficulty={difficulty}")
        loss = loss.mean()
        return loss

    def compute_weight_matrix(self, example_difficulty):
        weight_matrix = self.weight_first + example_difficulty * (self.weight_last - self.weight_first)
        return weight_matrix

    # def compute_weight_matrix(self, example_difficulty):
    #     batch_step_size = (self.weight_first - self.weight_last) / (
    #             len(example_difficulty) - 1
    #     )
    #     weight_matrix = torch.zeros_like(example_difficulty)
    #     indices = torch.argsort(example_difficulty)
    #
    #     for i, index in enumerate(indices):
    #         weight_matrix[index] = self.weight_first - batch_step_size * i
    #
    #     return weight_matrix


class SoftDiceLoss_SPL(nn.Module):
    def __init__(self, soft_dice_kwargs, epoch_for_weighting=0):
        super().__init__()
        self.dice = SoftDice(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.epoch_for_weighting = epoch_for_weighting

    def forward(self, x, y, current_epoch):
        dice_index = self.dice(x, y)
        print(f"dc score is {dice_index}")
        dice_loss = 1 - dice_index
        difficulty = 1 - dice_index
        spl = SymmetricSelfPacedLearning(current_epoch)
        weighted_loss = spl(dice_loss, difficulty)
        if current_epoch > self.epoch_for_weighting:
            print("dynamically weighted")
            return weighted_loss
        else:
            print("equally weighted")
            return dice_loss.mean()


class FocalLossNonBatch_SPL(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, fl_kwargs, soft_dice_kwargs, epoch_for_weighting=0):
        super().__init__()
        self.fl = FocalLossNonBatch(apply_nonlin=softmax_helper, **fl_kwargs)
        self.dice = SoftDice(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.epoch_for_weighting = epoch_for_weighting

    def forward(self, logit, target, current_epoch):
        result_fl = self.fl(logit, target)
        print(f"FocalLoss is {result_fl}")
        dice_index = self.dice(logit, target)
        print(f"dc score is {dice_index}")
        difficulty = 1 - dice_index
        spl = SymmetricSelfPacedLearning(current_epoch)
        weighted_loss = spl(dice_index, difficulty)
        if current_epoch > self.epoch_for_weighting:
            print("dynamically weighted")
            return weighted_loss
        else:
            print("equally weighted")
            return result_fl.mean()


class nnUNetTrainerV2_SoftDiceLoss_SPL(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = SoftDiceLoss_SPL")
        self.loss = SoftDiceLoss_SPL({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})

class nnUNetTrainerV2_SoftDiceLoss_SPL_Baseline(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = SoftDiceLoss_SPL_Baseline")
        self.loss = SoftDiceLoss_SPL({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False}, epoch_for_weighting=1000)

class nnUNetTrainerV2_FocalLossNonBatch_SPL(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = FocalLossNonBatch_SPL")
        self.loss = FocalLossNonBatch_SPL({}, {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})

class nnUNetTrainerV2_FocalLossNonBatch_SPL_Baseline(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = FocalLossNonBatch_SPL_Baseline")
        self.loss = FocalLossNonBatch_SPL({}, {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False}, epoch_for_weighting=1000)


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

    epoch = 500
    dice_loss_spl = SoftDiceLoss_SPL({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})
    focal_loss_spl = FocalLossNonBatch_SPL({}, {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})
    tmp = focal_loss_spl(pre, label, epoch)
    print(tmp)
