import torch
import torch.nn as nn

import torch.nn.functional as F
from nnunet.training.loss_functions.focal_loss import FocalLossPerSampleRaw
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

softmax_helper = lambda x: F.softmax(x, 1)


class SymmetricSelfPacedLearning(nn.Module):
    def __init__(self, current_epoch, gradients_map):
        super().__init__()
        self.eta = 1
        self.current_epoch = current_epoch + 1
        # self.epoch_step_size = 2 / (1000 - 1)
        self.epoch_step_size = 2 / 1000
        self.weight_first = 2 - self.current_epoch * self.epoch_step_size
        self.weight_last = 2 - self.weight_first
        print(f"weight_first = {self.weight_first}, weight_last={self.weight_last}")
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
        sorted_items = sorted(difficulty_map.items(), key=lambda item: item[1], reverse=True)
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
    def __init__(self, fl_kwargs, soft_dice_kwargs, epoch_for_weighting=0):
        super().__init__()
        self.fl = FocalLossPerSampleRaw(apply_nonlin=softmax_helper, **fl_kwargs)
        self.epoch_for_weighting = epoch_for_weighting

    def forward(self, logit, target, current_epoch, do_backprop, keys, gradients_map):
        result_fl = self.fl(logit, target)
        print(f"FocalLoss is {result_fl}, device={result_fl.device}, input_shape={logit.shape}")
        if do_backprop and current_epoch > self.epoch_for_weighting:
            spl = SymmetricSelfPacedLearning(current_epoch, gradients_map)
            weighted_loss = spl(result_fl, keys)
            print(f"dynamically weighted, do_backprop={do_backprop}, current_epoch={current_epoch}")
            return weighted_loss.mean()
        else:
            print(f"equally weighted, do_backprop={do_backprop}, current_epoch={current_epoch}")
            return result_fl.mean()

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
    focal_loss_spl = FocalLossNonBatch_SPL({}, {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})
    tmp = focal_loss_spl(pre, label, epoch, True, ['123', '789'], {'123':2, '345':100, '789':39})
    print(tmp)