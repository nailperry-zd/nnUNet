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
from torch import nn

from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerV2_focalLoss import FocalLoss


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
            result = self.alpha * fl_loss + (1 - self.alpha) * ce_loss
        else:
            raise NotImplementedError("nah son")
        return result