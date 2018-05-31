from __future__ import print_function, division
from torch.nn.modules.loss import _assert_no_grad, _Loss
import torch.nn.functional as F
import torch

# define a customized loss function for future development
class WeightedBCELoss(_Loss):

    def __init__(self, size_average=True, reduce=True):
        super(WeightedBCELoss, self).__init__(size_average, reduce)

    def forward(self, input, target, weight):
        _assert_no_grad(target)
        return F.binary_cross_entropy(input, target, weight, self.size_average,
                                      self.reduce)