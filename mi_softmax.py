import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter



class MISOFTMAX(nn.Module):
    r"""Applies a SOFTMAX transformation to the incoming vector: :math:`y = max(x)`
    Args:
    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`
    Attributes:
    Examples::
    """

    def __init__(self, alpha):
        super(MISOFTMAX, self).__init__()
        self.alpha = alpha

    def forward(self, input):
            return torch.sum(input*torch.exp(input*self.alpha))/torch.sum(torch.exp(input*self.alpha))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.alpha) + ')'

