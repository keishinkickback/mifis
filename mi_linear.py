import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from  mi_softmax import MISOFTMAX


class MILINEAR(nn.Module):
    r"""Applies a MI LINEAR transformation to the incoming data
    Args:
        in_features: dim input sample
        order = 0 | 1
    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math: (N, 1)`
    Attributes:
        weights: the learnable centers of the module of shape (1+in_features)
    """

    def __init__(self, in_features,order=0):
        super(GaussMF, self).__init__()
        self.in_features = in_features
        self.order = order 
        if self.order:
           self.weights = Parameter(torch.Tensor(1,in_features))
           self.bias = Parameter(torch.Tensor(1,1))
        else:
           self.bias = Parameter(torch.Tensor(1,1))
        self.softmax = MISOFTMAX(2)

    def reset_parameters(self,order):
        stdv = 1. / math.sqrt(self.centers.size(0))
        if order:
           self.weights.data.uniform_(-stdv, stdv)

    def forward(self, input):
            return torch.exp(torch.div(-torch.pow(input-self.centers.expand_as(input),2),(2*torch.pow(self.sigmas.expand_as(input),2))))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.in_features) + ')'

