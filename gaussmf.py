import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter



class GaussMF(nn.Module):
    r"""Applies a GaussMF transformation to the incoming data: :math:`y = gaussmf(x)`
    Args:
        in_features: size of each input sample
    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`
    Attributes:
        centers: the learnable centers of the module of shape (in_features)
        sigmas:   the learnable sigmas of the module of shape (in_features)
    Examples::
        >>> gmf = nn.GaussMF(4)
        >>> input = autograd.Variable(torch.randn(4))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features):
        super(GaussMF, self).__init__()
        self.in_features = in_features
        self.centers = Parameter(torch.Tensor(1,in_features))
        self.sigmas = Parameter(torch.Tensor(1,in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.centers.size(0))
        self.centers.data.uniform_(-stdv, stdv)
        self.sigmas.data.uniform_(1, 10)

    def forward(self, input):
            return torch.exp(torch.div(-torch.pow(input-self.centers.expand_as(input),2),(2*torch.pow(self.sigmas.expand_as(input),2))))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.in_features) + ')'

