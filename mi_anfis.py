import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from  gaussmf import GaussMF
from  mi_softmax import MISOFTMAX
from torch.nn.parameter import Parameter
import torch.optim as optim

class MIANFIS(nn.Module):
  def __init__(self,n_rules,n_inputs):
    super(MIANFIS, self).__init__()
    self.n_rules = n_rules
    self.gmfs = []
    for i in range(self.n_rules):
      setattr(self, 'gmf'+str(i), GaussMF(n_inputs))
      self.gmfs.append(eval('self.gmf'+str(i)))
    self.softmax = MISOFTMAX(2)
    self.bias = Parameter(torch.Tensor(self.n_rules)) 
    self.bias.data.uniform_(0, 1)

  def forward(self, input):
    x = [gmf(input) for gmf in self.gmfs] 
    #print("===Fuzzification=======")
    #print(x)
    ts = [torch.prod(actv,1) for actv in x]
    #print("===Truth Instances=====")
    #print(ts)
    f = self.softmax(ts[0])
    for i in range(1,self.n_rules):
      f = torch.cat((f,self.softmax(ts[i])))
    #print("===Rules Activations===")
    #print(f)
    nf = f/f.sum().expand_as(f)
    #print("===Norm Activations====")
    #print(nf)
    #print(self.bias)
    out = torch.sum(self.bias*f)
    #print("===Output==============")
    #print(out)
    return out
