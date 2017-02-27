from mi_anfis import MIANFIS
import torch
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import json
from json import encoder
import re
np.random.seed((1000,2000))

class mifis():
  def __init__(self,data,labels,n_rules,n_inputs):
    self.data = data
    self.labels = labels
    self.n_rules = n_rules
    self.n_inputs = n_inputs
    self.net = MIANFIS(n_rules,n_inputs)
    #self.init_centers()
  def plot_rules(self):
   fig = plt.figure()
   x = np.linspace(-8, 8, 100)
   k = 1 
   for r in range(self.n_rules):
     for d in range(self.n_inputs):
       cur_ax = fig.add_subplot(self.n_rules,self.n_inputs+1,k)
       cur_ax.plot(x, mlab.normpdf(x, self.net.gmfs[r].centers.data[0,d],self.net.gmfs[r].sigmas.data[0,d]), 'r-')
       k += 1
     cur_ax = fig.add_subplot(self.n_rules,self.n_inputs+1,k)
     cur_ax.plot(x, mlab.normpdf(x, self.net.bias.data[r],0.01), 'r-')
     k += 1
   plt.tight_layout()
   fig = plt.gcf() 
   plt.show()
 
  def init_centers(self):
    data_flat = list(self.data[0])
    for sample in self.data:
      data_flat += sample
    data_flat = np.array(data_flat)
    init_centers = kmeans(whiten(data_flat),self.n_rules)
    for gmf in self.net.gmfs:
      gmf.centers.data = torch.from_numpy(init_centers[0][0,:]).unsqueeze(0).type(torch.FloatTensor)

  def train(self,n_epochs):
    # create an optimizer
    optimizer = optim.SGD(self.net.parameters(), lr = 0.1)
    criterion = nn.MSELoss()
    # training loop:
    running_loss = 0.0
    j = 0
    for i in range(n_epochs):
      for sample,label in zip(self.data,self.labels):
        sample = torch.from_numpy(np.array(sample)).type(torch.FloatTensor)
        label = torch.from_numpy(np.array([label])).type(torch.FloatTensor)
        input,target = Variable(sample), Variable(label) 
        optimizer.zero_grad() # zero the gradient buffers
        output = self.net(input)
        loss = criterion(output, target)
        #print('Loss at %d=%f'%(i,loss.data[0]))
        loss.backward()
        optimizer.step() # Does the update
        # print statistics
        j += 1
        running_loss += loss.data[0]
        if j % 100 == 99: # print every 2000 mini-batches
            print('[%5d] loss: %.3f' % (j+1, running_loss / 100))
            running_loss = 0.0
    
  def to_rules(self,input_names,output_name):
    rules = {}
    for i in range(self.n_rules):
      rules['rule' + str(i+1)] ={'inputs': {input_names[j]:[self.net.gmfs[i].centers.data[0,j]-2*self.net.gmfs[i].sigmas.data[0,j],self.net.gmfs[i].centers.data[0,j],self.net.gmfs[i].centers.data[0,j]+2*self.net.gmfs[i].sigmas.data[0,j]] for j in range(self.n_inputs)}, output_name:[self.net.bias.data[i]-0.01,self.net.bias.data[i], self.net.bias.data[i]+0.01] } 
    disp = json.dumps(json.loads(json.dumps(rules), parse_float=lambda x: round(float(x), 2)),indent=4
)
    print(disp)

  def to_fuzzyai(self, input_names,output_name):
    txt = {}
    inputs_txt = {}
    for i in range(self.n_inputs):
       inputs_txt[input_names[i]] = {'term'+str(j+1):[self.net.gmfs[j].centers.data[0,i]-2*self.net.gmfs[j].sigmas.data[0,i],self.net.gmfs[j].centers.data[0,i],self.net.gmfs[j].centers.data[0,i]+2*self.net.gmfs[j].sigmas.data[0,i]] for j in range(self.n_rules)}
    output_txt = {}
    output_txt[output_name] = {'term'+str(j+1):[self.net.bias.data[j]-0.01,self.net.bias.data[j], self.net.bias.data[j]+0.01] for j in range(self.n_rules)}
    rules_txt = ["IF " + ''.join([input_names[i]+" IS "+ "term" + str(j+1)+ " AND " for i in range(self.n_inputs-1)]) + input_names[-1]+" IS "+ "term" + str(j+1) + " THEN " + output_name+" IS "+ "term" + str(j+1) for j in range(self.n_rules)]
    txt['inputs'] = inputs_txt
    txt['outputs'] = output_txt   
    txt['rules']=rules_txt
    disp = json.dumps(json.loads(json.dumps(txt), parse_float=lambda x: round(float(x), 2)),indent=4
)
    print(disp) 
