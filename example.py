from mifis import mifis
import numpy as np
import csv

def genSynth(n_dim, n_samples):
  n_instances = np.random.randint(2,10, size=n_samples)  
  data = []
  labels = []
  for i in range(n_samples):
    instances = 10 * np.random.random_sample((n_instances[i], n_dim))
    target = 1 # np.random.randn()
    data.append(instances.tolist())
    labels.append(target)
  return data,labels

def synth():
  with open('synth.csv', 'r') as csvfile:
    synthreader = csv.reader(csvfile, delimiter=',')
    next(synthreader, None)
    data = [None]*100
    labels = [None]*100
    for instance in  synthreader:
      target,bag_id = int(instance[3]), int(instance[0])
      dim1,dim2 = float(instance[1]), float(instance[2])
      if data[bag_id-1] is None:
        data[bag_id-1] = [[dim1,dim2]]
      else:
        data[bag_id-1].append([dim1,dim2])
      labels[bag_id-1] = target
  return data,labels

data,labels = synth()
print(data)
n_rules = 4
n_inputs = 2 
fis = mifis(data,labels,n_rules,n_inputs)
fis.to_fuzzyai(['number_clicks','session_duration'],'conversion')
#fis.plot_rules()
fis.train(200)
fis.plot_rules()
