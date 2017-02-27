from mifis import mifis
import numpy as np
import csv

# Load multiple instance data from csv
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

# read data and labels
data,labels = synth()

n_rules = 4
n_inputs = 2 
# create a multiple instance fuzzy inference system
fis = mifis(data,labels,n_rules,n_inputs)

# train MI-ANFIS for 100 epochs
fis.train(100)

# convert MIANFIS to Fuzzy.ai agent
input_names = ['number_clicks','session_duration']
output_name = 'conversion'
fuzzyai_agent = fis.to_fuzzyai_agent(input_names, output_name)
print(fuzzyai_agent)

#fis.plot_rules()
