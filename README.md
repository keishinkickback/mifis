#Multiple Instance Fuzzy Inference
## About 
Code for paper: 

•	Amine Ben Khalifa and Hichem Frigui, “MI-ANFIS: A Multiple Instance Adaptive Neuro-Fuzzy Inference System”, IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), Istanbul, Turkey, August 2015.

MIFIS is novel machine learning framework that employs fuzzy inference to solve the problem of multiple instance learning (MIL). Fuzzy Inference (FI) is a powerful tool to reason under uncertainty, imprecision, and vagueness. However, existing FI methods cannot address the ambiguity that arises in multiple instance learning (MIL) problems where each object is represented by a collection of instances, called a bag. In MIL, labels are available only at the bag level: A bag is labeled positive if at least one of its instances is positive and negative otherwise. In this work, we generalized FI systems to learn from partially and ambiguously labeled data. We expanded the architecture of the standard ANFIS to allow reasoning with bags and derive a back-propagation algorithm to learn the premise and consequent parameters of the expanded network.

## Installation
MIFIS is built with [Pytorch](http://pytorch.org/). To be able to use it you'll need to install the following dependencies

[Pytorch](https://github.com/pytorch/pytorch#install-optional-dependencies)

[Scipy](https://www.scipy.org/)

[Numpy](http://www.numpy.org/)

[Matplotlib](http://matplotlib.org/)

Then copy the code to your local machine

```bash
git clone https://github.com/aminert/mifis.git
```

## Usage
Currently only order zero MI-ANFIS is supported. To be able to train it to learn and use fuzzy rules, you'll need to assemble training data, see `synth.csv` for an example. The data doesn’t have to be labeled at a fine level, only high level labels are required. In general in Multiple Instance Learning the data is grouped into bags of instances and only labels of bags are available and not that of individual instances. 

In `example.py`, I'm providing a data loader that will read `synth.csv`. In this problem, we will try to predict a customer conversion score for a given user. We have collected historical data on a set of 100 users, the features we tracked are `number_clicks` and `session_duration`. These features where tracked over a period of time, in this setting, every time the user visited the website we recorded these values. We also have a label `conversion` either 0 or 1, for each user, indicating wether the user converted to a customer or not. Because, it's not clear which visit to the website made the user convert, we will assign the label to all his previous visits (the number of visits (instance) could be different from customer to customer), thus making the problem a multiple instance problem. 

Without having to label each instance (visit), we can use the collected data as is, and train an MI-ANIFS that given a set of visits and and corresponding `number_clicks` and `session_duration` we can predict the user `conversion` score.  

```bash
# read data and labels
data,labels = synth()

n_rules = 4
n_inputs = 2

# create a multiple instance fuzzy inference system
fis = mifis(data,labels,n_rules,n_inputs)

# train MI-ANFIS for 100 epochs
fis.train(100)
```

## Fuzzy.ai
In order to use MI-ANFIS generated rules with fuzzy.ai api, we created a function that approximates MI-ANFIS gaussian based MFs and rules to fuzzy.ai triangular fuzzy sets. Since MI-ANFIS doesn't know the linguistic terms, it assigns default terms to the fuzzy sets. feel free to edit those before creating your agent within fuzzy.ai.

Example Code
```bash
# convert MIANFIS to Fuzzy.ai agent
input_names = ['number_clicks','session_duration']
output_name = 'conversion'

fuzzyai_agent = fis.to_fuzzyai_agent(input_names, output_name)

print(fuzzyai_agent)
```

Example Generated fuzzy.ai agent
```bash
{
    "inputs": {
        "number_clicks": {
            "term1": [-2.7,-0.64,1.43],
            "term2": [0.36,2.99,5.62],
            "term3": [-2.71,1.02,4.74 ],
            "term4": [-2.11,2.92,7.94]
        },
        "session_duration": {
            "term1": [-2.91,-0.36,2.18],
            "term2": [-2.1,1.95,6.0],
            "term3": [-4.32,-0.1,4.11],
            "term4": [-4.05,1.57,7.19]
        }
    },
    "outputs": {
        "conversion": {
            "term1": [0.4,0.41, 0.42],
            "term2": [-2.58,-2.57,-2.56],
            "term3": [ -0.38,-0.37, -0.36],
            "term4": [2.32,2.33,2.34]
        }
    },
    "rules": [
        "IF number_clicks IS term1 AND session_duration IS term1 THEN conversion IS term1",
        "IF number_clicks IS term2 AND session_duration IS term2 THEN conversion IS term2",
        "IF number_clicks IS term3 AND session_duration IS term3 THEN conversion IS term3",
        "IF number_clicks IS term4 AND session_duration IS term4 THEN conversion IS term4"
    ]
}
```
Note how MI-ANFIS rules are well fine tuned!

##License
( The MIT License )

Copyright (c) 2017 Amine Ben khalifa 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE



