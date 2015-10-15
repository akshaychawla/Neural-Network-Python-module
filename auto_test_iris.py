import numpy as np
from neural import *
from data_helper import *

print 'This script auto tests the iris dataset with the Neural Network module'

#loading the dataset
dataset, labels = load_iris()

#normalise the dataset
dataset 		= data_normalize(dataset)

#split data into train and test data
train_data,train_labels,test_data,test_labels = train_test_split(dataset,labels,0.8)

#neural network training
nn = NeuralNet(4, 10, 3)
nn.train(train_data, train_labels, 500)

#neural network testing
nn.test(test_data, test_labels)



