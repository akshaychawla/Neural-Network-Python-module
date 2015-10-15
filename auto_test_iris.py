import numpy as np
from neural import *
from data_helper import *

print 'This script auto tests the iris dataset with the Neural Network module'

#loading the dataset
dataset, labels = load_iris()
