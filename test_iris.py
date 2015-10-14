#testing the iris dataset

import cPickle
import csv
import numpy as np 
import random
f = open('iris.csv','rt')
csv_reader = csv.reader(f)

dataset = []
labels  = []

for row in csv_reader:
	data = row[0:-1]
	data = [float(c) for c in data]
	
	if row[-1] == 'setosa':
		labels.append([1,0,0])
		dataset.append(data)
	elif row[-1] == 'versicolor':
		labels.append([0,1,0])
		dataset.append(data)
	elif row[-1] == 'virginica':
		labels.append([0,0,1])
		dataset.append(data)


#converting to a 150x4 numpy array
dataset = np.array(dataset)
labels  = np.array(labels)

#finding the maximum and minimum
minimum = np.min(dataset, axis=0)
maximum = np.max(dataset, axis=0)

#normalizing the dataset
dataset = (dataset - np.tile(minimum, (150,1))) / (np.tile(maximum,(150,1)) - np.tile(minimum,(150,1)) )

#checking dataset sizes
print dataset.shape
print labels.shape

#storing the data
f = open('iris_normalized','wb')
cPickle.dump(dataset, f)
f.close()

f = open('iris_labels','wb')
cPickle.dump(labels, f)
f.close()







