import numpy as np
import csv
import cPickle

def load_iris():
	'Load and format the iris dataset in numpy matrix format'
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

	return dataset,labels

def data_normalize(dataset):
	'Normalise the dataset between [0,1]'
	
	minimum = np.min(dataset, axis=0)
	maximum = np.max(dataset, axis=0)

	rows,_  = dataset.shape

	dataset_norm = (dataset - np.tile(minimum, (rows,1))) / (np.tile(maximum,(rows,1)) - np.tile(minimum,(rows,1)) )
	return dataset_norm

def train_test_split(dataset, labels, ratio = 0.7):
	'Split the dataset between training set and test set'

	

