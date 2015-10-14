#A class for neural network training 
import numpy as np 
import matplotlib.pyplot as plt 
from math import exp
import cPickle
import sys
import random

class NeuralNet():

	def __init__(self, input_neurons, hidden_neurons, output_neurons):
		'initializes the required matrices for the network'
		np.random.seed(1)
		self.v = np.matrix(np.random.random((input_neurons, hidden_neurons)))  
		self.w = np.matrix(np.random.random((hidden_neurons, output_neurons)))
		self.alpha = 0.1

	def activation(self,x):
		'appply sigmoid to all values of vector'
		return np.matrix(1./(1 + np.exp(-x)))

	def activation_der(self,x):
		'apply derivative of sigmoid to all values'
		op = np.multiply( self.activation(x) , 1 - self.activation(x) )
		return op 

	def train2(self, dataset, labels, epochs):
		'train the network based given dataset and labels'
		(row,col) = dataset.shape
		print 'we have {} training examples and {} length feature vectors for each example'.format(row,col)
		#TRAINING
		for j in range(epochs):
			
			row,_ = dataset.shape
			for i in range(row):

				#----FORWARD PROPAGATION----
				x = np.matrix(dataset[i,:]) #taking the ith taining example
				T = np.matrix(labels[i,:]) #taking the ith training label
				#finding output at hidden nodes
				local_field_hidden = x * self.v
				hidden_output = self.activation(local_field_hidden)
				
				#finding output at output node
				final_local_field  = hidden_output * self.w
				final_output       = self.activation(final_local_field)

				#----BACK PROPAGATION-------
				#finding error_delta at output layer
				del_output_layer = np.multiply( T - final_output , self.activation_der(final_local_field) )
			
				#finding error at hidden layer
				error_hidden_layer = del_output_layer * self.w.T

				#finding error_delta at hidden layer
				a = self.activation(local_field_hidden)
				del_hidden_layer = np.multiply(error_hidden_layer , np.multiply(a, 1-a))

				#-----WEIGHT UPDATION-------
				del_v      = x.T * del_hidden_layer
				self.v     = self.v + del_v*self.alpha

				del_w      = hidden_output.T * del_output_layer
				self.w     = self.w + del_w*self.alpha


			if j%100 is 0:
				print 'epoch: ',j,' error: ', np.mean(np.linalg.norm(abs(labels[i]-final_output)))

	def train(self, dataset, labels, epochs):
		'an alternative training method'
		for j in range(epochs):
			rows,_ = dataset.shape
			err    = 0
			for i in range(rows):
				#forward propagation
				x 					= np.matrix(dataset[i,:]) 
				T 					= np.matrix(labels[i,:]) 
				local_field_hidden 	= x * self.v
				hidden_output 		= self.activation(local_field_hidden)
				final_local_field  	= hidden_output * self.w
				final_output       	= self.activation(final_local_field)

				#back propagation of error
				del_output_layer = np.multiply( T - final_output , self.activation_der(final_local_field) )
				error_hidden_layer = del_output_layer * self.w.T
				a = self.activation(local_field_hidden)
				del_hidden_layer = np.multiply(error_hidden_layer , np.multiply(a, 1-a))

				#weight update using delta rule
				del_v      = x.T * del_hidden_layer
				self.v     = self.v + del_v*self.alpha
				del_w      = hidden_output.T * del_output_layer
				self.w     = self.w + del_w*self.alpha

				#storing error
				err += np.mean(np.linalg.norm(abs(T-final_output)))

			if j%100 is 0:
					print 'epoch: ',j,' MSE: ',err/rows				
	
	
	def classify(self,x):
		'classify the input vector'

		x = np.matrix(x)
		#finding output at hidden nodes
		local_field_hidden = x * self.v
		hidden_output = self.activation(local_field_hidden)
				
		#finding output at output node
		final_local_field  = hidden_output * self.w
		final_output       = self.activation(final_local_field)

		return final_output

	def test(self,dataset,labels):
		'test the neural network'
		err = 0
		for ex,T in zip(dataset,labels):
			nw_op = self.classify(ex)
			if np.argmax(nw_op) != np.argmax(T):
				err += 1
		print 'Report:'
		print 'Error Rate: {} \t Success Rate: {}'.format( float(err)*100/labels.shape[0]  , (1 - float(err)/labels.shape[0])*100 )

	def save_weights(self,name):
		'save the weights to a cPickle file'
		toStore = [self.v , self.w]
		f = open(name,'wb')
		cPickle.dump(f , toStore)
		f.close()

	def load_weights(self,name):
		'load the weights from a local stored file'
		f = open(name,'rb')
		self.v , self.w = cPickle.load(f)
		f.close()
			

#--testing--
f = open('iris_normalized','rb')
dataset = cPickle.load(f)
f.close()

f = open('iris_labels','rb')
labels = cPickle.load(f)
f.close()

data = np.concatenate((dataset, labels), axis=1)

import random
random.seed(5)
random.shuffle(data)

dataset, labels = data[:, :4], data[:, 4:]
trX, teX = dataset[:100, :], dataset[100:, :]
trY, teY = labels[:100, :], labels[100:, :]


nn = NeuralNet(4, 10, 3)
nn.train(trX, trY, 500)
nn.test(teX, teY)






