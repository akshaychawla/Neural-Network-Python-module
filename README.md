# Neural Network Python module

This is a modular and easy to use supervised neural network library based on the Backpropagation algorithm.

## 1. Dependencies

  * Numpy
  * cPickle (inbuilt)
  * csv		(inbuilt)
  * random	(inbuilt)

## 2. Features

  * Simple and easy to use from a python/ipython shell.
  * Just supply with dataset and labels numpy matrices.
  * User configurable parameters:
    * Number of epochs for training.
    * split ratio for training data and test data.
  * load and save weights to a local file for accessing later.

## 3. Usage

### 3.1 Test using Iris flower dataset
  Running the script auto_test_iris.py will automatically load the iris dataset, normalise it, and train a backpropagation neural network for the data. It will also test the data and give the Success and Error rate.

### 3.2 Running in a python/ipython shell
  Launch an ipython/python shell in the directory containing the module. Import the numpy module and load your own dataset. 
  After that use the NeuralNet library in neural.py on the data for training. 

  ```python
  #assuming the user data is in the format -> dataset , labels

  from neural import *

  nn = NeuralNet(4, 10, 3)	# initialise neural network 
  nn.train(dataset,labels)  # train it on the dataset
  print nn.classify(x)		# classify the test vector x and print the results
  output_network = nn.classify(x) # OR classify the test vector x and store the results
  ```



