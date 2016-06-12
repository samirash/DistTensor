import scipy.io
import numpy as np


if dataset_name == 'cove1':

	cov1_data = scipy.io.loadmat('datasets/cov1.mat')
	print cov1_data
	#  cov1_data is a dict with the main keys being: [ 'testing_vectors', 'training_labels', 'training_vectors', 'testing_labels' ]
	dataset = cov1_data
	return dataset

elif dataset_name == 'mnist':

	mnist_data = scipy.io.loadmat('datasets/mnist.mat')
	print mnist_data
	#print type(mnist_data)
	#print mnist_data.keys()
	#  mnist_data is a dict with the main keys being: [ 'testing_vectors', 'training_labels', 'training_vectors', 'testing_labels' ]
	dataset = mnist_data
	return dataset

else:

	# I'm not entirely sure about using this dataset
	astro_file = open('datasets/ca-AstroPh.txt', 'r')
	print astro_file.readline()

