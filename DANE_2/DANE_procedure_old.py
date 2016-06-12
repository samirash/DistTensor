import numpy as np
import scipy.io as io
import scipy.sparse as sp
import scipy.linalg as la
from .general_function_class import Ridge_regression

''' I need a class for my machines: needs different attributes: number of
 datapoints, previous parameters, previous gradient, etc.'''


print 'test-1'
x = np.array([[0,1],[1,2],[2,1]])
y = np.array([3,-4,7]).T
eta_val = 1
mu_val = 0
print len(x)
rg = Ridge_regression(x,y,eta_val, mu_val)


print rg.regul_coef
print rg.dim

w = np.array([0.0,0.0])
w_neighbour = np.array([0.01,0.005])
print rg.eval(w)
# rg.eval should be 74/3

print rg.grad(w)

print rg.prox(w,w_neighbour)




'''
class Computing_machine:

	def __init__(self, id, w_length):
		self.id = id
		self.weight = np.zero(w_length)

	def set_data(self, data):
		# data is a numpy array here
		self.data = data

	def add_data(self, data):
		self.data = np.concatenate(self.data, data)

	def set_weight(self, weigth):
		self.weigth = weigth



def initialize_machines(m, full_data):

	datapoints_number = shape(full_data)[1]
	a = datapoints_number/m
	b = datapoints_number
	for 


def DANE(N, m, eta=1, mu=0, w_length):
	# if mu=0 does not converge then you shall use 0.3
	# this is the main DANE procedure
	w_init = np.zeros(w_length)

	for t in range(max_iter):
		collective_gradient = 

'''
