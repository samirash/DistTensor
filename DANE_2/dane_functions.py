import numpy as np
import scipy.io as io
import scipy.sparse as sp
import scipy.linalg as la
from general_function_class import Ridge_regression
from general_function_class import Classification_smooth_hinge
import matplotlib.pyplot as plt
from dane_machine import Computing_machine



def initialize_machines(m, data):
	#
	# initialize_machines: Function for allocating machines, initializing their weights and distributing data between them.
	#
	# inputs:
	# 	data: N*(d+1) dimensional matrix. All training data. With d being the dimension of each datapoint and N the number of those
	#	m: the number of machines
	#	objective_form: A string which shows what kind of function we are assuming. E.g. ridge regression. 
	#
	# data_partial: is a n*d (or n*(d+1) matrix which would be assigned to one machine
	# machines: I used a list to store the machines since it can contain any king of object

	N = np.shape(data)[0]
	w_length = np.shape(data)[1]-1
	n = N/m
	b = N - m*(N/m)

	machines = [] 		# this is a list  


	# test!:  check the indices here!
	for i in range(b):
		#print i
		machine = Computing_machine( i, w_length )
		machine.get_data(data[ (i) * (n+1)  : (i+1) * (n+1) , :] )
		#print (i) * (n+1)  , (i+1) * (n+1)
		
		machines.append(machine)

	for i in range(m-b):
		#print b+i
		machine = Computing_machine( b+i , w_length )
		machine.get_data(data[  b * (n+1) + (i) * n   :  b * (n+1) + (i+1)*n   , :]  )
		#print  b * (n+1) + (i) * n   ,  b * (n+1) + (i+1)*n 
		machines.append(machine)

	return machines


def machines_setup(machines,w_opt, objective_form, objective_param, optimization_algorithm, *alg_param):
	'''
	 objective_form:			specifies what kind of function we are optimizing in our machines, e.g. 'ridge_regression'
	 optimization_algorithm:	specifies what algorithm we are using. e.g. DANE or ADMM					
	 '''

	 # here if not needed remove w_opt from the parameters

	m = len(machines)

	for i in range(m):

		machines[i].set_objective_form( objective_form, objective_param )
		machines[i].set_objective( )
		machines[i].set_optimization_algorithm( optimization_algorithm,  alg_param )


