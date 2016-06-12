import numpy as np
import scipy.io as io
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
from general_function_class import Ridge_regression
from general_function_class import Classification_smooth_hinge
from dane_machine import Computing_machine
from dane_functions import initialize_machines, machines_setup
from dane_procedure import DANE_procedure
import copy
from sklearn.linear_model import SGDClassifier
import math



def run_distSGD_ridgeregression_experiment_2(N, m, max_iter, flag, data, w_opt , mode , max_inner_iter , sampling_flag_rate , rate_param):

	'''we give 0 for data and w_opt if we want to draw them fresh, but
	give them as input if we want to use the same ones and run on different number of machines or different iteration numbers'''

	print m
	
	# setting the objective and DANE parameters:
	objective_param = 0.005
	eta=1.0
	mu=0.000001
	# mu = 0
	DANE_params = [ eta , mu ]

	if flag ==0:

		# generating N 500-d points from  y = <x, w_opt> + noise:
	
		w_opt = np.ones( [ 500, 1 ] )  # line parameters
		
		# distribution for data points:
		mean = np.zeros( [ 500 ] )   
		cov = np.diag( (np.array(range(1, 501))) ** ( -1.2 ) )   # ** (-1.2)

		# draw random data points:
		X = np.random.multivariate_normal(mean, cov, ( N )) 
		# estimate y for x given w:
		Y = np.dot( X , w_opt )   
		
		# adding the noise :
		noise = np.array(np.random.standard_normal( size=( N, 1) ))	
		Y = Y + noise  

		data = np.concatenate(( X , Y ), axis = 1 )
		
		w_opt = np.reshape(w_opt, (500))  # this might be not needed anymore


		# '''better to change it to use the machines rather than directly using ridge-regression class
		#        since we want to have it in general form'''
		# mainrg = Ridge_regression( X, np.reshape(Y, (N)), [0.005] )
		# main_opt_eval = mainrg.eval(w_opt)
		# print 'first main_opt_eval, ', main_opt_eval


	# I am calling initialize_machines to set up our computing machines:
	machines = initialize_machines( m, data )

	'''Running Dane procedure:'''
	evals, runtimes, w_ans , number_of_gradients , number_of_gradients_2 = DANE_procedure( machines ,  w_opt, 'ridge_regression', objective_param , max_iter, DANE_params[0] , DANE_params[1] , mode , max_inner_iter , sampling_flag_rate , rate_param )

	return evals, runtimes, w_ans , w_opt, data , number_of_gradients , number_of_gradients_2


# # think what to set for the w_opt, the optimal of dane or the optimal of sgd!
# def run_sgd_baseline( N, max_iter , data , w_opt ,  )


