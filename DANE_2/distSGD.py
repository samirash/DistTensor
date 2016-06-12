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

	# here it's simple not dane and I do not need these dane parameters:
	# eta=1.0
	# mu=0.000001
	# mu = 0
	# DANE_params = [ eta , mu ]

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

	'''Running distSGD procedure:'''
	evals, runtimes, w_ans , number_of_gradients , number_of_gradients_2 = distSGD_procedure( machines ,  w_opt, 'ridge_regression', objective_param , max_iter,  mode , max_inner_iter , sampling_flag_rate , rate_param )

	return evals, runtimes, w_ans , w_opt, data , number_of_gradients , number_of_gradients_2


# # think what to set for the w_opt, the optimal of dane or the optimal of sgd!
# def run_sgd_baseline( N, max_iter , data , w_opt ,  )




def distSGD_procedure(machines,  w_opt, objective_form, objective_param, max_iter,  mode , max_inner_iter , sampling_flag_rate , rate_param ):  # check the experiments to see what \mu needs to be is it what I have in machines_setup function above?

	# The main DANE procedure given the machines already with their data points
	# if mu=0 does not converge then you shall use 0.3*lambda where the function is lambda-strong convex

	# 	 eta_val: the value of mu used in Eq.13 in DANE paper. The factor for the global gradent of w.
	# sampling_flag_rate is either a list containing only 0 (no sampling) ot a list of 2 elements, containig 1 (sampling) and the number of points needed to be sampled:
	# either [0] or [0, N_sample]

	machines_setup( machines, w_opt, objective_form, objective_param, "dist_SGD") 


	m = len(machines)
	w_length = machines[0].w_length

	sampling_flag = sampling_flag_rate[0]
	if sampling_flag == 1:
		N_sample = sampling_flag_rate[1]
		N_sample = N_sample/m

	''' Initializing global and local weights and gradients with 0 matrices (or vectors): '''
	w_global = np.zeros(w_length)
	grad_global = np.zeros(w_length)
	eval_global = 0

	local_gradients = np.zeros((w_length, m))
	local_evals = np.zeros(m)
	local_ws = np.zeros((w_length, m))

	eval_diffs = np.zeros(max_iter)  # might want to remove this
	suboptimalities = np.zeros(max_iter+1)
	evals = np.zeros(max_iter+1)

	runtimes = []
	
	start_time = time.time()
	
	num_of_gradients_list = []
	num_of_gradients_list_2 = []


	runtimes.append( 0 )
	num_of_gradients_list.append( 0 )
	num_of_gradients_list_2.append( 0 )


	''' Defining functions used in the main loop of distSGD: '''

	def sample_data_locally( machines , N_sample ):
		for i in range(m):
			machines[i].sample_data( N_sample )
			machines[i].updata_objective_data
		
	def compute_local_gradients(machines):
		# computes all local gradients
		for i in range(m):
			a , b = machines[i].compute_local_grad_and_eval()
			print 'checking the shape ...' , np.shape(a) , np.shape(b)
			local_gradients[:,i], local_evals[i] = machines[i].compute_local_grad_and_eval()
			print local_evals[i], 'here dear!'
		

	def compute_grad_global(local_gradients):
		# computes global grad as the average of the local gradients
		grad_global = np.mean(local_gradients, axis=1)
		# test! : check all the dimensions
		return grad_global

	def distribute_grad_global(machines, grad_global):
		# distributed the value of the global gradient to all machines
		for i in range(m):
			machines[i].update_grad_global_copy(grad_global)

	# this can become perfom_simple_local_optimizations
	def perform_SGD_local_optimizations(machines, grad_global , mode , max_inner_iter , rate_param , dane_iter_number  , initial_iter):
		''' test!: # we do not actually need to pass this grad_global here, but is it better to use this and totally remove distribute_grad_global ?'''
		# computes all local optimims which are essentially local w's
		max_number_of_gradients = 0
		total_number_of_gradients = 0
		for i in range(m):
			print 'PERFORM LOCAL OPTIMIZATION (simple optimize -- SGD ) FOR       ..............    MACHINE NUMBER, ', i
			# here I am using SGD for the mode since I am just adding the simple distributed optimization with distributed SGD,
			# but can be extended to other approaches and the I should pass mode here!
			temp , number_of_gradients = machines[i].simple_local_optimization(grad_global , 'SGD' , max_inner_iter , rate_param , dane_iter_number , initial_iter )
			local_ws[:,i]  = np.reshape( temp , (-1,) )
			print 'max_number_of_gradients ------>  ' , max_number_of_gradients
			print 'total_number_of_gradients ------>  ' , total_number_of_gradients
			max_number_of_gradients = max( max_number_of_gradients , number_of_gradients )
			total_number_of_gradients = total_number_of_gradients + number_of_gradients 
			
		
		print 'max_number_of_gradients' , max_number_of_gradients
		print 'total_number_of_gradients ------>  ' , total_number_of_gradients
		num_of_gradients_list.append( max_number_of_gradients )
		num_of_gradients_list_2.append( total_number_of_gradients )




	def compute_w_global(local_ws):
		# computes global w as the average of all local w's
		w_global = np.mean(local_ws, axis=1)
		return w_global

	def distribute_w_global(machines, w_global):
		'''distributes w_global to all machines and sets their w to w_global '''
		for i in range(m):
			print i
			machines[i].update_w_loc(w_global)
			eval_machines_i = machines[i].compute_eval(w_global)
			print '******     eval_machines_i ',  eval_machines_i

	def compute_eval_global( w_global ):
		# taking means of local evaluations, or compute the evaluation on all data
		# means of local evaluations:
		print np.shape(local_evals)
		if np.shape(local_evals)[0] == 1:
			eval_global = local_evals[0]
		else:
			eval_global = np.mean(local_evals,axis = 0)

		print 'eval_global ', eval_global, ' hello'
		return eval_global


	''' Main loop of the distSGD Algorithm: '''

	eval_global = compute_eval_global( w_global )
	eval_pred = eval_global
	eval_diff = eval_global

	initial_iter = 0


	for t in range(max_iter):

		#	print eval_diff

		# if eval_diff < 0.000000000001:
		# 	print 'small difference'
		# 	return evals, w_global, eval_diffs, suboptimalities

		# pre step for sampling data if it the flag is on
		print 'MAIN distSGD ITERATION   ...............................   NUMBER  ', t
		if sampling_flag == 1:
			sample_data_locally( machines , N_sample )
		# step 1
		compute_local_gradients( machines )
		# step 2
		grad_global = compute_grad_global( local_gradients )

		# evaluation for current iteration
		eval_global = compute_eval_global( w_global )
		evals[t] = eval_global


		# step 3
		distribute_grad_global( machines, grad_global )
		# step 4
		perform_SGD_local_optimizations( machines, grad_global , mode , max_inner_iter , rate_param , t+1 , initial_iter)
		initial_iter = initial_iter + max_inner_iter
		# step 5
		w_global = compute_w_global( local_ws )
		# step 6
		distribute_w_global( machines, w_global )


		runtimes.append( time.time() - start_time )

		eval_diff = eval_pred - eval_global    # might want to remove this
		print 'eval_diff, ', eval_diff
		eval_pred = eval_global


	# evaluation for current iteration
	eval_global = compute_eval_global( w_global )
	evals[t+1] = eval_global


	return evals, runtimes, w_global , num_of_gradients_list , num_of_gradients_list_2




