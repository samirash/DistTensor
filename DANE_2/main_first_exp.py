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


def run_DANE_ridgeregression_experiment_2(N, m, max_iter, flag, data, w_opt , mode , max_inner_iter , sampling_flag_rate ):

	'''we give 0 for data and w_opt if we want to draw them fresh, but
	give them as input if we want to use the same ones and run on different number of machines or different iteration numbers'''

	print m
	
	# setting the objective and DANE parameters:
	objective_param = 0.005
	eta=1.0
	mu=0.00000001
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
	evals, runtimes, w_ans = DANE_procedure( machines ,  w_opt, 'ridge_regression', objective_param , max_iter, DANE_params[0] , DANE_params[1] , mode , max_inner_iter , sampling_flag_rate )

	return evals, runtimes, w_ans , w_opt, data





def run_DANE_smoothhingeclassification_experiment_2(N, m, max_iter, flag, data, w_opt , mode , max_inner_iter , sampling_flag_rate ):

	'''we give 0 for data and w_opt if we want to draw them fresh, but
	give them as input if we want to use the same ones and run on different number of machines or different iteration numbers'''

	print m
	
	# setting the objective and DANE parameters:
	objective_param = 0.005
	eta=1.0
	mu=0.00000001
	DANE_params = [ eta , mu ]


	if flag ==0:

		# generating N 500-d points from  y = <x, w_opt> + noise:
	
		w_opt = np.ones( [ 500, 1 ] )  # line parameters
		
		# distribution for data points:
		mean = np.zeros( [ 500 ] )   
		cov = np.diag( (np.array(range(1, 501))) ** ( -1.2 ) )

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
	evals, runtimes, w_ans = DANE_procedure( machines ,  w_opt, 'ridge_regression', objective_param , max_iter, DANE_params[0] , DANE_params[1] , mode , max_inner_iter , sampling_flag_rate )

	return evals, runtimes, w_ans , w_opt, data


''''''''''''''''''''''''''''''''''''
''''''''' main loop: '''''''''''''''
''''''''''''''''''''''''''''''''''''


def ridgeregression_experiment_2_inner_iter( mode , max_inner_iter ):


	max_iter = 20
	optimal_iter = 20


	# experiment_machines_number = [ 4 , 8 ]# [ 4 , 16]
	# experiment_data_size = [6000, 18000, 30000, 10000 , 30000, 50000  ]#  6000, 10000, 14000,

	# sampling_flag_rates = [ [0] ,  [1,6000] , [1,6000] , [0] , [1,10000], [1,10000] ]

	experiment_machines_number = [ 4 , 16 ]# [ 4 , 16]
	
	# experiment_data_size = [10000, 10000, 10000, 20000, 20000, 20000  ]#  6000, 10000, 14000,
	# sampling_flag_rates = [ [0] , [1,2000] , [1, 500] ,  [0] ,  [1,4000] , [1, 1000] ]


	experiment_data_size = [ 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000 ]#  6000, 10000, 14000,

	sampling_flag_rates = [ [0] , [1,500] , [1, 100] ,  [1, 20] ,  [1,10] , [1, 5] , [1, 2] , [1, 1] ]

	# experiment_data_size = [  10000, 10000, 10000, 10000 ]#  6000, 10000, 14000,

	# sampling_flag_rates = [ [0] , [1, 5] , [1, 2] , [1, 1] ]

	# experiment_data_size = [  20000 ]#  6000, 10000, 14000,

	# sampling_flag_rates = [ [1,1] ]

	experiment_size =  len(experiment_machines_number) * len(experiment_data_size)
	print experiment_size


	all_suboptimalities = np.zeros((max_iter , experiment_size))
	all_evals = np.zeros((max_iter , experiment_size))
	all_runtimes = np.zeros((max_iter , experiment_size))
	all_optimals = np.zeros((1 , experiment_size))

	# all_suboptimalities = np.zeros((max_iter , 6))

	# all_evals = np.zeros((max_iter , 6))
	# all_runtimes = np.zeros((max_iter , 6))
	# all_optimals = np.zeros((1 , 6))


	i = 0
	for m in experiment_machines_number: # m is the number of machines,  4, 16
		for j in range(len(experiment_data_size)):  # N is the number of points
			
			print 'New experiment round:'
			print 'i:', i
			print 'j:', j

			N = experiment_data_size[j]
			
			# sampling_flag_rates_copy = sampling_flag_rates[:]

			sampling_flag_rate = sampling_flag_rates[j][:]

			print sampling_flag_rate

			if sampling_flag_rate[0] == 1:
				sampling_flag_rate[1] = sampling_flag_rate[1] * m

			print sampling_flag_rate
			print 'now check here:', sampling_flag_rates
			# print 'and this:' , sampling_flag_rates_copy
			print ' m, N , sampling_flag_rate =',m ,N, sampling_flag_rate

			# this is supposed to provide the optimals obtainable on one machine, in order to compute suboptimalities
			evals_0, runtimes_0, w_ans_0 , w_opt_0, data_0 = run_DANE_ridgeregression_experiment_2(N, 1, optimal_iter, 0, 0, 0 , 'inverse_exact'  , max_inner_iter , [0] )# drawing fresh data --> flag=0 , data is not given -> data = 0 , we don't have w_opt to give -> w_opt = 0

			minimum_dane_f = evals_0[ max(np.nonzero(evals_0))[1]]
			# min(eval_0[np.nonzero(evals_0)])  	# this does not work if I leave 0 for below some threshold
			# minimum_dane_f = 0

			# running on multiple machines:
			evals, runtimes, w_ans , w_opt, data = run_DANE_ridgeregression_experiment_2(N, m, max_iter, 1, data_0, w_opt_0 , mode , max_inner_iter , sampling_flag_rate) # flag=1, use the same data, w

			all_evals[:,i] = evals
			all_runtimes[:,i] = runtimes

			all_optimals[0,i] = minimum_dane_f

			i = i + 1

			print 'OK! norm_ of _w_opt (original w we made data with) *** , ', np.dot(w_opt,w_opt)  # the algorithm finds w with smaller w (regularization term)

			print 'norm_ of _w_ans_single_machine, ', np.dot(w_ans_0 , w_ans_0)
			print 'norm_ of _w_ans_multiple_machines *** , ', np.dot(w_ans,w_ans)
			
			print 'minimum_(single machine) ', minimum_dane_f
			print 'minimum_found (multiple machines) ', evals[ max(np.nonzero(evals))[0]]

			# print 'w_ans_0', w_ans_0
			# print 'w_ans', w_ans
			# print '*******'

	print 'all_evals:'
	print all_evals
	print 'all_optimals'
	print all_optimals
	print np.repeat(all_optimals, max_iter, axis=0)

	all_suboptimalities = all_evals - np.repeat(all_optimals, max_iter, axis=0)
	print 'all_suboptimalities:'
	print all_suboptimalities
	print type(all_suboptimalities[0,0])

	all_suboptimalities = all_suboptimalities.clip(0.00000000000001)
	print 'all_suboptimalities:'
	print all_suboptimalities

	# all_suboptimalities = np.log10(all_suboptimalities)
	t = np.arange(max_iter)
	print 't', t


	def save_results( all_evals , all_optimals , all_suboptimalities , all_runtimes ):
		np.save('samiraresults', [all_evals , all_optimals , all_suboptimalities , all_runtimes])


	def draw_plots(all_evals , all_optimals , all_suboptimalities , all_runtimes, experiment_machines_number, experiment_data_size , max_iter ):

		all_suboptimalities = np.log10(all_suboptimalities)
		all_suboptimalities = all_suboptimalities.clip(-6)

		t = np.arange(max_iter)

		plot_colors = [ 'b' , 'g' , 'r' , 'c' , 'm' , 'y' , 'k' , 'purple' ]
		# g: green
		# r: red
		# c: cyan
		# m: magenta
		# y: yellow
		# k: black
		# w: white

		a = len(experiment_machines_number)
		b = len(experiment_data_size)
		print 'IS PRINTING NOW .... ..... ......'
		for i in range(a):

			for j in range(b):
				plt.plot(t,all_suboptimalities[ : , i*b + j ], plot_colors[ j ] )
				print all_suboptimalities[ : , i*b + j ]
			plt.show()
			# for j in range(b):
			# 	plt.plot(all_runtimes[ : , i*b + j ],all_suboptimalities[ : , i*b + j ] , plot_colors[ j ] )
			# plt.show()

	save_results( all_evals , all_optimals , all_suboptimalities , all_runtimes )
	draw_plots(all_evals , all_optimals , all_suboptimalities , all_runtimes, experiment_machines_number, experiment_data_size , max_iter )



# mode = 'limit_iters' #  either 'exact' or 'limit_iters'
# # exact: using np.linalg.lstsq
# # limit_iters: using sparsela.lsmr
# # for limit_iters, I want to add other modes like sgd and svrg and mini_batch SGD 

mode = 'inverse_exact'  # using matrix pseudoinvers
mode = 'linearEq_exact'  # using np.linalg.lstsq
mode = 'linearEq_inexact'  # using sparsela.lsmr
# mode = 'SGD'  # using SGD
mode = 'inverse_exact'
max_inner_iter = 0

ridgeregression_experiment_2_inner_iter( mode , max_inner_iter )


# % python -mtimeit  "l=[]"
