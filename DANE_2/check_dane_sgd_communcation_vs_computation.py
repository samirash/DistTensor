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
from distSGD import run_distSGD_ridgeregression_experiment_2 , distSGD_procedure


# hellooooo we are editting this 1 2 3 , 1 2 3

def run_DANE_ridgeregression_experiment_2(N, m, max_iter, flag, data, w_opt , mode , max_inner_iter , sampling_flag_rate , rate_param):

	'''we give 0 for data and w_opt if we want to draw them fresh, but
	give them as input if we want to use the same ones and run on different number of machines or different iteration numbers'''

	print m

	# Hmm 
	
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





''''''''''''''''''''''''''''''''''''
''''''''' main loop: '''''''''''''''
''''''''''''''''''''''''''''''''''''


def ridgeregression_experiment_2_inner_iter( mode , max_inner_iter ):


	max_iter = 20
	optimal_iter = 20


	# experiment_machines_number = [ 4 , 8 ]# [ 4 , 16]
	# experiment_data_size = [6000, 18000, 30000, 10000 , 30000, 50000  ]#  6000, 10000, 14000,

	# sampling_flag_rates = [ [0] ,  [1,6000] , [1,6000] , [0] , [1,10000], [1,10000] ]

	experiment_machines_number = [ 4 ]# [ 4 , 16]
	
	# experiment_data_size = [10000, 10000, 10000, 20000, 20000, 20000  ]#  6000, 10000, 14000,
	# sampling_flag_rates = [ [0] , [1,2000] , [1, 500] ,  [0] ,  [1,4000] , [1, 1000] ]


	# experiment_data_size = [ 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000 ]#  6000, 10000, 14000,

	# sampling_flag_rates = [ [0] , [1,500] , [1, 100] ,  [1, 20] ,  [1,10] , [1, 5] , [1, 2] , [1, 1] ]

	# experiment_data_size = [  10000, 10000, 10000, 10000 ]#  6000, 10000, 14000,

	# sampling_flag_rates = [ [0] , [1, 5] , [1, 2] , [1, 1] ]

	# experiment_data_size = [  20000 ]#  6000, 10000, 14000,

	# sampling_flag_rates = [ [1,1] ]

	experiment_data_size = [ 6000 ]#  6000, 10000, 14000,

	# sampling_flag_rates = [ [0] , [1,500] , [1, 100] ,  [1, 20] ,  [1,10] , [1, 5] , [1, 2] , [1, 1] ]

	# experiment_modes = [ [ 'GD' , 200 ] ]
	# experiment_modes = [ [ 'inverse_exact' , 2000 ] ]
	# experiment_modes = [ [ 'Batch_SGD' , 4 ] ]

	T1 = int( math.ceil(experiment_data_size[0]/ 2. ) )
	T2 = int( math.ceil(experiment_data_size[0]/ 1. ) )
	
	T3 = int( math.ceil(2.0 * experiment_data_size[0] ) )
	T4 = int( math.ceil(4.0 * experiment_data_size[0] ) )
	T5 = int( math.ceil(6.0 * experiment_data_size[0] ) )
	
	
	experiment_modes = [ [ 'SGD' , T1 ]  , [ 'SGD' , T2 ]   , [ 'SGD' , T3 ]  , [ 'SGD' , T4 ]  , [ 'SGD' , T5 ] ] # , [ 'SGD' , T6 ]  , [ 'SGD' , T7 ]  , [ 'SGD' , T8 ]  ]# , [ 'SGD' , T3 ]    , [ 'SGD' , T4 ]  ,  [ 'SGD' , T5 ] ]
	# experiment_modes = [   [ 'linearEq_exact' , 0 ] ,  [ 'SGD' , T3 ]  , [ 'SGD' , T5 ] , [ 'SGD' , T7 ] ]# , [ 'SGD' , T3 ]    , [ 'SGD' , T4 ]  ,  [ 'SGD' , T5 ] ]
	rate_params = [ [0,0], [ 'fix' , 0.01 ] ]#, [ 'fix' , 0.001 ] , [ 'fix' , 0.002 ] ]
	rate_params = [[ 'inverse_t' , 0.15 ]  ]#, [ 'fix' , 0.001 ] , [ 'fix' , 0.002 ] ]
	# rate_params = [ [0,0], [ 'inverse_t' , 0.001 ] , [ 'inverse_t' , 0.01 ] , [ 'inverse_t' , 0.05 ], [ 'inverse_t' , 0.1 ] , [ 'inverse_t' , 0.5 ]  ]   #--> ino edame bede bebin badesh chi mishe!
	# rate_params = [ [0,0], [ 'inverse_t' , 0.5 ] , [ 'inverse_t' , 0.7 ]   ]   #--> ino edame bede bebin badesh chi mishe!
	# # rate_params = [ [ 'inverse_t_sqrt' , 0.5 ] , [ 'inverse_t_sqrt' , 1 ] , [ 'inverse_t_sqrt' , 4 ] ]

	# experiment_modes = [   [ 'GD' , 50 ] , [ 'GD' , 100 ]   [ 'GD' , 200 ] , [ 'GD' , 400 ]  ]
	# rate_params = [  [0,0] , [ 'fix' , 0.8 ]  ]  #   , [ 'fix' , 0.5 ] , [ 'fix' , 0.1 ] # I also has ['fix' , 1] but it was diverging with every number of GD steps/ 0.8 was the best
	# # rate_params = [ [0,0] , [ 'inverse_t' , 1 ], [ 'inverse_t' , 0.8 ], [ 'inverse_t' , 0.5 ]  ]#, [ 'inverse_t' , 4 ]  ]
	# # rate_params = [ [ 'inverse_t_sqrt' , 1 ] , [ 'inverse_t' , 2 ] , [ 'inverse_t_sqrt' , 4 ] ]

	
	# these are the meain ones:
	# experiment_modes = [ [ 'Batch_SGD' , 100 ] , [ 'Batch_SGD' , 200 ] , [ 'Batch_SGD' , 400 ] , [ 'Batch_SGD' , 800 ] ]
	# rate_params = [   [ 'fix' , 0.05 , 100 ] ,  [ 'fix' , 0.01 , 100 ] ,  [ 'fix' , 0.005 , 100 ] ,  [ 'fix' , 0.05 , 200 ] ,  [ 'fix' , 0.01 , 200 ] ,  [ 'fix' , 0.005 , 200 ] ]  #   , [ 'fix' , 0.5 , 100 ] , [ 'fix' , 0.1 , 100 ]    (exp 14.png)

	# experiment_modes = [ [ 'Batch_SGD' , 100 ] , [ 'Batch_SGD' , 200 ] , [ 'Batch_SGD' , 400 ] , [ 'Batch_SGD' , 800 ] ]
	# rate_params = [   [ 'inverse_t' , 0.1 , 100 ] ,[ 'inverse_t' , 0.05 , 100 ] ,  [ 'inverse_t' , 0.01 , 100 ] ,  [ 'inverse_t' , 0.1 , 200 ] ,  [ 'inverse_t' , 0.05 , 200 ] ,  [ 'inverse_t' , 0.01 , 200 ] ]  #   , [ 'fix' , 0.5 , 100 ] , [ 'fix' , 0.1 , 100 ]    (exp 15.png)


	# experiment_modes = [ [ 'Batch_SGD' , 400 ] , [ 'Batch_SGD' , 800 ] ]
	# rate_params = [   [ 'inverse_t' , 0.1 , 100 ] , [ 'inverse_t' , 0.2 , 100 ] , [ 'inverse_t' , 0.3 , 100 ]  ]  #   , [ 'fix' , 0.5 , 100 ] , [ 'fix' , 0.1 , 100 ]    (exp 16.png)

	# these are the final results I am keeping:
	# experiment_modes = [ [ 'Batch_SGD' , 800 ] , [ 'Batch_SGD' , 1600 ] , [ 'Batch_SGD' , 3200 ] ]
	# rate_params = [   [ 'inverse_t' , 1.0 , 100 ]  , [ 'inverse_t' , 1.5 , 100 ] ,  [ 'inverse_t' , 2.0 , 100 ]  ,  [ 'inverse_t' , 2.5 , 100 ]  ]  #   , [ 'fix' , 0.5 , 100 ] , [ 'fix' , 0.1 , 100 ]    (exp 17.png)

	# plot_colors = [ 'b' , 'g' , 'r' , 'c' , 'm' , 'y' , 'k' , 'purple' ]



	experiment_size =  len(experiment_machines_number) * len(experiment_data_size)

	# NOTICE:    I change this for the new experiemnts! be careful with it
	experiment_size = len( experiment_modes ) * len( rate_params ) 
	print experiment_size

	m=4
	all_suboptimalities = np.zeros((max_iter+1 , experiment_size))
	all_evals = np.zeros((max_iter+1 , experiment_size))
	all_runtimes = np.zeros((max_iter+1 , experiment_size))
	all_optimals = np.zeros((1 , experiment_size))

	all_gradient_counts = np.zeros((max_iter+1 , experiment_size))
	all_gradient_counts_2 = np.zeros((max_iter+1 , experiment_size))

	# all_suboptimalities = np.zeros((max_iter , 6))

	# all_evals = np.zeros((max_iter , 6))
	# all_runtimes = np.zeros((max_iter , 6))
	# all_optimals = np.zeros((1 , 6))


	i = 0
	# for m in experiment_machines_number: # m is the number of machines,  4, 16
	# 	for j in range(len(experiment_data_size)):  # N is the number of points
	for k in range(len(experiment_modes)): # m is the number of machines,  4, 16
		for j in range(len(rate_params)):  # N is the number of points
			
			print 'New experiment round:'
			print 'i:', i
			print 'j:', j

			# N = experiment_data_size[j]
			N = experiment_data_size[0]
			rate_param = rate_params[j]


			
			# # sampling_flag_rates_copy = sampling_flag_rates[:]
			# sampling_flag_rate = sampling_flag_rates[j][:]

			# print sampling_flag_rate

			# if sampling_flag_rate[0] == 1:
			# 	sampling_flag_rate[1] = sampling_flag_rate[1] * m

			# print sampling_flag_rate
			# print 'now check here:', sampling_flag_rates
			# # print 'and this:' , sampling_flag_rates_copy
			# print ' m, N , sampling_flag_rate =',m ,N, sampling_flag_rate

			sampling_flag_rate = [0]

			mode_raw = experiment_modes[k][:]

			# NOTICE: added this
			mode = mode_raw[0]

			if rate_param[0] == 0:
				print 'I AM HERE....................!!!!!!!!!! ......................'
				mode = 'linearEq_exact'			

			if mode == 'linearEq_exact':
				max_inner_iter = 0
			else:
				max_inner_iter = mode_raw[1]

			m = experiment_machines_number[0]






			print ' m, N , sampling_flag_rate , mode , max_inner_iter , rate_param =',m ,N, sampling_flag_rate, mode, max_inner_iter , rate_param

			# this is supposed to provide the optimals obtainable on one machine, in order to compute suboptimalities
			print '	HEEEEEEEEEEREEEEEEEEEE: .............. (1)'
			evals_0, runtimes_0, w_ans_0 , w_opt_0, data_0 , number_of_gradients_0 , number_of_gradients_0_2 = run_DANE_ridgeregression_experiment_2( N , 1 , optimal_iter , 0 , 0 , 0 , 'linearEq_exact'  , max_inner_iter , [0], rate_param )# drawing fresh data --> flag=0 , data is not given -> data = 0 , we don't have w_opt to give -> w_opt = 0

			minimum_dane_f = evals_0[ max(np.nonzero(evals_0))[1]]
			# min(eval_0[np.nonzero(evals_0)])  	# this does not work if I leave 0 for below some threshold
			# minimum_dane_f = 0

			# print '	HEEEEEEEEEEREEEEEEEEEE: .............. (2)'
			# # running on one machines, with total/m amount of data:
			# # max_inner_iter_ = max_inner_iter/m
			# # evals, runtimes, w_ans , w_opt, data , number_of_gradients , number_of_gradients_2 = run_DANE_ridgeregression_experiment_2( N/m , 1 , max_iter , 1 , data_0 , w_opt_0 , mode , max_inner_iter_ , sampling_flag_rate ,  rate_param ) # flag=1, use the same data, w
			# evals, runtimes, w_ans , w_opt, data , number_of_gradients , number_of_gradients_2 = run_distSGD_ridgeregression_experiment_2( N/m , 1 , max_iter , 1 , data_0 , w_opt_0 , mode , max_inner_iter/m , sampling_flag_rate ,  rate_param ) # flag=1, use the same data, w

			# print 'look at here to check the results:'
			# print evals
			# print number_of_gradients

			# all_evals[0: max_iter+1,i] = evals
			# all_runtimes[0: max_iter+1,i] = runtimes
			# all_gradient_counts[0: max_iter+1,i] = number_of_gradients
			# all_gradient_counts_2[0: max_iter+1,i] = number_of_gradients_2

			# all_optimals[0,i] = minimum_dane_f

			# i = i + 1
			print '	HEEEEEEEEEEREEEEEEEEEE: .............. (3)'
			# running on all machines- main dane:
			max_inner_iter_ = max_inner_iter/m
			evals, runtimes, w_ans , w_opt, data , number_of_gradients , number_of_gradients_2 = run_DANE_ridgeregression_experiment_2( N , m , max_iter , 1 , data_0 , w_opt_0 , mode , max_inner_iter_ , sampling_flag_rate ,  rate_param ) # flag=1, use the same data, w

			all_evals[:,i] = evals
			all_runtimes[:,i] = runtimes
			all_gradient_counts[:,i] = number_of_gradients
			all_gradient_counts_2[:,i] = number_of_gradients_2

			all_optimals[0,i] = minimum_dane_f

			i = i + 1

			# # running on one machine with all data:
			# # max_inner_iter_ = max_inner_iter/m
			# # evals, runtimes, w_ans , w_opt, data , number_of_gradients , number_of_gradients_2 = run_DANE_ridgeregression_experiment_2( N , 1 , max_iter , 1 , data_0 , w_opt_0 , mode , max_inner_iter , sampling_flag_rate ,  rate_param ) # flag=1, use the same data, w
			# evals, runtimes, w_ans , w_opt, data , number_of_gradients , number_of_gradients_2 = run_distSGD_ridgeregression_experiment_2( N , 1 , max_iter*m , 1 , data_0 , w_opt_0 , mode , max_inner_iter/m , sampling_flag_rate ,  rate_param ) # flag=1, use the same data, w
			
			# print 'look at here to check the results:'
			# print evals
			# print number_of_gradients


			# all_evals[:,i] = evals
			# all_runtimes[:,i] = runtimes
			# all_gradient_counts[:,i] = number_of_gradients
			# all_gradient_counts_2[:,i] = number_of_gradients_2

			# all_optimals[0,i] = minimum_dane_f

			# i = i + 1

			# # running on one machine with all data, but as if it was on all machines (ideal distributed case):
			# all_evals[:,i] = evals
			# all_runtimes[:,i] = runtimes
			# all_gradient_counts[:,i] = number_of_gradients
			# all_gradient_counts_2[:,i] = number_of_gradients_2

			# all_optimals[0,i] = minimum_dane_f

			# i = i + 1


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
	# print 'all_optimals'
	# print all_optimals
	# print np.repeat(all_optimals, max_iter, axis=0)

	all_suboptimalities = all_evals - np.repeat(all_optimals, max_iter + 1, axis=0)
	print 'all_suboptimalities:'
	print all_suboptimalities
	print type(all_suboptimalities[0,0])

	all_suboptimalities = all_suboptimalities.clip(0.00000000000001)
	# print 'all_suboptimalities:'
	# print all_suboptimalities

	# all_suboptimalities = np.log10(all_suboptimalities)
	

	# def save_results( all_evals , all_optimals , all_suboptimalities , all_runtimes  ,all_gradient_counts,  all_gradient_counts_2 ,  max_iter ):
	# 	save_str = np.str('all_results_' + str(time.localtime()[2]) + '-' + str(time.localtime()[1]) + '-' + str(time.localtime()[0]) + '-' + str(time.localtime()[3]) + '-' + str(time.localtime()[4]) )
	# 	# outfile = TemporaryFile()
	# 	# np.savez(outfile, **{x_name: x, y_name: y})
	# 	# outfile.seek(0)
	# 	# npzfile = np.load(outfile)
	# 	# npzfile.files
	# 	np.savez( save_str , 'evals' = all_evals , 'optimals' =  all_optimals , 'suboptimalities' = all_suboptimalities , 'runtimes' = all_runtimes, 'grad_counts'= all_gradient_counts, 'grad_counts_2' = all_gradient_counts_2 , 'max_iter' = max_iter )


	def draw_plots(all_evals , all_optimals , all_suboptimalities , all_runtimes, all_gradient_counts , all_gradient_counts_2 , a, b , max_iter , versus_time_flag , m):

		all_suboptimalities = np.log10(all_suboptimalities)
		all_suboptimalities = all_suboptimalities.clip(-3)
		m=4
		t = np.arange( max_iter+1 )

		plot_colors = [ 'b' , 'g' , 'r' , 'c' , 'm' , 'y' , 'k' , 'purple' ]
		# g: green
		# r: red
		# c: cyan
		# m: magenta
		# y: yellow
		# k: black
		# w: white

		print 'Is PLOTTING now! .... ..... ......'
		# for i in range(a):

		# 	for j in range(b):
		# 		plt.plot(t ,all_suboptimalities[ : , i*b + j ], plot_colors[ j ] )
		# 		print all_suboptimalities[ : , i*b + j ]
		# 	plt.show()
		# 	# for j in range(b):
		# 	# 	plt.plot(all_runtimes[ : , i*b + j ],all_suboptimalities[ : , i*b + j ] , plot_colors[ j ] )
		# 	# plt.show()

		# for j in range(b):


		# LATEST CHANGES: COMMENTING THIS AS MAIN ITERATIONS DOES NOT MAKE MUCH SENCE NOW:
		# for i in range(a):
		# 	# plt.plot(t ,all_suboptimalities[ : , i*b + j ], plot_colors[ i ] )
		# 	# print all_suboptimalities[ : , i*b + j ]
		# 	print i
		# 	# print b
		# 	# print j
		# 	# print i*b + j

		# 	plt.plot(t , all_suboptimalities[ : , i*4  ], color=plot_colors[ 0] , marker='o', label='single_machine_total/m_data')
			
		# 	plt.plot(t , all_suboptimalities[ : , i*4  +1], color=plot_colors[ 1 ] ,marker ='v', label='all_machines_all_data_dane')
			
		# 	plt.plot(t , all_suboptimalities[ : , i*4  +2], color=plot_colors[ 2 ]  ,marker = '+' , label='single_machine_all_data')
		
		# 	plt.plot(t , all_suboptimalities[ : , i*4  +3], color=plot_colors[ 3 ]  ,marker = '*', label='single_machine_all_data_devided_by_m')
			
		# 	plt.xlabel('iteration#')
		# 	plt.ylabel('suboptimalities')
		# 	plt.legend()
		# 	plt.title(('suboptimality vs. dane_iterations'))

		# 	plt.show()
		# 	# for j in range(b):
		# 	# 	plt.plot(all_runtimes[ : , i*b + j ],all_suboptimalities[ : , i*b + j ] , plot_colors[ j ] )
		# 	# plt.show()

		if versus_time_flag == 1:

			print  'Is PLOTTING versus TIME now! .... ..... ......'

			# for i in range(a):
			# 	# plt.plot(all_runtimes[ : , i*b + j ] , all_suboptimalities[ : , i*b + j ], plot_colors[ i ] , )
			# 	# print all_runtimes[ : , i*b + j ]

			# 	plt.plot(all_runtimes[ : , i*4  ] , all_suboptimalities[ : , i*4 ], plot_colors[ 0 ] , marker = 'o' , label='single_machine_total/m_data')
				
			# 	plt.plot(all_runtimes[ : , i*4  +1] , all_suboptimalities[ : , i*4 +1], plot_colors[ 1 ]  , marker = 'v' , label='all_machines_all_data_dane')
				
			# 	plt.plot(all_runtimes[ : , i*4  +2] , all_suboptimalities[ : , i*4 +2], plot_colors[ 2 ] ,marker = '+' ,  label='single_machine_all_data')
				
			# 	plt.plot(all_runtimes[ : , i*4  +3]/m , all_suboptimalities[ : , i*4 +3], plot_colors[ 3 ]  ,marker = '*' ,  label='single_machine_all_data_devided_by_m')
			

			# 	plt.xlabel('run-time')
			# 	plt.ylabel('suboptimalities')
			# 	plt.legend()
			# 	plt.title(('suboptimality vs. run_times'))


			# 	plt.show()
			# 	# for j in range(b):
			# 	# 	plt.plot(all_runtimes[ : , i*b + j ],all_suboptimalities[ : , i*b + j ] , plot_colors[ j ] )
			# 	# plt.show()
			print 'all_gradient_counts' , all_gradient_counts
			grad_counts = np.cumsum(all_gradient_counts , axis = 0 )
			print 'grad_counts' , np.shape(grad_counts)

			# print 'all_gradient_counts_2' , all_gradient_counts_2
			# grad_counts_2 = np.cumsum(all_gradient_counts_2 , axis = 0 )
			# print 'grad_counts_2' , np.shape(grad_counts_2)

			# for j in range(b):
			for i in range(a):
				# plt.plot(all_runtimes[ : , i*b + j ] , all_suboptimalities[ : , i*b + j ], plot_colors[ i ] )
				# print all_runtimes[ : , i*b + j ]

				# crop_point = -1.6;

				ax = plt.subplot(111)
				# ax.set_ylim([crop_point, 1.0])
				ax.set_xlim([0.0, 20000.0])

				# plt.plot(grad_counts[ 0: max_iter+1 , i*4 ] , all_suboptimalities[ 0: max_iter+1 , i*4 ], plot_colors[ 0 ] ,marker = 'o' ,  label='single_machine_total/m_data')
				
				plt.plot(grad_counts[ 0: max_iter+1 , i] , all_suboptimalities[ 0: max_iter+1 , i ], plot_colors[ i ] ,marker = '+' ,  label=np.str('all_machines_all_data_dane___num_iters = ' + str(i) )  )

				
				# plt.plot(grad_counts[ : , i*4 +2] , all_suboptimalities[ : , i*4 +2], plot_colors[ 2 ] ,marker = '+' ,  label='single_machine_all_data')
				
				# plt.plot(grad_counts[ : , i*4 +3]/m , all_suboptimalities[ : , i*4 +3], plot_colors[ 3 ]  , marker = '*' , label='single_machine_all_data_devided_by_m')

			plt.plot(grad_counts[ 0: max_iter+1 , i] , -1.5*np.ones( [max_iter+1 , 1]), 'k' ,marker = '.' )
			plt.xlabel('grad_counts')
			plt.ylabel('suboptimalities')
			plt.legend()
			plt.title(('suboptimality vs. grad-count'))



			plt.show()

			# for j in range(b):
			# 	for i in range(a):
			# 		# plt.plot(all_runtimes[ : , i*b + j ] , all_suboptimalities[ : , i*b + j ], plot_colors[ i ] )
			# 		# print all_runtimes[ : , i*b + j ]
			# 		plt.plot(grad_counts_2[ : , i*b + j ] , all_suboptimalities[ : , i*b + j ], plot_colors[ i ] )
			# 		print grad_counts_2[ : , i*b + j ]
			# 	plt.show()



	# NOTICE: I am changing this for the new experiments:
	a = len( experiment_modes )
	b = len( rate_params )

	# save_results( all_evals , all_optimals , all_suboptimalities , all_runtimes  ,all_gradient_counts,  all_gradient_counts_2 ,  max_iter)
	draw_plots(all_evals , all_optimals , all_suboptimalities , all_runtimes, all_gradient_counts , all_gradient_counts_2 , a, b , max_iter , 1 , experiment_machines_number[0])


mode = 'inverse_exact'  # using matrix pseudoinvers
# mode = 'linearEq_exact'  # using np.linalg.lstsq
# mode = 'linearEq_inexact'  # using sparsela.lsmr
# mode = 'SGD'  # using SGD
max_inner_iter = 500   # for SGD I am feeding this parameter inside again


ridgeregression_experiment_2_inner_iter( mode , max_inner_iter )


# % python -mtimeit  "l=[]"
