import numpy as np
import scipy.io as io
import scipy.sparse as sp
import scipy.linalg as la
from general_function_class import Ridge_regression
from general_function_class import Classification_smooth_hinge
import random


class Computing_machine:
	'''
	Computing_machine class stores what we need for each computing machine: each machine has a set of datapoints
	and stores it's computed parameters in vector w_loc (loc suggests that this is local to the machine)
	
		w_length:	dimension of the parameter vector (d)
		w_loc:		contains the parameters for the ridge regression (d-dimensional) on this specific machine (hence it's w_loc or local w)
	       		-> w_loc is initialized to 0
		data: 		m*(d+1) dimensional matrix. Training data on this particular machine. With d being the dimension of each datapoint
				and m the number of those training data
	'''

	# Computing_machine should have access to function_classes like Ridge_regression
	
	def __init__(self, id, w_length):

		self.id = id
		self.w_length = w_length
		self.data = np.zeros(w_length+1)
		self.x = np.zeros(w_length)
		self.y = np.zeros(1)

		self.w_loc = 1.0 * np.zeros(w_length)
		# self.w_loc = 1.0 * np.ones(w_length)
		self.grad_global_copy = np.zeros(w_length)
		self.grad_local = np.zeros(w_length)




	def get_data( self , data ):
		# data should be given as a numpy array 
		self.full_machine_data = data
		self.full_machine_x = data[:,0:-1]
		self.full_machine_y = data[:,-1]
		self.full_machine_n = np.shape(self.full_machine_y)[0]
		print 'peinting1: allocating data to the machines , ' , np.shape(self.full_machine_y), np.shape(self.full_machine_x),np.shape(self.full_machine_data)
		print self.full_machine_n

		self.data = data
		self.x = data[:,0:-1]
		self.y = data[:,-1]
		self.n = np.shape(self.y)[0]
		print 'peinting :  allocating data to the machines ,  ' , np.shape(self.y), np.shape(self.x),np.shape(self.data)
		print self.n
		#print self.id, np.shape(data)

	
	def add_data( self , data ):  # this is not up to date and I am not using it at all
		self.data = np.concatenate(self.data, data)
		self.x = np.concatenate(self.x, data[:,0:-1])
		self.y = np.concatenate(self.y, data[:,-1])
		self.n = np.shape(self.y)[0]


	def sample_data( self  , N_sample ):
		# samples N_sample points from the data stored on the machine and works with that data
		print self.full_machine_n
		print N_sample
		ind = random.sample( xrange( self.full_machine_n ), N_sample )
		#print np.shape(ind)
		ind_ = np.reshape(ind , (len(ind) , 1))
		#print np.shape(ind_)
		#print np.shape(self.full_machine_data)
		self.data = self.full_machine_data[ind , : ]
		# np.array(np.reshape( ind , (len(ind) , 1)))
		self.x = self.data[:,0:-1]
		self.y = self.data[:,-1]
		self.n = np.shape(self.y)[0]
		print self.n

 
	def set_optimization_algorithm( self, algorithm, *param ):
		self.opt_alg = algorithm			# this optimization algorithm can be DANE for instance or ADMM  or "dist_SGD"
		if len(param) >= 1:
			self.opt_alg_param  = param		# if the algorithm is DANE for instance, param would be [dane_eta, dane_mu]

	def set_objective_form( self, objective_form, *param ):
		# objective form specifies what kind of function we are optimizing in our machines, e.g. 'ridge_regression'
		self.objective_form = objective_form
		if len(param) >= 1:
			self.objective_param = param # this can be more than 1 parameter. For ridge regression, this is the coefficient for norm-2

	def set_objective(self):
		# complete this with more function types
		if self.objective_form == 'ridge_regression':
			# for 'ridge_regression', do I wanna keep any parameter ? 
			self.objective = Ridge_regression( self.x, self.y, self.objective_param )

		elif self.objective_form == 'smoothhinge_classification':
			# for 'smoothhinge_classification', do I wanna keep any parameter ? 
			self.objective = Classification_smooth_hinge( self.x, self.y, self.objective_param )

	def updata_objective_data(self):
		self.objective.update_data( self.x , self.y )

	def update_w_loc(self, w_new):
		self.w_loc = w_new

	def update_grad_local(self, grad_new):
		self.grad_local = grad_new

	def update_grad_global_copy(self, grad_global):
		# Gets the global computed gradients and set it on all machines
		self.grad_global = grad_global

	
	''' I am not using this anymore: '''

	def compute_eval(self, curr_w):
		'''computes eval for a given w'''
		self.curr_eval = self.objective.eval(curr_w)
		#print 'opt-eval, ', self.this_eval
		return self.curr_eval

	def compute_local_grad_and_eval(self):
		grad_new = self.objective.grad(self.w_loc)
		self.update_grad_local(grad_new)

		eval_new = self.objective.eval(self.w_loc)
		# I am not keeping eval_local here, because so far I am not using it

		return self.grad_local, eval_new


	def compute_batch_local_grad(self):
		batch_grad_new = self.objective.batch_grad(self.w_loc)
		
		return batch_grad_new


	def dane_local_optimization(self, grad_global , mode , max_iter , rate_param , dane_iter_number ):
		
		# this way we do not really need grad_global to be passed here since the object has it after using 'update_grad_global_copy' function
		#print 'machine-', self.id

		print 'LOCAL OPTIMIZATON ------- DANE MODE'
		
		eta = self.opt_alg_param[0][0]
		mu = self.opt_alg_param[0][1]

		w_loc =self.w_loc
		#print 'shape of w_loc, ', np.shape(w_loc)

		if mu != 0:
	
			z_temp = self.grad_local - eta * self.grad_global
			v_temp = self.w_loc + ( 1. / mu ) * z_temp
			# v_temp = 1. * mu * self.w_loc + 2. * z_temp
	
			v_temp = np.reshape(v_temp, (-1, 1))
			w_new , number_of_gradients = self.objective.prox( v_temp , mu ,  mode , max_iter , w_loc , rate_param , dane_iter_number )

		else:
			z_temp = self.grad_local - eta * self.grad_global
			v_temp = np.reshape(z_temp, (-1, 1))
			w_new , number_of_gradients = self.objective.prox( v_temp , mu , mode , max_iter , w_loc , rate_param , dane_iter_number )

		self.update_w_loc( w_new )

		return self.w_loc , number_of_gradients



	def simple_local_optimization(self, grad_global , mode , max_iter , rate_param , dane_iter_number , initial_iter ):
		
		# this way we do not really need grad_global to be passed here since the object has it after using 'update_grad_global_copy' function
		#print 'machine-', self.id
		print 'LOCAL OPTIMIZATON ------- SIMPLE MODE'
	
		w_loc =self.w_loc
		#print 'shape of w_loc, ', np.shape(w_loc)
	
		w_new , number_of_gradients = self.objective.simple_optimize( mode , max_iter , w_loc , rate_param , dane_iter_number , initial_iter)

		self.update_w_loc( w_new )

		return self.w_loc , number_of_gradients
