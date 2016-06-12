import numpy as np
import scipy.io as io
import scipy.sparse as sp
import scipy.linalg as la
from general_function_class import Ridge_regression
import matplotlib.pyplot as plt


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

		self.w_loc = np.zeros(w_length)
		self.grad_global_copy = np.zeros(w_length)
		self.grad_local = np.zeros(w_length)


	def get_data(self, data):
		# data should be given as a numpy array 
		self.data = data
		self.x = data[:,0:-1]
		self.y = data[:,-1]
		print self.id, np.shape(data)

	def add_data(self, data):
		self.data = np.concatenate(self.data, data)
		self.x = np.concatenate(self.x, data[:,0:-1])
		self.y = np.concatenate(self.y, data[:,-1])


	def set_optimization_algorithm( self, algorithm, *param ):
		self.opt_alg = algorithm			# this optimization algorithm can be DANE for instance or ADMM
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
			self.rg = Ridge_regression( self.x, self.y, self.objective_param )


	def update_w_loc(self, w_new):
		self.w_loc = w_new

	def update_grad_local(self, grad_new):
		self.grad_local = grad_new

	def update_grad_global_copy(self, grad_global):
		# Gets the global computed gradients and set it on all machines
		self.grad_global = grad_global

	def compute_this_eval(self, this_w):
		self.this_eval = self.rg.eval(this_w)

	def compute_local_grad_and_eval(self):
		grad_new = self.rg.grad(self.w_loc)
		self.update_grad_local(grad_new)

		eval_new = self.rg.eval(self.w_loc)
		# I am not keeping eval_local here, because so far I am not using it

		return self.grad_local, eval_new


	def dane_local_optimization(self, grad_global):
		
		# this way we do not really need grad_global to be passed here since the object has it after using 'update_grad_global_copy' function
		#print 'machine-', self.id
		eta = self.opt_alg_param[0][0]
		mu = self.opt_alg_param[0][1]
	
		z_temp = self.grad_local - eta * self.grad_global
		v_temp = self.w_loc + ( 2. / mu ) * z_temp
	
		v_temp = np.reshape(v_temp, (-1, 1))
		w_new = self.rg.prox( v_temp , mu )
		self.update_w_loc( w_new )

		return self.w_loc


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


	opt_evals = np.zeros(m)

	for i in range(m):

		machines[i].set_objective_form( objective_form, objective_param )
		machines[i].set_objective( )
		machines[i].set_optimization_algorithm( optimization_algorithm,  alg_param )

		# not the nicest place to put this, maybe change later:
		opt_evals[i] = machines[i].compute_this_eval(w_opt)

	return opt_evals



def DANE_procedure(machines, w_opt, objective_form, objective_param, max_iter, eta, mu ):  # check the experiments to see what \mu needs to be is it what I have in machines_setup function above?

	# The main DANE procedure given the machines already with their data points
	# if mu=0 does not converge then you shall use 0.3*lambda where the function is lambda-strong convex

	# 	 eta_val: the value of mu used in Eq.13 in DANE paper. The factor for the global gradent of w.

	opt_evals = machines_setup( machines, w_opt, objective_form, objective_param, "DANE",  eta,  mu)


	m = len(machines)
	w_length = machines[0].w_length

	''' Initializing global and local weights and gradients with 0 matrices (or vectors): '''
	w_global = np.zeros(w_length)
	grad_global = np.zeros(w_length)
	eval_global = 0

	local_gradients = np.zeros((w_length, m))
	local_evals = np.zeros(m)
	local_ws = np.zeros((w_length, m))

	eval_diffs = np.zeros(max_iter)
	submodularities = np.zeros(max_iter)


	''' Defining functions used in the main loop of DANE: '''

	def compute_local_gradients(machines):
		# computes all local gradients
		for i in range(m):
			local_gradients[:,i], local_evals[i] = machines[i].compute_local_grad_and_eval()
		

	def compute_grad_global(local_gradients):
		# computes global grad as the average of the local gradients
		grad_global = np.mean(local_gradients, axis=1)
		# test! : check all the dimensions
		return grad_global

	def distribute_grad_global(machines, grad_global):
		# distributed the value of the global gradient to all machines
		for i in range(m):
			machines[i].update_grad_global_copy(grad_global)

	def perform_local_optimizations(machines, grad_global):
		''' test!: # we do not actually need to pass this grad_global here, but is it better to use this and totally remove distribute_grad_global ?'''
		# computes all local optimims which are essentially local w's
		for i in range(m):
			local_ws[:,i] = machines[i].dane_local_optimization(grad_global)

	def compute_w_global(local_ws):
		# computes global w as the average of all local w's
		w_global = np.mean(local_ws, axis=1)
		#print w_global.T
		return w_global

	def distribute_w_global(machines, w_global):
		'''distributes w_global to all machines and sets their w to w_global '''
		for i in range(m):
			machines[i].update_w_loc(w_global)

	def compute_eval_global(machines):
		eval_global = np.mean(local_evals)
		return eval_global




	''' Main loop of the DANE Algorithm: '''
	eval_pred = eval_global

	for t in range(max_iter):

		compute_local_gradients( machines )
		grad_global = compute_grad_global( local_gradients )
		distribute_grad_global( machines, grad_global )
		perform_local_optimizations( machines, grad_global )
		w_global = compute_w_global( local_ws )
		distribute_w_global( machines, w_global )
		eval_global = compute_eval_global(machines)

		eval_diff = eval_global - eval_pred
		eval_diffs[t] = eval_diff
		print eval_diff

		#submodularities[t] = eval_global - np.mean(opt_eval)

	return w_global, eval_diffs, submodularities




def run_DANE_experiment(N, m, max_iter):
	# generating N 500-d points from  y = <x, w_opt> + noise:

	#N = 10000
	w_opt = np.ones( [ 500, 1 ] )
	cov = np.diag( (np.array(range(1, 501))) ** ( -1.2 ) )
	mean = np.zeros( [ 500 ] )

	X = np.random.multivariate_normal(mean, cov, ( N ))

	noise = np.array(np.random.standard_normal( size=( N, 1) ))

	Y = np.dot( X , w_opt )
	Y = Y + noise

	data = np.concatenate(( X , Y ), axis = 1 )

	#m = 4

	# I am calling initialize_machines to set up out computing machines:
	machines = initialize_machines( m, data )
	w_ans, eval_diffs, submodularities = DANE_procedure( machines , w_opt, 'ridge_regression', 0.005, max_iter, eta=1 , mu=0.1  )
	#print w_ans
	print np.sqrt(np.dot(w_ans , w_opt))

	return w_ans, eval_diffs, submodularities




max_iter = 5
all_eval_diffs = np.zeros((max_iter , 6))
all_submodularities = np.zeros((max_iter , 6))
i = 0
for m in [4, 16]:
	for N in [6000, 10000, 14000]:
		print ' m, N =',m ,N
		w_ans, eval_diffs, submodularities = run_DANE_experiment(N, m, max_iter)

		all_eval_diffs[:,i ] = eval_diffs  
		all_submodularities[:,i ] = submodularities
		i = i + 1
		print 'i', i
		#print all_eval_diffs

all_eval_diffs = np.log10(all_eval_diffs)
all_submodularities = np.log10(all_submodularities)
t = np.arange(max_iter)

plt.plot(t,all_submodularities[:,0],'r')
plt.plot(t,all_submodularities[:,1],'b')
plt.plot(t,all_submodularities[:,2],'g')
plt.show()

plt.plot(t,all_submodularities[:,3],'r')
plt.plot(t,all_submodularities[:,4],'b')
plt.plot(t,all_submodularities[:,5],'g')

plt.show()



# % python -mtimeit  "l=[]"

# maybe make a class for central_machine as well, but not sure it is of any benefit!!






'''

class Central_machine:


	def compute_grad_global(local_gradients):
		# computes global grad as the average of the local gradients
		grad_global = np.mean(local_gradients, axis=1)
		# test! : check all the dimensions
		return grad_global

	def distribute_grad_global(machines, grad_global):
		# distributed the value of the global gradient to all machines
		for i in range(len(machines)):
			machines[i].update_grad_global_copy(grad_global)

	def compute_w_global(local_ws):
		# computes global w as the average of all local w's
		w_global = np.mean(local_ws, axis=1)
		return w_global

	def distribute_w_global(machines, w_global):
		# distributes w_global to all machines and sets their w to w_global
		for i in range(len(machines)):
			machines[i].update_w_loc(w_global)


'''



