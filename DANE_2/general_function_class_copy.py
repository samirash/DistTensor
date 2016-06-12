import numpy as np
import scipy.io as io
import scipy.sparse as sp
import scipy.linalg as la
import scipy.sparse.linalg as sparsela
from cvxopt import matrix
from cvxopt import solvers
from scipy.optimize import minimize
import random
import math

class Ridge_regression:
	'''
	Rigde_regression class and what it needs to provide for the DANE algorithm
	
		x: a m*d-dimensional matrix
		y: m-dimensional vector
		w: contains the parameters for the ridge regression (d-dimensional)
		mu_internal: is set according to the mu_val we have in DANE formulation
		v: is an auxilary variable here which would be substituted by the appropriate vector in order to
			make prox equal to the optimization solution for Eq.13 in DANE
	'''

	# Note for me: it might be better to keep x and y out of phi and give them as the input to the functions,
	# just like what we do with w and v, or since these are what we use for fitting out function, we might
	# want to treat them as part of our objects.
	

	def __init__(self, x, y, param = 0.005):

		# general attributes needed for any kind of function:
		self.x = x
		self.y = y
		self.dim = np.shape(x)[1]
		self.n = np.shape(y)[0]

		# coefficient for the regularizer in linear_regression :
		self.regul_coef = param[0]		# this is linear-regression regularizer
		#  print 'self.regul_coef, ',self.regul_coef

	def update_data( self , x_current , y_current ):
		self.x = x_current
		self.y = y_current
		self.n = np.shape(y_current)[0]

	def eval(self, w):
		''' computes the value of phi(w)'''
		print 'shape of w in eval function:   -- >   ' , np.shape(w)
		w = np.reshape(w, [self.dim])
		# # print ( np.dot( ( np.dot(self.x, w) - self.y ).T, ( np.dot(self.x, w) - self.y ) ) ) / self.n + self.regul_coef * np.dot(w,w) 
		print type(w)
		np.dot(w,w) 
		return ( np.dot( ( np.dot(self.x, w) - self.y ).T, ( np.dot(self.x, w) - self.y ) ) ) / self.n + self.regul_coef * np.dot(w,w) 


	def grad(self, w):
		''' computes the value of grad(phi(w))'''
		w = 1.0 * w
		return  ( 2.0 * np.dot( self.x.T , (np.dot (self.x , w) - self.y) ) ) / self.n + self.regul_coef * 2.0 * w
		# test! test these multiplications

	'''
		prox: computes the solution to argmin_w { phi(w) + (mu/2)*||w-v||^2 }

			v: is an auxilary variable here which would be substituted by the appropriate vector in order to
				make this optimization equal to Eq.13 in DANE
			mu_internal: is set according to the mu_val we have in DANE formulation
	'''
	def prox_exact_inverse(self, v, mu_internal):
		'''
		mu_internal is not 0
		'''
		x = self.x
		y = self.y
		dim = self.dim
		regul_coef = self.regul_coef
		# mu_val = self.mu_val # not needed here, it should be set in the call from outside
		n = self.n

		v = 1.0 * v
		mu_internal = 1.0 * mu_internal

		if mu_internal != 0:
	
			temp_pseudo_inverse = np.linalg.pinv( 2. * np.dot(x.T, x)/n + (mu_internal + 2. * regul_coef) * np.identity(dim), rcond=1e-15)
			w_opt = np.dot(temp_pseudo_inverse, ( mu_internal * v +  2. * np.reshape( np.dot(x.T,y) , (-1,1) ) /n ))
		elif mu_internal == 0:
	
			# print 'given v,' , v
			temp_pseudo_inverse = np.linalg.pinv( 2./ n * np.dot(x.T, x) + 2 * regul_coef * np.identity(dim) , rcond=1e-15)
			w_opt = np.dot(temp_pseudo_inverse, v +  2./ n * np.reshape( np.dot(x.T,y) , (-1,1) ) )

		# print 'result w_opt, ', w_opt		

		return w_opt 


	def prox_linear_equation(self, v, mu_internal , mode , max_iter):
		'''
		instead of using the mateix inverse we solve linear equation here!
		mu_internal != 0
		'''
		x = self.x
		y = self.y
		dim = self.dim
		regul_coef = self.regul_coef
		# mu_val = self.mu_val # not needed here, it should be set in the call from outside
		n = self.n

		v = 1.0 * v
		mu_internal = 1.0 * mu_internal

		if mu_internal != 0:
			A = 2. * np.dot(x.T, x)/n + (mu_internal + 2. * regul_coef) * np.identity(dim)
			u = mu_internal * v +  2. * np.reshape( np.dot(x.T,y) , (-1,1) ) /n 
		
			if mode == 'linearEq_inexact':
				w_opt = sparsela.lsmr(A, u, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, maxiter=max_iter, show=False)[0]
			elif mode == 'linearEq_exact':
				w_opt = np.linalg.lstsq(A, u)[0]

		elif mu_internal == 0:
			A = 2./ n * np.dot(x.T, x) + 2 * regul_coef * np.identity(dim)
			u = v +  2./ n * np.reshape( np.dot(x.T,y) , (-1,1) )
		
			if mode == 'linearEq_inexact':
				w_opt = sparsela.lsmr(A, u, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, maxiter=max_iter, show=False)[0]
			elif mode == 'linearEq_exact':
				# w_opt = np.linalg.lstsq(A, u)[0]
				w_opt = np.linalg.lstsq(A, u)[0]

		# you may want this:
		# print 'w_opt shape:'
		# print np.shape(w_opt)

		return w_opt 



	def prox_GD(self, v, mu_internal, mode , max_iter , w_loc , eval_mode ):
		'''
		instead of exact computation use a few (maybe only one) run of SGD!
		mu_internal == 0

		'''
		x = self.x
		y = self.y
		dim = self.dim
		regul_coef = self.regul_coef
		# mu_val = self.mu_val # not needed here, it should be set in the call from outside
		n = self.n

		v = 1.0 * v
		mu_internal = 1.0 * mu_internal


		# here I am just taking w_0 to be the w_loc which is set from the previous iteration
		w_curr = np.reshape( w_loc , (-1, 1) )
		sgd_gamma_0 = 1.

		print 'w_loc shape: ' , np.shape( w_loc )
		print 'w_curr shape: ' , np.shape( w_curr )


		# here I am just taking w_0 to be the w_loc which is set from the previous iteration
		# print 'check the shapes: 2 lines:'
		# print np.shape(w_loc)
		# w_curr = w_loc
		# print np.shape(w_curr)
		# sgd_gamma_0 = 1.


		if eval_mode == 1:
			w_optimum = self.prox_linear_equation(v, mu_internal , 'linearEq_exact' , 0 )
			value_optimum = self.eval(w_optimum) + mu_internal / 2. * np.dot( (w_optimum - v).T , (w_optimum - v) )


		for i in range( max_iter ):
			# print 'w in sgd .....:'
			# print w_curr

			if eval_mode == 1:
				print 'shapes: ,' , np.shape(w_curr) , np.shape(v)
				value_current = self.eval(w_curr) + mu_internal / 2. * np.dot( (w_curr - v).T , (w_curr - v) )
				print 'norm of w:   ', np.dot( w_curr.T , w_curr )
				print 'this is the suboptimality in SGD: for step, ', i , ', ' , value_current - value_optimum
			
			# sgd_gamma = sgd_gamma_0 * 1./ ( 1 + sgd_gamma_0 * 0.9 * regul_coef * ( i + 1 ) )
			sgd_gamma = sgd_gamma_0 / 1. #math.sqrt(i+1)


			print 'this is GD learning rate: ' , sgd_gamma
			
			# grad = 2 * np.dot ( A.T , (np.dot (A , w_curr) - u ) ) 

			GD_grad = self.grad( w_curr )
			print np.shape(GD_grad)
			print np.shape(w_curr)
			# print np.shape()
			print np.shape( w_curr - v)
			GD_grad = GD_grad + 1. * mu_internal * ( w_curr - v )
			grad = GD_grad

			print 'shape of grad: ', np.shape(grad) 
		
			w_curr = w_curr - sgd_gamma * grad
			
			# print 'grad in sgd .....'
			# print sample_grad

		w_opt = w_curr

		return w_opt




	def prox_SGD(self, v, mu_internal, mode , max_iter , w_loc , eval_mode ):
		'''
		instead of exact computation use a few (maybe only one) run of SGD!
		mu_internal == 0

		'''
		x = self.x
		y = self.y
		dim = self.dim
		regul_coef = self.regul_coef
		# mu_val = self.mu_val # not needed here, it should be set in the call from outside
		n = self.n

		v = 1.0 * v
		mu_internal = 1.0 * mu_internal

		# here I am just taking w_0 to be the w_loc which is set from the previous iteration
		w_curr = np.reshape( w_loc , (-1, 1) )
		sgd_gamma_0 = 1.


		if eval_mode == 1:
			w_optimum = self.prox_linear_equation(v, mu_internal , 'linearEq_exact' , 0 )
			value_optimum = self.eval(w_optimum) + mu_internal / 2. * np.dot( (w_optimum - v).T , (w_optimum - v) )


		for i in range( max_iter ):
			# print 'w in sgd .....:'
			# print w_curr

			if eval_mode == 1:
				value_current = self.eval(w_curr) + mu_internal / 2. * np.dot( (w_curr - v).T , (w_curr - v) )
				print 'this is the SUBOPTIMALITY in SGD: for step, ', i , ', ' , value_current - value_optimum
			
			# sgd_gamma = sgd_gamma_0 * 1./ ( 1 + sgd_gamma_0 * 0.9 * regul_coef * ( i + 1 ) )
			# sgd_gamma = sgd_gamma_0 / math.sqrt(i+1)
			sgd_gamma = 0.002
			# sgd_gamma = sgd_gamma_0 / (i+1)


			print 'this is SGD learning rate: ' , sgd_gamma

			rand_index = random.randrange( 0 , n ) 
			print 'this is my random index: ' ,rand_index, '/ ,' , n
			sample_x = x[ rand_index , : ]
			print 'shape of random point: ', np.shape( sample_x )
			sample_x = np.reshape( sample_x , (-1,1))
			print 'shape of random point reshaped: ', np.shape( sample_x )
			sample_y = y[ rand_index ]
			
			# first computing the gradient for phi(w_0) :
			sample_grad = ( 2.0 * ( np.dot ( sample_x.T , w_curr ) - sample_y )  * sample_x )  + self.regul_coef * 2.0 * w_curr # / self.n this was for the first summation
			sample_grad = sample_grad + 1. * mu_internal * ( w_curr - v )
		
			w_curr = w_curr - sgd_gamma * sample_grad
			
			# print 'grad in sgd .....'
			# print sample_grad

		w_opt = w_curr

		return w_opt




	def prox(self, v, mu_internal , mode , max_iter , w_loc ):
		
		# w_opt = self.prox_exact_inverse(v, mu_internal)
		if mode == 'inverse_exact':
			w_opt = self.prox_exact_inverse(v, mu_internal)

		elif mode == 'linearEq_exact' or mode == 'linearEq_inexact' : 
			w_opt = self.prox_linear_equation(v, mu_internal , mode , max_iter )

		elif mode == 'SGD':
			print 'HELLOOOO    ................    RUNNING SGD HERE ......  ......'
			eval_mode = 0
			w_opt = self.prox_SGD(v, mu_internal , mode , max_iter , w_loc , eval_mode)

		elif mode == 'GD':
			print 'HELLOOOO    ................    RUNNING Full Batch GD HERE ......  ......'
			eval_mode = 1
			w_opt = self.prox_GD(v, mu_internal , mode , max_iter , w_loc , eval_mode)


		return w_opt

		# injaa kheili messed up shode bayad dorostesh konam!!!






















##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################







def smooth_hinge_loss( u , gamma ):	
	'''
	Calculates the smoothed loss with parameter gamma on each element of the input numpy vector u
    (u is assumed to contain classification raw results (y*<x.T, w>))
    
    input: vector u  (u is assumed to contain classification raw results (y*<x.T, w>))
    '''
	loss_vector = np.zeros( np.shape(u)[0])

	for index in xrange( np.shape(u)[0] ):
		if u[index] > 1:
			loss_vector [index] = 0 
		elif u[index] < 1 - gamma:
			loss_vector [index] = ( 1 - u[index] ) - gamma/2.
		else:
			loss_vector [index] = 1. / ( 2 * gamma ) * ( 1 - u[index] )**2
	return loss_vector

def smooth_hinge_loss_grad( u , gamma ):	
	'''
	Calculates the grad of the smoothed loss with parameter gamma with respect to u, on each element of the input numpy vector u
   (where u is itself a function of x_i, y_i, parameter w and this should be taken into account later to obtain a vector as the gradient for each element of u)
	'''
	loss_grad_vector = np.zeros( np.shape(u)[0] )

	for index in xrange( np.shape(u)[0] ):

		if u[index] > 1:
			loss_grad_vector [index] = 0 
		elif u[index] < 1 - gamma:
			loss_grad_vector [index] = -1
		else:
			loss_grad_vector [index] = ( u[index] - 1 )/gamma

	return loss_grad_vector

def smooth_hinge_loss_hessian( u , gamma ):	
	'''
	Calculates the grad of the smoothed loss with parameter gamma with respect to u, on each element of the input numpy vector u
   (where u is itself a function of x_i, y_i, parameter w and this should be taken into account later to obtain a vector as the gradient for each element of u)
	'''
	loss_hessian_vector = np.zeros( np.shape(u)[0] )

	for index in xrange( np.shape(u)[0] ):

		if u[index] > 1:
			loss_hessian_vector [index] = 0 
		elif u[index] < 1 - gamma:
			loss_hessian_vector [index] = 0
		else:
			loss_hessian_vector [index] = 1./gamma

	return loss_hessian_vector



#################################
#################################
#################################

class Classification_smooth_hinge:

	def __init__(self, x, y, param ):

		# general attributes needed for any kind of function:
		self.x = x
		self.y = y
		print np.shape(x)
		self.dim = np.shape(x)[1]
		self.n = np.shape(y)[0]

		# coefficient for the regularizer in linear_regression :
		self.regul_coef = param[0]		# this is linear-regression regularizer
		self.gamma = param[1]		# this is the parameter for the smoothed hinge loss (shows the ammount of smoothness)
		#  print 'self.regul_coef, ',self.regul_coef


	def eval(self, w):
		''' computes the value of phi(w)'''
		w = np.reshape(w, [self.dim])
		
		hinge_loss_vector = smooth_hinge_loss( np.multiply ( self.y , np.dot(self.x, w) ) , self.gamma )
		overal_hinge_loss = np.dot( np.ones((self.n , 1)).T , hinge_loss_vector ) / self.n
		
		return (overal_hinge_loss + self.regul_coef / 2. * np.dot(w,w))


	def grad(self, w):
		''' computes the value of grad(phi(w))'''
		# print np.shape(w)
		# print np.shape(self.x)
		hinge_loss_vector_grad = smooth_hinge_loss_grad( np.multiply ( self.y , np.dot(self.x, w) ) , self.gamma )
		overal_hinge_loss_grad = np.dot( np.multiply( hinge_loss_vector_grad , self.y ) , self.x )

		return overal_hinge_loss_grad + self.regul_coef * w


	def hessian(self, w):
		# I have not tested this function, because it is not used having hessian_times_p
		''' computes the value of hessian(phi(w))'''
		# print np.shape(w)
		# print np.shape(self.x)
		hinge_loss_vector_hessian = smooth_hinge_loss_hessian( np.multiply ( self.y , np.dot(self.x, w) ) , self.gamma )
		temp = np.dot( ( np.multiply( np.sqrt(hinge_loss_vector_hessian) , self.y ) ).T , self.x )
		overal_hinge_loss_hessian = np.dot( temp , temp.T )

		return overal_hinge_loss_hessian + self.regul_coef * np.identity( self.dim )


	def hessian_times_p(self, w, p):
		''' computes the value of hessian(phi(w))'''
		# print np.shape(w)
		# print np.shape(self.x)
		hinge_loss_vector_hessian = smooth_hinge_loss_hessian( np.multiply ( self.y , np.dot(self.x, w) ) , self.gamma )
		temp = np.multiply( np.multiply( hinge_loss_vector_hessian , np.power( self.y , 2 ) ) , np.dot( self.x , p ) )
		overal_hinge_loss_hessian = np.dot( temp.T , self.x )

		return overal_hinge_loss_hessian + self.regul_coef * p


	def prox_smoothhinge_eval( self , w , *args ):
		v = args[0]
		mu_internal = args[1]
		return self.eval(w) + mu_internal / 2. * np.dot( w - v , w - v )

	def prox_smoothhinge_grad( self , w , *args ):
		v = args[0]
		mu_internal = args[1]
		return self.grad(w) + mu_internal * ( w - v)

	def prox_smoothhinge_hessian( self , w , *args ):
		v = args[0]
		mu_internal = args[1]
		return self.hessian(w) + mu_internal * np.identity( self.dim )

	def prox_smoothhinge_hessian_times_p( self , w , p , *args ):
		v = args[0]
		mu_internal = args[1]
		return self.hessian_times_p( w, p ) + mu_internal * p


	def prox(self, v, mu_internal, mode , max_iter ):
		'''
		prox: computes the solution to argmin_w { phi(w) + (mu/2)*||w-v||^2 }

		v: is an auxilary variable here which would be substituted by the appropriate vector in order to
			make this optimization equal to Eq.13 in DANE
		mu_internal: is set according to the mu_val we have in DANE formulation
		
		'''
		# Now we should run Newton to estimate the optimal solution w,
		# given the function value, grad, and hessian from the above functions:

		w_0 = np.zeros( ( self.dim , 1 ) )

		if mode == 'limit_iters':
			res = minimize( self.prox_smoothhinge_eval, w_0 , args=( v , mu_internal ) , method='Newton-CG', jac=self.prox_smoothhinge_grad, hessp=self.prox_smoothhinge_hessian_times_p, options={'xtol': 1e-8, 'disp': True , 'maxiter': max_iter } )
		elif mode == 'exact':
			res = minimize( self.prox_smoothhinge_eval, w_0 , args=( v , mu_internal ) , method='Newton-CG', jac=self.prox_smoothhinge_grad, hessp=self.prox_smoothhinge_hessian_times_p, options={'xtol': 1e-8, 'disp': True  } )
		w_opt = res.x

		# scipy.optimize.minimize(fun, x0, args=(), method='Newton-CG', jac=None, hess=None, hessp=None, tol=None, callback=None, options={'disp': False, 'xtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': None})

		return w_opt

#################################
#################################
#################################
#################################


	def prox_SGD2(self, v, mu_internal, mode , max_iter , w_loc , eval_mode ):
		'''
		instead of exact computation use a few (maybe only one) run of SGD!
		mu_internal == 0

		'''
		x = self.x
		y = self.y
		dim = self.dim
		regul_coef = self.regul_coef
		# mu_val = self.mu_val # not needed here, it should be set in the call from outside
		n = self.n

		v = 1.0 * v
		mu_internal = 1.0 * mu_internal

		# here I am just taking w_0 to be the w_loc which is set from the previous iteration
		w_curr = np.reshape( w_loc , (-1, 1) )
		sgd_gamma_0 = 1.


		if eval_mode == 1:
			w_optimum = self.prox_linear_equation(v, mu_internal , 'linearEq_exact' , 0 )
			value_optimum = self.eval(w_optimum) + mu_internal / 2. * np.dot( (w_optimum - v).T , (w_optimum - v) )

		# if mu_internal != 0:
		# 	A = 2. * np.dot(x.T, x)/n + (mu_internal + 2. * regul_coef) * np.identity(dim)
		# 	u = mu_internal * v +  2. * np.reshape( np.dot(x.T,y) , (-1,1) ) /n 
		

		# elif mu_internal == 0:
		# 	A = 2./ n * np.dot(x.T, x) + 2 * regul_coef * np.identity(dim)
		# 	u = v +  2./ n * np.reshape( np.dot(x.T,y) , (-1,1) )
		

		for i in range( max_iter ):
			# print 'w in sgd .....:'
			# print w_curr

			if eval_mode == 1:
				value_current = self.eval(w_curr) + mu_internal / 2. * np.dot( (w_curr - v).T , (w_curr - v) )
				print 'this is the suboptimality in SGD: for step, ', i , ', ' , value_current - value_optimum
			
			# sgd_gamma = sgd_gamma_0 * 1./ ( 1 + sgd_gamma_0 * 0.9 * regul_coef * ( i + 1 ) )
			sgd_gamma = sgd_gamma_0 / math.sqrt(i+1)


			print 'this is SGD learning rate: ' , sgd_gamma

			rand_index = random.randrange( 0 , n ) 
			print 'this is my random index: ' ,rand_index, '/ ,' , n
			sample_x = x[ rand_index , : ]
			print 'shape of random point: ', np.shape( sample_x )
			sample_x = np.reshape( sample_x , (-1,1))
			sample_y = y[ rand_index ]
			

			if mu_internal != 0:
				A = 2. * np.dot(sample_x.T, sample_x)/n + (mu_internal + 2. * regul_coef) * np.identity(dim)
				u = mu_internal * v +  2. * np.reshape( np.dot(sample_x.T,sample_y) , (-1,1) ) /n 
			elif mu_internal == 0:
				A = 2./ n * np.dot(sample_x.T, sample_x) + 2 * regul_coef * np.identity(dim)
				u = v +  2./ n * np.reshape( np.dot(sample_x.T,sample_y) , (-1,1) )

			# first computing the gradient for phi(w_0) :
			# sample_grad = ( 2.0 * ( np.dot ( sample_x.T , w_curr ) - sample_y )  * sample_x ) / self.n + self.regul_coef * 2.0 * w_curr
			# sample_grad = sample_grad + 1. * mu_internal * ( w_curr - v )
			sample_grad = 2 * np.dot ( A.T , np.dot (A , w_curr) - u )
		
			w_curr = w_curr - sgd_gamma * sample_grad
			
			# print 'grad in sgd .....'
			# print sample_grad

		w_opt = w_curr

		return w_opt


