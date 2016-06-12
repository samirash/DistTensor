import numpy as np
import scipy.io as io
import scipy.sparse as sp
import scipy.linalg as la
import scipy.sparse.linalg as sparsela
from cvxopt import matrix
from cvxopt import solvers
from scipy.optimize import minimize
from general_function_class import Ridge_regression
from general_function_class import Classification_smooth_hinge



""" The main test function to check all of the functions in regression class: """
def test_regression_class_2(mu_internal):

	print 'test-2'

	w_opt = 1.0 * np.ones(1)
	print w_opt

	x1 = np.array([[0],[1]])
	y1 = np.array([0,1]).T

	x2 = np.array([[-1],[-2]])
	y2 = np.array([-1,-2]).T

	x = np.array([[0],[1],[-1],[-2]])
	y = np.array([0,1, -1, -2]).T

	rg1 = Ridge_regression(x1,y1,[0.005])
	rg2 = Ridge_regression(x2,y2,[0.005])
	rg = Ridge_regression(x,y,[0.005])

	print 'regul_coef , ', rg1.regul_coef
	print 'dim, ', rg1.dim
	print 'n, ', rg1.n

	w1_0 = 1.0 * np.zeros(1)
	w2_0 = 1.0 * np.zeros(1)

	w_0 = 1.0 * np.zeros(1)


	print "values on opt w"
	print rg1.eval(w_opt)
	print rg2.eval(w_opt)
	print rg.eval(w_opt)

	print 'correct answer for all three:'
	print '0.005'

	print "values on initial w"
	print rg1.eval(w1_0)
	print rg2.eval(w2_0)
	print rg.eval(w_0)
	
	print 'correct answers:'
	print '1/2'
	print '5/2'
	print '3/2'

	# print rg.grad(w)
	print "grads on initial w"
	print rg1.grad(w1_0)
	print rg2.grad(w2_0)
	print rg.grad(w_0)

	print 'correct answers:'
	print '-1'
	print '-5'
	print '-3'

	global_grad = ( rg1.grad(w1_0) + rg2.grad(w2_0) ) / 2.
	print 'global_grad', global_grad

	print 'correct answer:'
	print '-3'

	v1 = 1. * rg1.grad(w1_0) - 1. * global_grad
	v2 = 1. * rg2.grad(w2_0) - 1. * global_grad



	print "v1 and v2:"
	print v1
	print v2

	print 'correct answers:'
	print '2'
	print '-2'

	if mu_internal != 0:
		v1 = w1_0 + v1 / mu_internal
		v2 = w2_0 + v2 / mu_internal

	# mu_internal = 0.000000001

	print "w_opt values on initial_w  (w_opt1, w_opt2, w_opt_global)"
	print rg1.prox(v1, mu_internal)
	print rg2.prox(v2, mu_internal)
	print ( rg1.prox(v1, mu_internal) + rg2.prox(v2, mu_internal) ) / 2.
	w_res = ( rg1.prox(v1, mu_internal) + rg2.prox(v2, mu_internal) ) / 2.

	print 'correct answers:'
	print 300./101
	print 300./501
	print (300./101 + 300./501)/2
	
	print 'final evaluation on the new w'
	print (rg1.eval(w_res) + rg2.eval(w_res))/2.

	print 'correct answer should be:'
	print ( rg1.eval((300./101+300./501)/2 ) + rg2.eval((300./101+300./501)/2 ) )/ 2.

mu_internal =  0.00000000000000
# test_regression_class_2(mu_internal)


#################################
#################################
#################################
#################################



def test_smoothhinge_classification_class( mu_internal ):

	x1 = np.array([[1, -1]])
	x2 = np.array([[1, 0.5]])
	x3 = np.array([[-0.5, -1]])

	y1 = np.array([1])
	y2 = np.array([1])
	y3 = np.array([1])

	x4 = np.array([[-1, 1]])
	x5 = np.array([[-1, -0.5]])
	x6 = np.array([[0.5, 1]])

	y4 = np.array([-1])
	y5 = np.array([-1])
	y6 = np.array([-1])

	x = np.concatenate((x1, x2, x3, x4, x5, x6), axis = 0)
	print x
	y = np.concatenate((y1, y2, y3, y4, y5, y6), axis = 0)
	print y
	print y.T


	sh = Classification_smooth_hinge( x, y ,[0.005, 0.2])

	print 'printing values for w = (1,1), (1, -1), (-1, 1):'
	print sh.eval( np.array( [1, 1] ) )
	print sh.eval( np.array( [1, -1] ) )
	print sh.eval( np.array( [-1, 1] ) )

	print 'printing gradiants for w = (1,1), (1, -1), (-1, 1):'
	print sh.grad( np.array( [1, 1] ) )
	print sh.grad( np.array( [1, -1] ) )
	print sh.grad( np.array( [-1, 1] ) )

	print 'printing gradiants for w = (1,1), (1, -1), (-1, 1):'
	print sh.grad( np.array( [1, 1] ) )
	print sh.grad( np.array( [1, -1] ) )
	print sh.grad( np.array( [-1, 1] ) )

	v = np.array( [1, -1] )

	print "w_opt values on initial_w "
	print sh.prox(v, mu_internal)
	
# test_smoothhinge_classification_class( mu_internal )




		# # Forming the matrices needed to pass to the Quadratic solver (CVXOPT package):

		# P = np.concatenate( ( np.eye( self.dim + self.n , self.dim) , np.zeros( ( self.dim + self.n , self.n ) ) ) , axis = 1 )
		# P = ( self.regul_coef + mu_internal ) * P

		# q = np.concatenate( ( - mu_internal*v , ( 1/self.n )*np.ones( self.n ) ) , axis = 0)

		# G_top_block = np.concatenate( ( np.zeros( (self.n , self.dim) ) , - np.eye(self.n) ) , axis = 1 )
		# G = np.concatenate ( (G_top_block , G_top_block , G_top_block) , axis = 0 )

		# h_first_block = np.zeros( (self.n , 1) )
		# h_second_block = self.gamma/2. - 1 + np.multiply( self.y , np.dot(self.x , w) )
		# h_third_block = - ( 1 / ( 2 * self.gamma) ) * np.power( (1 - np.multiply( self.y , np.dot(self.x , w) ) ) , 2 )
		# h = np.concatenate ( (h_first_block , h_second_block , h_third_block) , axis = 0 )

		# P_ = matrix(P)
		# q_ = matrix(q)
		# G_ = matrix(G)
		# h_ = matrix(h)


		# # solving the quadratic problem with the solvers qp function:

		# sol = solvers.qp( P_ ,q_ ,G_ ,h_ )

		# l_optimal = sol['x']
		# solution_optimal = sol['primal objective']


		# # checking the solution:
		# flag = (solution_optimal == ( 0.5 * np.dot( np.dot( l_optimal.T , P ) , l_optimal ) + np.dot ( q.T , l_optimal) ))
			
		# if flag:
		# 	print 'The qadratic solver provides the right answer!! '
		# else:
		# 	print 'warning: the solver does not provide the right answer!'

		# w_opt = l_optimal[ 0:self.dim ]
