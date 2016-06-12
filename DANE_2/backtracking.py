import time
import numpy as np


def backtracking( func, x, direction, alpha=0.4, beta=0.9, maximum_iterations=65536 ):
    """ 
    Backtracking linesearch
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    x:                  the current iterate
    direction:          the direction along which to perform the linesearch
    alpha:              the alpha parameter to backtracking linesearch
    beta:               the beta parameter to backtracking linesearch
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    """

    if alpha <= 0:
        raise ValueError("Alpha must be positive")
    if alpha >= 0.5:
        raise ValueError("Alpha must be less than 0.5")
    if beta <= 0:
        raise ValueError("Beta must be positive")
    if beta >= 1:
        raise ValueError("Beta must be less than 1")
        
        
    x = np.asarray( x )
    direction = np.asarray( direction )
    
    
    # value, gradient = func( x , 1 )
    value = func.eval(x)
    gradient = func.grad(x)
    value = np.double( value )
    gradient = np.asarray( gradient )
    
    derivative = np.vdot( direction, gradient )
    
    # checking that the given direction is indeed a descent direciton
    if derivative >= 0:
        return 0
        
    else:
        t = 1
        iterations = 0
        while True:        
        
            # if (TODO: TERMINATION CRITERION): break
            x_temp = x + t * direction
            # value_temp = func( x_temp , 0 )
            value_temp = func.eval(x_temp)
            value_temp = np.double( value_temp )

            if value_temp <= value + alpha * t * derivative:
                # print(iterations)
                return t

            # t = TODO: BACKTRACKING LINE SEARCH
            else:
                t = beta * t
            
            iterations += 1
            if iterations >= maximum_iterations:
                raise ValueError("Too many iterations")
      
        return t
