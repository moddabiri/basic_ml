from numpy import e
from ml_util.functional_utility import check_none

import math
import numpy as np

def sigmoid(z):
    """
    Sigmoid function 1 / (1 + e ^ (-z)).

    Parameters
    ----------
    z : matrix
        The input matrix.
        
    Returns
    -------
    sigmoid: matrix
    
    """

    check_none(z=z)

    return np.matrix(1.0 / (1.0 + e**(-1 * np.asarray(z))))

def sigmoid_gradient(z):
    sig = sigmoid(z)
    return np.multiply(sig, 1 - sig)

def almost_equal(actual, desired, difference):
    """
    Compares 2 values with a difference range tolerance.

    Parameters
    ----------
    actual : number
        The actual number.

    desired : number
        The desired number.
    difference : number
        The approximation tolerance.

    Returns
    -------
    approximate equality: Boolean
        If the two values given the approximation tolerance are equal.
    
    Throws:
    -------
    Value error:
        If difference argument is less than 0.

    """
    check_none(actual=actual, desired=desired,difference=difference)

    if difference < 0:
        raise ValueError("The 'parameter' difference should be a positive value.")

    return math.fabs(actual - desired) <= difference

def get_polynomial(x, degree):
    """
    Takes and array or [n,1] numpy matrix and polynomial degree, returns a matrix of [n, degree] dimensions containing the polynomial sets from x by degree.

    Parameters
    ----------
    x : list/numpy.ndarray/numpy
        The list of numbers to become polynomial.
    degree : int
        The polynomial degree

    Returns
    -------
    featurePol : ndarray
        A matrix of polynomial sets.
    
    Throws:
    -------
    Type error:
        In case argument x is not a valid list/numpy.ndarray/numpy.matrix or degree is not a valid integer.

    Value error:
        If argument x is not in expected dimensions ([n, 1]).

    """

    check_none(x=x, degree=degree)

    if (not isinstance(x, list) and not isinstance(x, np.ndarray) and not isinstance(x, np.matrixlib.defmatrix.matrix)):
        raise TypeError("get_polinaminal_set is only allowed on list, numpy matrix or ndarray. The type of input was " + str(type(x)))

    if (not isinstance(x, list)):
        if (not hasattr(x, 'shape')):
            raise TypeError("get_polinaminal_set is only allowed on list, numpy matrix or ndarray. The argument x did not have attribute shape.")

        if (x.shape[0] != 1):
            raise ValueError("get_polinaminal_set is only allowed on list or numpy matrix/ndarray of size n,1. The size of input was: " + str(x.shape))

    if (isinstance(x, np.ndarray) or isinstance(x, np.matrixlib.defmatrix.matrix)):
        n = x.shape[0]
        m = x.shape[1]
    else:
        m = len(x)
        n = 1

    feature = np.matrix(x)
    featurePol = np.ones([n * degree, m], dtype = float)

    #Get x^0, x^1,... matrix
    for deg in range(degree):
        featurePol[deg * n : (deg + 1) * n, :] = np.power(feature, deg)

    return featurePol

def cross_multiplication(m1, m2, axis=1):
    """
    Does a cross multiplication across the columns or rows of two matrices. Notice! It is not a cross product function. 
    Example (axis = 1):
        | 1   2|     | 1   2 |    |1  2   2   4 |
        | 3   4|  x  | 3   4 | =  |9  12  12  16|
          2x2           2x2            2x4

    Example (axis = 0):
        | 1   2|     | 1   2 |    |1  4 |
        | 3   4|  x  | 3   4 | =  |3  8 |
                                  |3  8 |
                                  |9  16|
          2x2           2x2         4x2

    Parameters
    ----------
    m1 : numpy.matrix
    m2 : numpy.matrix
    axis : int, optional
        1 for horizontal, 0 for vertical

    Returns
    -------
    cross multiplication : numpy.matrix
    
    Throws:
    -------
    Type error:
        In case argument m1 or m2 are not a valid numpy.matrix.

    Value error:
        In case argument m1 and m2 are not in expected dimensions

    Value error:
        In case argument axis is not either 0 or 1
    """

    check_none(m1=m1,m2=m2,axis=axis)

    #TODO: Below check does not work for logistic regression (map features) code. Fix this. 
    #Current behaviour is the type(m1) gives <type>. I am mistaking in a python concept that it can't realize the type!
    #if (not isinstance(m1, np.matrixlib.defmatrix.matrix) or not isinstance(m2, np.matrixlib.defmatrix.matrix)):
    #    raise TypeError("Matrix m1 and m2 must be a valid numpy matrix.")

    if (axis != 0 and axis != 1):
        raise ValueError("axis argument must either be 0 or 1")

    if (axis == 1 and m1.shape[0] != m2.shape[0]):
        raise ValueError("Arguments m1 and m2 are not in expected dimensions. For axis=1, m1 and m2 must have the same number of rows")

    if (axis == 0 and m1.shape[1] != m2.shape[1]):
        raise ValueError("Arguments m1 and m2 are not in expected dimensions. For axis=0, m1 and m2 must have the same number of columns")

    result = None
    if (axis == 1):
        result = np.ones([m1.shape[0], m1.shape[1] * m2.shape[1]], dtype=np.float)

        for i in range(m1.shape[1]):
            for j in range(m2.shape[1]):
                result[:, (i * m2.shape[1]) + j] = np.multiply(m1[:, i], m2[:, j]).transpose()
    else:
        result = np.ones([m1.shape[0] * m2.shape[0], m1.shape[1]], dtype=np.float)

        for i in range(m1.shape[0]):
            for j in range(m2.shape[0]):
                result[(i * m2.shape[0]) + j, :] = np.multiply(m1[i, :], m2[j, :])

    return result

    

def normalizeMatrix(x, axis=1, ddof=1):
    m = x.shape[1]
    mu = np.mean(x, axis=axis)
    sigma = np.std(x, axis=axis, ddof=ddof)
    original_shape = sigma.shape
    #Eliminate zero std values
    sigma = np.asarray([(1.0 if a == 0.0 else a) for a in np.asarray(sigma)]).reshape(original_shape)
    x = np.divide(np.subtract(x, np.tile(mu, (m, 1)).transpose()), np.tile(sigma, (m, 1)).transpose())

    return x, mu, sigma

#TODO: fmin_cg is available in scipy, but try to implement it if needed in the future
#def fmincg(func, theta, options, *args):
#    #Read options
#    if not options is None and hasattr(options, 'MaxIter'):
#        length = options.MaxIter
#    else:
#        length = 100


#    RHO = 0.01                             # a bunch of constants for line searches
#    SIG = 0.5                              # RHO and SIG are the constants in the Wolfe-Powell conditions
#    INT = 0.1                              # don't reevaluate within 0.1 of the limit of the current bracket
#    EXT = 3.0                              # extrapolate maximum 3 times the current bracket
#    MAX = 20                               # max 20 function evaluations per line search
#    RATIO = 100                            # maximum allowed slope ratio
     

#    if len(length) == 2:
#       red=length[1]
#       length=length[0]
#    else:
#        red=1

#    S=['Iteration ']

#    i = 0                                           # zero the run length counter
#    ls_failed = 0                            # no previous line search has failed
#    fX = []
#    f1, df1 = func(theta)                     # get function value and gradient
#    i = i if length < 0 else i + 1                                          # count epochs?!
#    s = -df1                                        # search direction is steepest
#    d1 = -s.transpose()*s;                                                 # this is the slope
#    z1 = red/(1-d1);                                  # initial step is red/(|s|+1)

#    while i < abs(length)                                      # while not finished
#      i = i + (length>0);                                      # count iterations?!

#      X0 = theta; f0 = f1; df0 = df1;                   # make a copy of current values
#      X = X + z1*s;                                             # begin line search
#      [f2 df2] = eval(argstr);
#      i = i + (length<0);                                          # count epochs?!
#      d2 = df2'*s;
#      f3 = f1; d3 = d1; z3 = -z1;             # initialize point 3 equal to point 1
#      if length>0, M = MAX; else M = min(MAX, -length-i); end
#      success = 0; limit = -1;                     # initialize quanteties
#      while 1
#        while ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0) 
#          limit = z1;                                         # tighten the bracket
#          if f2 > f1
#            z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);                 # quadratic fit
#          else
#            A = 6*(f2-f3)/z3+3*(d2+d3);                                 # cubic fit
#            B = 3*(f3-f2)-z3*(d3+2*d2);
#            z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;       # numerical error possible - ok!
#          end
#          if isnan(z2) || isinf(z2)
#            z2 = z3/2;                  # if we had a numerical problem then bisect
#          end
#          z2 = max(min(z2, INT*z3),(1-INT)*z3);  # don't accept too close to limits
#          z1 = z1 + z2;                                           # update the step
#          X = X + z2*s;
#          [f2 df2] = eval(argstr);
#          M = M - 1; i = i + (length<0);                           # count epochs?!
#          d2 = df2'*s;
#          z3 = z3-z2;                    # z3 is now relative to the location of z2
#        end
#        if f2 > f1+z1*RHO*d1 || d2 > -SIG*d1
#          break;                                                # this is a failure
#        elseif d2 > SIG*d1
#          success = 1; break;                                             # success
#        elseif M == 0
#          break;                                                          # failure
#        end
#        A = 6*(f2-f3)/z3+3*(d2+d3);                      # make cubic extrapolation
#        B = 3*(f3-f2)-z3*(d3+2*d2);
#        z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));                # num. error possible - ok!
#        if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0      # num prob or wrong sign?
#          if limit < -0.5                                       # if we have no upper limit
#            z2 = z1 * (EXT-1);                                  # the extrapolate the maximum amount
#          else
#            z2 = (limit-z1)/2;                                  # otherwise bisect
#          end
#        elseif (limit > -0.5) && (z2+z1 > limit)                # extraplation beyond max?
#          z2 = (limit-z1)/2;                                    # bisect
#        elseif (limit < -0.5) && (z2+z1 > z1*EXT)               # extrapolation beyond limit
#          z2 = z1*(EXT-1.0);                                    # set to extrapolation limit
#        elseif z2 < -z3*INT
#          z2 = -z3*INT;
#        elseif (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT))    # too close to limit?
#          z2 = (limit-z1)*(1.0-INT);
#        end
#        f3 = f2; d3 = d2; z3 = -z2;                             # set point 3 equal to point 2
#        z1 = z1 + z2; X = X + z2*s;                             # update current estimates
#        [f2 df2] = eval(argstr);
#        M = M - 1; i = i + (length<0);                          # count epochs?!
#        d2 = df2'*s;
#      end                                                       # end of line search

#      if success                                                # if line search succeeded
#        f1 = f2; fX = [fX' f1]';
#        fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
#        s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;             # Polack-Ribiere direction
#        tmp = df1; df1 = df2; df2 = tmp;                        # swap derivatives
#        d2 = df1'*s;
#        if d2 > 0                                               # new slope must be negative
#          s = -df1;                                             # otherwise use steepest direction
#          d2 = -s'*s;    
#        end
#        z1 = z1 * min(RATIO, d1/(d2-realmin));                  # slope ratio but max RATIO
#        d1 = d2;
#        ls_failed = 0;                                          # this line search did not fail
#      else
#        X = X0; f1 = f0; df1 = df0;                             # restore point from before failed line search
#        if ls_failed || i > abs(length)                         # line search failed twice in a row
#          break;                                                # or we ran out of time, so we give up
#        end
#        tmp = df1; df1 = df2; df2 = tmp;                        # swap derivatives
#        s = -df1;                                               # try steepest
#        d1 = -s'*s;
#        z1 = 1/(1-d1);                     
#        ls_failed = 1;                                          # this line search failed
#      end