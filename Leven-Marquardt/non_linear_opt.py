"""
non-linear optimization using: 
    1. Newton algorithm
    2. Leven-marquardt algorithm

functions of computing gradient and hessian as well as the Newton method
    is from: <leouieda@gmail.com>
    https://code.google.com/p/heuristic-methods/source/browse/hm/gradient.py

May 22, 2013
By Ambling<ambling.ding@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt


def gradient(func, dims, params, delta):
    """
    Calculate the gradient of func evaluated at params
    """    
        
    grad = np.zeros(dims)
    tmp = np.zeros(dims)

    # Compute the gradient
    #   compute for each dimension seperately
    for i in xrange(dims):
        tmp[i] = delta
        grad[i] = (func(*(params + tmp)) - func(*(params - tmp)))/delta
        tmp[i] = 0

    return grad


def hessian(func, dims, params, delta):
    """
    Calculate the Hessian matrix of func evaluated at params
    """

    hessian = np.zeros((dims, dims))
    tmpi = np.zeros(dims)
    tmpj = np.zeros(dims)

    for i in xrange(dims):
    
        tmpi[i] = delta
        params1 = params + tmpi
        params2 = params - tmpi    
        
        for j in xrange(i, dims):
        
            tmpj[j] = delta
            deriv2 = (func(*(params2 + tmpj)) - func(*(params1 + tmpj)))/delta
            deriv1 = (func(*(params2 - tmpj)) - func(*(params1 - tmpj)))/delta
            hessian[i][j] = (deriv2 - deriv1)/delta
            
            # Since the Hessian is symmetric, spare me some calculations
            hessian[j][i] = hessian[i][j]
            
            tmpj[j] = 0
        
        tmpi[i] = 0
    
    return hessian
    
    
def newton(func, dims, initial, maxit=100, stop=1e-15):
    """
    Newton's method of optimization.
    Note: the derivative of func is calculated numerically
    """
    
    delta = 0.0001
    solution = np.copy(initial)
    estimates = [solution]
    goals = [func(*solution)]
    
    for i in xrange(maxit):
        
        grad = gradient(func, dims, solution, delta)
        H = hessian(func, dims, solution, delta)
        correction = np.linalg.solve(H, grad)
        solution = solution + correction
        estimates.append(solution)
        goals.append(func(*solution))
        
        if abs((goals[i+1] - goals[i])/goals[i]) <= stop:
            break
        
    return solution, goals[-1], np.array(estimates), goals


def leven_marquardt(func, dims, initial, maxit=100, stop=1e-15):
    """
    Leven-Marquardt method of optimization.
    http://www.cad.zju.edu.cn/home/zhx/csmath/lib/exe/fetch.php?\
        media=2011:or-2011-3.pdf
    """
    
    delta = 0.0001
    solution = np.copy(initial)
    estimates = [solution]
    goals = [func(*solution)]

    # initial mu
    mu = delta*10
    mus = [mu]
    # initial q
    qs = goals                
    
    for i in xrange(maxit):
        
        grad = gradient(func, dims, solution, delta)
        H = hessian(func, dims, solution, delta)

        if (grad**2).sum() < stop:
            break;

        is_singular = True
        while is_singular:
            try:
                correction = np.linalg.solve(H+mu*np.eye(H.shape[0]), \
                                            grad)
                is_singular = False
            except Exception, e:
                mu = 4 * mu
                mus.append(mu)

        solution = solution + correction
        estimates.append(solution)
        goals.append(func(*solution))

        q = goals[-1] + (grad*correction).sum() + \
            0.5*np.mat(correction) * np.mat(H) * np.mat(correction).T
        qs.append(q)
        r = (goals[i+1]-goals[i]) / (qs[i+1] - qs[i])
        
        if r <= 0:
            # solution remains unchanged
            solution = solution - correction
        if r < 0.25:
            mu = 4 * mu
            mus.append(mu)
        elif r > 0.75:
            mu = 0.5 * mu
            mus.append(mu)
        else :
            pass
        
        
    return solution, mus, goals[-1], np.array(estimates), goals


def object(x):
    return (x-3)**4 + (x-2)**3 + (x-1)**2 + 1

    # s1 = np.cos(2*np.pi*x)
    # e1 = np.exp(-x)
    # return np.multiply(s1,e1)

    # return np.sin(x)

if __name__ == '__main__':
    solution, mus, result, estimates, goals = \
                                        leven_marquardt(object, 1, [70])
    # solution, result, estimates, goals = \
    #                                     newton(object, 1, [-1])
    print "solution: "+ str(solution)
    # print "mus: "+ str(mus)
    print "result: "+ str(result)
    print "estimates: "+ str(estimates)
    print "goals: "+ str(goals)

    # draw the result
    fig = plt.figure()
    t = np.arange(-100.0, 100.0, 0.02) 
    plt.plot(t, object(t), 'g--', label='(x-3)**4 + (x-2)**3 + (x-1)**2 + 1')
    plt.plot(estimates, object(estimates), 'r-', label='estimates')
    plt.legend()
    plt.grid(True)
    fig.savefig("result.png")
    plt.show()