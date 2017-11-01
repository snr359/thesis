# This is an implementation of CoCO's fitness functions as defined in
# http://coco.lri.fr/downloads/download15.01/bbobdocnoisyfunctions.pdf

import random
import sys
import math
import scipy.optimize


# PRIMARY FUNCTION
def get_fitness(x, function_name):
    # returns the adjusted fitness by running the genome x on the funciton indicated by function_name
    # always returns a fitness directly related to solution quality (i.e. never inversely related)
    if function_name == 'rosenbrock_moderate_uniform_noise':
        val = rosenbrock_moderate_uniform_noise(x)
        fitness = -1 * val
        return fitness
    elif function_name == 'rastrigin_moderate_uniform_noise':
        val = rastrigin_moderate_uniform_noise(x)
        fitness = -1 * val
        return fitness
    elif function_name == 'trap':
        fitness = 0
        i = 0
        k = 7
        while i < len(x):
            fitness += trap(x[i:(i+k)], k)
            i += k
        return fitness
    elif function_name == 'deceptive_trap':
        fitness = 0
        i = 0
        k = 7
        s = 2
        while i < len(x):
            fitness += ((k-s)%s) + trap(x[i:(i+k)], k) / s
            i += k
        return fitness
    elif function_name == 'hierarchical_if_and_only_if':
        fitness, _ = hiff(x)
        return fitness
    elif function_name == 'bbob_rastrigin':
        val = bbob_rastrigin(x)
        fitness = -1 * val
        return fitness

    # base case: function_name not found
    else:
        print('ERROR: function name {0} not found. Defaulting to rosenbrock_moderate_uniform_noise'.format(function_name))
        val = rosenbrock_moderate_uniform_noise(x)
        fitness = -1 * val
        return fitness

def get_optimum(x, function_name):
    if function_name == 'rosenbrock_moderate_uniform_noise':
        return 0
    elif function_name == 'rastrigin_moderate_uniform_noise':
        return 0
    elif function_name == 'trap':
        return len(x)
    elif function_name == 'deceptive_trap':
        k = 7
        s = 2
        return sum((((k-s)%s)+k)/s for _ in range(len(x)//k))
    elif function_name == 'hierarchical_if_and_only_if':
        n = len(x) // 2
        fitness = 0
        d = 1
        while n > 0:
            fitness += (2**(d+1)-1) * n
            n = n//2
            d += 1
        return fitness

# BBOB FUNCTIONS

def bbob_rastrigin(x):
    if not hasattr(bbob_rastrigin, 'fun'):
        import cocoex
        dim = len(x)
        suite = cocoex.Suite('bbob',
                             'year:2017',
                             'dimensions: {0} function_indices: 15 instance_indices: 1'.format(dim))
        for fun in suite:
            bbob_rastrigin.fun = fun
            break

    return bbob_rastrigin.fun(x)

# NOISY FUNCTIONS

def rosenbrock_moderate_uniform_noise(x, f_opt=0):
    D = len(x)
    alpha = 0.01 * (0.49 + 1/D)
    beta = 0.01
    # return uniform_noise(rosenbrock(x, f_opt), alpha, beta) + function_penalty(x)
    return uniform_noise(rosenbrock(x, f_opt), alpha, beta)

def rastrigin_moderate_uniform_noise(x):
    D = len(x)
    alpha = 0.01 * (0.49 + 1/D)
    beta = 0.01
    return uniform_noise(rastrigin(x), alpha, beta)

# UTILITIES

def function_penalty(x):
    return 100 * sum(max(0, abs(xi) - 5)**2 for xi in x)

# NOISE FUNCTIONS

def uniform_noise(f, alpha, beta):
    # The function must be below this threshold to be returned undisturbed
    disturbance_threshold = 10**-8

    if f >= disturbance_threshold:
        eps = sys.float_info.epsilon
        f_noisy = f * random.random()**beta * max(1, (10**9 / (f + eps)) ** (alpha * random.random()))
        return f_noisy + 1.01**disturbance_threshold
    else:
        return f


# BASE FUNCTIONS

def rosenbrock(x, f_opt=0):
    return scipy.optimize.rosen(x) + f_opt

def rastrigin(x, a=10):
    return a*len(x) + sum(xi**2 - a*math.cos(2*math.pi*xi) for xi in x)

def trap(t, k):
    fitness = 0
    t_sum = sum(t)
    if t_sum == k:
        fitness += k
    else:
        fitness += k - 1 - t_sum
    return fitness

def hiff(x):
    # returns (fitness, value), where fitness is the fitness of the subtree x and value is the boolean value of the tree,
    # or '-' if the trees do not match
    if len(x) == 1:
        return 0, x[0]
    else:
        split = int(len(x) / 2)
        fitnessLeft, valueLeft = hiff(x[:split])
        fitnessRight, valueRight = hiff(x[split:])
        if valueRight != valueLeft or valueLeft == '-' or valueRight == '-':
            return fitnessLeft + fitnessRight, '-'
        else:
            return len(x) + fitnessLeft + fitnessRight, valueLeft

# TESTING MAIN

if __name__ == '__main__':
    print(rosenbrock_moderate_uniform_noise([1]*5))