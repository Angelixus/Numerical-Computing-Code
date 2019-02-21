
import numpy as np
import matplotlib.pyplot as ptl
#%% Incremental Search Algorithm
"""
lfunction is a lambda function representing the mathematical function
leftbound is the a of the interval [a,b] we are working over
rightbound is the b of the interval [a,b] we are working over
increment is the step taken at each iteration for both x0 and x1
"""
def incrementalSearch(lfunction, leftbound, rightbound, increment):
    x0 = leftbound
    x1 = leftbound + increment
    
    while(x1 < rightbound):
        if(lfunction(x0) * lfunction(x1) < 0):
            return (x0, x1)
        x0 += increment
        x1 += increment
    return (None, None)

f = lambda x : x**3 - 10*x**2 + 5
a = -15.; b = 15.; dx = 0.1

x0, x1 = incrementalSearch(f, a, b, dx) # Tuple
while x0 != None:
    print("There is a zero in [%.1f, %.1f]" %(x0, x1))
    x0, x1 = incrementalSearch(f, x1, b, dx)

"""
Same as incrementalSearch function but return all the bracketed roots on [a, b]
"""
def incrementalSearchAllInOne(lfunction, leftbound, rightbound, increment):
    x0 = leftbound
    x1 = leftbound + increment

    res = []
    while(x1 < rightbound):
        if(lfunction(x0) * lfunction(x1) < 0):
            res.append((x0, x1))
        x0 += increment
        x1 += increment
    return res

res = incrementalSearchAllInOne(f, a, b, dx)
for tuple in res:
    print("There is a zero in [%.1f, %.1f]" %(tuple[0], tuple[1]))


"""IMPLEMENT INCREMENT RECURSIVELY"""
#%% Bisection Method, only one point
def bisection(f, a, b, maxiter = 100, tol = 1e-12):
    i = 0
    x0 = a
    x1 = b
    while(i  < maxiter and abs(abs(x1 - x0) / 2) > tol):
        m = (x0 + x1) / 2
        if f(a) * f(m) < 0:
            x1 = m
        elif f(m) * f(b) < 0:
            x0 = m
        else:
            return m, i + 1
        i+=1
    return m, i + 1



f1 = lambda x : x**3 - 10*x**2 + 5
a1 = -15.; b1 = 15.; dx = 0.1

print('Founded root: %.5f' % bisection(f1, a1, b1)[0])

#%% Bisection method, n points
def bisectionN(f, a, b, tol = 1e-12, maxiter = 100):

    intervals = incrementalSearchAllInOne(f, a, b, 0.1)

    points = []
    for tuple in intervals:
        points.append(bisection(f, tuple[0], tuple[1], tol = tol, maxiter = maxiter))
    return points

res = bisectionN(f, a, b)
for tuple in res:
    print('Aproximation: %.16f Iterations: %d' %(tuple[0], tuple[1]))

#%% Newton-Raphson's Method


#%% IncremetalAll functional
def incrementalSearchAllInOneFunctional(lfunction, leftbound, rightbound, increment, maxiter = 100):
    x0 = leftbound
    x1 = leftbound + increment
    i = 0
    foundedSol = False
    while(x1 < rightbound):
        solution = (None, None)
        if(lfunction(x0) * lfunction(x1) < 0):
            solution = (x0, x1)
            foundedSol = True
            yield solution
        x0 += increment
        x1 += increment
        i+=1
    if(not(foundedSol)):
        yield None, None

f = lambda x : x**3 - 10*x**2 + 5
a = -15.; b = 15.; dx = 0.1

for x in incrementalSearchAllInOneFunctional(f, a, b, dx): 
    print(x) 