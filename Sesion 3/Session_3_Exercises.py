
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
#%% Bisection Method
