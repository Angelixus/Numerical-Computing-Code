
import numpy as np
import matplotlib.pyplot as plt
import re
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
def newton(f, df, x0, tol=1e-12, maxiter=100):
    newGuess = x0 - (f(x0) / df(x0))
    condition = x0 - newGuess
    i = 0
    while(i < maxiter and abs(condition) > tol):
        x0 = newGuess
        newGuess = x0  - (f(x0) / df(x0))
        i+=1
        condition = x0 - newGuess
    return (newGuess, i)

f = lambda x : x**3 - 10*x**2 + 5
df = lambda x : 3*x**2 - 20*x

tuple1 = newton(f, df, -1)
tuple2 = newton(f, df, 1)
tuple3 = newton(f, df, 10)
print("%.16f %d" %(tuple1[0], tuple1[1]))
print("%.16f %d" %(tuple2[0], tuple2[1]))
print("%.16f %d" %(tuple3[0], tuple3[1]))

xPoints = np.linspace(-1 ,10.5)
yPoints = []
yPoints += [f(i) for i in xPoints] # Generate the image for all the xPoints and store them on yPoints
plt.plot(xPoints, np.zeros_like(xPoints), 'g')

plt.plot(np.zeros_like(yPoints), yPoints, 'g')
plt.plot(xPoints, yPoints, 'k' ,label = 'Function')
plt.plot(tuple1[0], 0, "ro")
plt.plot(tuple2[0], 0, "ro")
plt.plot(tuple3[0], 0, "ro")
plt.show()

#%% Proposed Exercise 1 (Use bisectionN defined above)
f = lambda x : (x**4) + (2*(x**3)) - (7*(x**2)) + (3)
pointsF = bisectionN(f, -15, 15)

xPoints = np.linspace(-4.5 ,2.5)
yPoints = []
yPoints += [f(i) for i in xPoints]

plt.plot(xPoints, np.zeros_like(xPoints), 'k')
plt.plot(np.zeros_like(yPoints), yPoints, 'g')

plt.plot(xPoints, yPoints, 'b' ,label = 'Function')

for point in pointsF:
    plt.plot(point[0], 0, 'ro')
plt.show()
#%% Proposed Exercise 2