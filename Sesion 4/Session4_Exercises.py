import numpy as np
import matplotlib.pyplot as plt

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
#%% Secant
def secant(f, x0, x1, tol=1e-12, maxiter=100):
    x2 = x1 - ((f(x1) * (x1 - x0)) / (f(x1) - f(x0)))
    i = 0
    while(i < maxiter and abs(x2 - x1) > tol):
        x0 = x1
        x1 = x2
        x2 = x1 - ((f(x1) * (x1 - x0)) / (f(x1) - f(x0)))
        i+=1
    return (x2, i)

f  = lambda x : (x**3) - (10*(x**2)) + 5

possibleRoots = incrementalSearchAllInOne(f, -1, 10, 0.1)

roots = []
for tupleR in possibleRoots:
    pointIter = secant(f, tupleR[0], tupleR[1])
    point = pointIter[0]
    roots.append(point)
    print("%.16f %d" %(pointIter[0], pointIter[1]))

xPoints = np.linspace(-1, 10)
yPoints = []
yPoints += [f(i) for i in xPoints]

r = np.zeros(3)
r[0] = roots[0]
r[1] = roots[1]
r[2] = roots[2]

plt.plot(xPoints, xPoints * 0, 'k')

plt.plot([0, 0], [-200, 200], 'k') # Asociates the i positions of both arrays (pos 0 with pos 0 and pos 1 with pos 1, never pos 0 with pos 1)

plt.plot(xPoints, yPoints, 'b')
plt.plot(r, r*0, 'ro')
plt.show()