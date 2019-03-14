import numpy as np
import matplotlib.pyplot as plt
#%% Exercise 2

def lagrange_fundamental(x,k,z):
        n = len(x)
        yz = 1.
        for i in range(n):
                if i != k:
                        yz *= (z - x[i])/(x[k]-x[i])
        return yz

x = np.array([2., 3., 4., 5., 6.])
y = np.array([2., 6., 5., 5., 6.])

k = 0
z = np.linspace(min(x), max(x))

for i in range (len(x)):
        plt.title('L' + str(i))
        plt.plot([2, 6], [0, 0], 'k', label='x-axis')
        res=lagrange_fundamental(x, i, z)
        plt.plot(z, res, label='lagrange'+str(i))

        points = np.zeros_like(x)
        points[i] = 1
        
        plt.plot(x, points, 'ro')
        plt.legend()
        plt.show()

#%% Exercise 3
def lagrange_polynomial(x,y,z):
        n = len(y)
        res = 0.
        for i in range(n):
                res = res + y[i] * lagrange_fundamental(x, i, z)
        return res

res1 = lagrange_polynomial(x, y, z)

plt.title('First lagrange polynomial')
plt.plot(z, res1, label='lagrange polynomial 1')
plt.plot(x, y, 'ro' )
plt.legend()
plt.show()

x1 = np.array([0.,1.,2.,3.,4.,5.,6.])
y1 = np.array([3.,5.,6.,5.,4.,4.,5.])
z1 = np.linspace(min(x1), max(x1))

res2 = lagrange_polynomial(x1, y1, z1)

plt.title('Second lagrange polynomial')
plt.plot(z1, res2, label='lagrange polynomial 2')
plt.plot(x1, y1, 'ro')
plt.legend()
plt.show()