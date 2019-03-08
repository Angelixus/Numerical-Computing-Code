import numpy as np
import matplotlib.pyplot as plt
#%% Exercise 2
x = np.array([2., 3., 4., 5., 6.])
y = np.array([2., 6., 5., 5., 6.])

n = len(x)
k = 1
z = np.linspace(min(x), max(x))

yz = 1.
for i in range(n):
    if i!= k:
        yz *= (z - x[i])/(x[k]-x[i])

plt.plot(z, yz, label='lagrange'+str(k))
plt.legend()
plt.show()