import numpy as np
import matplotlib.pyplot as plt

# Using python functions
#%% Data
x = np.array([2., 3., 4., 5., 6.])
y = np.array([2., 6., 5., 5., 6.])

#%% Coefficients of the interpolation polynomial
polynomial = np.polyfit(x, y, len(x) - 1)

#%% Plotting area
xx = np.linspace(min(x), max(x)) # We use min and max because we do INTERPOLATION, we are defining the polynomial in range [min(x), max(x)] (only valid here)
yy = np.polyval(polynomial, xx)

plt.plot(x, y, 'ro', label='nodes')
plt.plot(xx, yy, label = 'Int. poly')
plt.legend()
plt.show()

#Own Implementation
#%% Vandermonde way

def Vandermonde(x):
    dimension = len(x)
    vanMatrix = np.zeros((dimension, dimension))

    for j in range(0, len(x)):
        # Take the columns (iterate through columns)
        vanMatrix[:, j] = x[:]**j
    
    return vanMatrix
    
def polVandermonde(x, y):
    vanMatrix = Vandermonde(x)
    return np.linalg.solve(vanMatrix, y)[::-1]

#%% Data
x = np.array([2., 3., 4., 5., 6.])
y = np.array([2., 6., 5., 5., 6.])

#%% Coefficients of the interpolation polynomial
polynomial = polVandermonde(x, y)

print(polynomial)