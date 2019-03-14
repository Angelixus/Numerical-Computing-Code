import numpy as np
import matplotlib.pyplot as plt

#%% Vandermonde aproximation 1
def Vandermonde(x, degree):
    dimensionX = len(x)
    vanMatrix = np.zeros((dimensionX, degree))

    for j in range(0, degree):
        # Take the columns (iterate through columns)
        vanMatrix[:, j] = x[:]**j
    
    return vanMatrix

def aproximate1(x, y, degree):
    V = Vandermonde(x, degree)
    # System A: p = b
    A = np.dot(V.T, V)

    b = 2

    # Solve system 

    return p # fliped

x = np.linspace(-1, 1, 5) # 5 nodes
f = lambda x : np.cos(x)
y=f(x)
degree = 2

p = aproximate1(x. y. degree)

# Plot
plt.plot()

#%% Vandermonde aproximation 2

def aproximate2(x, y, degree):
    n = len(x)
    A = np.zeros((n, n))
    b = np.zeros(n)
    # Fill A and B

    for i in range(n):
        b[i] = sum(x*y) # Not this
        for j in range(n):
            A[i, j] = sum(x**2) # Not this
