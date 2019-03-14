# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#%% Define McLaurin Polynomial for e**x Without Vectorization
tolerance = 1e-6
iterations = 0.
factorial = 1
pol = 0
x = -0.5
term = 1

e = lambda x : np.exp(x)

while(iterations <= 100 and abs(term) >= tolerance):
    term =  x ** iterations / factorial 
    pol += term
    factorial *= iterations + 1
    iterations = iterations + 1
print(e(x)) # Lambda function representing the exponential function at -0.5
print(pol) # Aproximation using McLaurin Polynomial

#%% Define McLaurin Polynomial for e**x With Vectorization
tolerance = 1e-6
iterations = 0.
factorial = 1
pol = 0
x = np.linspace(-1, 1) # Generate a vector with 50 values, at each iteration we will calculate a McLaurin Polynomial term for this 50 values of x
term = 1

e = lambda x : np.exp(x)

while(iterations <= 100 and np.max(abs(term)) >= tolerance):
    term =  x ** iterations / factorial 
    pol += term
    factorial *= iterations + 1
    iterations = iterations + 1
print(e(x)) # Lambda function representing the exponential function at -0.5
print(pol) # Aproximation using McLaurin Polynomial

#%% Define McLaurin Polynomial for e**x With Vectorization inside a function
def funExp(x, tol, maxNumSum):
    factorial = 1
    pol = 0
    term = 1
    iterations = 0.
    
    while(iterations <= maxNumSum and np.max(abs(term)) >= tol):
        term =  x ** iterations / factorial 
        pol += term
        factorial *= iterations + 1
        iterations = iterations + 1
    return pol

x = np.linspace(-1, 1)

plt.plot(x, e(x), label = 'Original exponential')
plt.legend()
plt.show()
plt.figure()

plt.plot(x, funExp(x, 1e-6, 100), label = 'McLaurin Vectorized')
plt.legend()
plt.show()
plt.figure()

plt.plot(x, e(x), label = 'Original exponential', linewidth=2.7)
plt.plot(x, funExp(x, 1e-6, 100), linestyle = 'dashed', label = 'McLaurin Vectorized', linewidth=2.7)
plt.grid(True)
plt.legend()
plt.show()
