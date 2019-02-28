import numpy as np
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-

"""
Divide the code on diferent sections with '#%%' so we will only execute the part enclosed
Use Ctrl + i to get the function documentation
"""
#%% plot exponential function
f = lambda x: np.exp(x) # Lambda calculus stuff, we define that on f we are going to store the code of the exponential passing x as parameter
x = np.linspace(-1, 1) # Create array of numpy, different to list (for example if we use + operator with numpy arrays we add the elements one by one, with list we concatenate them, reffer to Notions -> Basic Operations -> In[7])
y = f(x) # Lambda function invocation with an array to get all values, we pass an array and exponential is applied to every point

ox = np.zeros_like(x) # Calculate x axis, just fills an array of the same size as x with zeroes

plt.plot(x, y) # Plots only the points, not joining them 
plt.plot(x, ox, 'k', label = 'OX axis') # Show x axis in black
plt.legend() # Show the name of the x axis
plt.title('Exponential Function') # Give a title to the plot
plt.show() # The figure ends here
#%% Taylor Polynomial / General form = f(x) = f(a) + ((f'(a) * (x - a)) / 1!) + ((f''(a) * (x - a)^2) / 2!) + ... +  ((f nth derivative(a) * (x - a)^n) / n!)
f = lambda x: np.exp(x)
x0 = 0.5
pol = 0.
factorial = 1.

for i in range(3):
    term = x0 ** i / factorial
    pol += term
    
    factorial *= i + 1
print('Taylor`s Polynomial of order 3(0.5) = ', pol)
print('exponential(0.5) = ', f(0.5))
#%% Taylor Polynomial with arrays
f = lambda x: np.exp(x)
x = np.linspace(-3, 3)
pol = 0.
factorial = 1.

for i in range(3):
    term = x ** i / factorial # Here term will be transformed into a numpy array (at the right we have an array) / we are also doing Vectorization, we are working with 50 points at the same time
    pol += term # Vectorization, we are working with 50 points at the same time
    
    factorial *= i + 1
plt.plot(x, f(x), label = 'Exponential function')
plt.plot(x, pol, label = 'Taylor´s polynomial')
plt.legend()
plt.show()
#%% Taylor Polynomial
# Define a function to calculate the polynomial
def taylor(x, degree):
    f = lambda x: np.exp(x)
    x = np.linspace(-3, 3)
    pol = 0.
    factorial = 1.
    
    for i in range(degree + 1):
        term = x ** i / factorial # Here term will be transformed into a numpy array (at the right we have an array) / we are also doing Vectorization, we are working with 50 points at the same time
        pol += term # Vectorization, we are working with 50 points at the same time
        
        factorial *= i + 1
    return pol

#Invocation
f = lambda x: np.exp(x)
x = np.linspace(-3, 3)
plt.plot(x, f(x), label = 'Exponential function')
plt.plot(x, taylor(x, 2), label = 'Taylor´s polynomial using custom function')
plt.legend()
plt.show()
#%% Taylor Polynomial
# Define a function to calculate the polynomial
def taylor2(x, degree):
    f = lambda x: np.exp(x)
    x = np.linspace(-3, 3)
    pol = 0.
    factorial = 1.
    
    for i in range(degree + 1):
        term = x ** i / factorial # Here term will be transformed into a numpy array (at the right we have an array) / we are also doing Vectorization, we are working with 50 points at the same time
        pol += term # Vectorization, we are working with 50 points at the same time
        
        factorial *= i + 1
    return pol

f = lambda x: np.exp(x)
x = np.linspace(-3, 3)
plt.plot(x, f(x), label = 'Exponential function')
for degree in range(5):
    plt.plot(x, taylor2(x, degree), label = 'Taylor´s polynomial using custom function ' + str(degree))
   # plt.pause(2)
plt.legend()
plt.show()
#%% Vectorization!!!! IMPORTANT
import time
z = np.linspace(-1, 1, 1000000)
yz = np.zeros_like(z)

t0 = time.time()
for i in range(len(z)):
    yz[i] = f(z[i])
t1 = time.time()
elapsed = t1 - t0
print('Without vectorization', elapsed)

t0 = time.time()
yz = f(z)
t1 = time.time()
t = t1 - t0
print('With vectorization', t)
#%% Example using math
import math as mt
np.sin(x)
# mt.sin(x) Error!! Cant put a numpy array into math sin