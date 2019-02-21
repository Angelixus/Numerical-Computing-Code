# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:05:49 2019

@author: UO263930
"""

import numpy as np
import matplotlib.pyplot as plt
#%% 
originalf = lambda x : np.log(x) + x
x = np.linspace(0.1, 2)
plt.plot(x, originalf(x), 'b')
plt.plot(x, 0*x, 'k')
#%% Bisection
a = 0.25; b = 1.
for i in range(50):
    m = (a+b)/2
    if(originalf(a) * originalf(m) < 0):
        b = m
    else:
        a = m
print('Zero of the equation = ', m)
plt.plot(m, 0, 'ro') # Zero of the equation
#%% Check that our aproximation m gives as an image the intersection between y = x and e**-x = x
# Advantages: It is an easy algorithm that it is versatile enough to analize many things on maths!!!!
g = lambda x : np.exp(-x)
plt.plot(x, g(x), 'g') 
plt.plot(x, x, 'k')

x0 = 0.5 # Initial value
for i in range(50):
    x0 = g(x0)

plt.plot(x0, x0, 'ro')