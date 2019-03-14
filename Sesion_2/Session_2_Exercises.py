# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Los def son lo que se "compila primero" asi que funciones escritas encima de la declaracion de variables globales usando estas funcionan perfectamente, entonces da igual donde hagamos el def
#%% Exercise 1, Horner Method Without Vectorization    
def horner(p, x):
    out = np.zeros_like(p)
    out[0] = p[0]
    for i in range(1, len(p)):
        out[i] = p[i] + out[i - 1] * x
    return out

p = np.array([1., -1., 2., -3.,  5., -2.])
x0 = 1.

r = np.array([5., -3.,  1., -1., -4.,  0.,  0., 3.])
x1 = -1.

print(horner(p, x0)[-1])
print(horner(r, x1)[-1])
#%% Excercise 2, Horner Method With Vectorization
def hornerV(polynomial, points):
    out = np.zeros_like(points)
    aux = np.zeros_like(polynomial)
    aux[0] = polynomial[0]
    
    for j in range(0, len(points)):
        point = points[j]
        for i in range(1, len(polynomial)):
            aux[i] = polynomial[i] + aux[i - 1] * point
        out[j] = aux[-1]
    return out

p = np.array([1., -1., 2., -3.,  5., -2.])
x = np.linspace(-1, 1)

y = hornerV(p, x)

plt.plot(x, y)
plt.show()

plt.figure()
r = np.array([5., -3., 1., -1., -4., 0., 0., 3.])
z = hornerV(r, x)
plt.plot(x, z)
plt.show()
#%% Exercise 3, Higher order derivatives
def polDer(p, x0):
    res = np.zeros_like(p)
    out = np.zeros_like(p)
    out[0] = p[0]
    for i in range(1, len(p)):
        out[i] = p[i] + out[i - 1] * x0
    res[0] = out[-1]
    resArray = computeDerivatives(res, len(res) - 1, 1, out, out, x0)
    
    for i in range(0, len(resArray)):
        factorialAux = factorial(i)
        resArray[i] = resArray[i] * factorialAux
    return resArray
    
def computeDerivatives(array, lengthOf, currentIndex, out, p, x):
    if(lengthOf == 1):
        array[currentIndex] = out[-2]
        return array
    
    for i in range(1, lengthOf):
        out[i] = p[i] + out[i - 1] * x
        
        
    array[currentIndex] = out[-2]
    currentIndex = currentIndex + 1
    return computeDerivatives(array, lengthOf - 1, currentIndex, out[0:len(out) - 1], out, x)

def factorial(x):
    if(x == 0 or x == 1):
        return 1
    return x * factorial(x - 1)
    
print(polDer(p, x0))
np.set_printoptions(suppress=True)
print(polDer(r, x1))
#%% Exercise 3 Vectorized
def derivativesH(p, x0):
    out = np.zeros_like(x0)
    aux = np.zeros_like(p)
    
    aux[0] = p[0]
    for j in range(0, len(x0)):
        point = x0[j]
        for i in range(1, len(p)):
            aux[i] = p[i] + aux[i - 1] * point
        out[j] = aux[-1]
   
def factorialH(x):
    if(x == 0 or x == 1):
        return 1
    return x * factorialH(x - 1)

p = np.array([1., -1., 2., -3.,  5., -2.])
x0 = 1.

r = np.array([5., -3.,  1., -1., -4.,  0.,  0., 3.])
x1 = -1.

derivativesH(p, np.linspace(-1, 1))