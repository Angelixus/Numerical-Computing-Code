import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

#%% Vandermonde aproximation 1
def Vandermonde(x, degree):
    dimensionX = len(x)
    vanMatrix = np.zeros((dimensionX, degree))

    for j in range(0, degree):
        # Take the columns (iterate through columns)
        vanMatrix[:, j] = x[:]**j
    
    return vanMatrix

def TransposeMatrix(x):
    res = np.zeros((len(x[0]), len(x)))
    for i in range(len(x)):
        for j in range(len(x[0])):
            res[j, i] = x[i, j]

    return res

def aproximate1(x, y, degree):
    V = Vandermonde(x, degree + 1)
    VTranspose = TransposeMatrix(V)
    # System A: p = b
    A = np.dot(VTranspose, V)

    b = np.dot(VTranspose, y)

    p = np.linalg.solve(A, b)
    # Solve system 

    return np.flip(p) # fliped

x = np.linspace(-1, 1, 5) # 5 nodes
f = lambda x : np.cos(x)
y=f(x)
degree = 2

p = aproximate1(x, y, degree)
xp = np.linspace(min(x), max(x))
yp = np.polyval(p, xp) # p are the unknowns of our polynomial, the polynomial is defined as: a2*x**2 + a1*x + a0, we only have to give 
# the values of a2, a1, a0 in this order. Moreover, we have to evaluate on the given range because is there where the polynomial is valid

plt.title('Exercise 1')
plt.plot(x, y, 'ro', label='nodes')
plt.plot(xp, yp, label='Polynomial')
plt.legend()
plt.show()

#%% Exercise 2
x = np.linspace(-1, 1, 10)
f = lambda x : np.cos(np.arctan(x)) - (np.exp(x**2)*np.log(x + 2))
y=f(x)
degree=4
p = aproximate1(x, y, degree)

xp = np.linspace(min(x), max(x))
yp = np.polyval(p, xp)

plt.title('Exercise 2')
plt.plot(x, y, 'ro', label='nodes')
plt.plot(xp, yp, label='Polynomial')
plt.legend()
plt.show()
#%% How to integrate with quad thing
#I = quad(g, a1, b1)[0] # g is a lambda function, a1 and b1 are the intervals, so integral of g on [a1, b1], returns a tuple, first value is the integral and the second one the error bound

#%% Exercise 3
def aproximate2(f, a1, b1, degree):
    A = np.zeros((degree + 1, degree + 1))
    B = np.zeros((degree + 1, 1))
    for i in range(len(A)):
        for j in range(len(A[i])):
            g = lambda x : x**(i+j)
            A[i, j] = quad(g, a1, b1)[0]

    for i in range(len(B)):
        g = lambda x : f(x)*x**(i)
        B[i, 0] = quad(g, a1, b1)[0]
    
    p = np.flip(np.linalg.solve(A, B))

    print('Coefficient Matrix:\n', A)
    print('RIght hand side matrix:\n', np.transpose(B))
    print('Polynomial Coefficients (fliped):\n', np.transpose(p))
    return p

f = lambda x : np.cos(x)
a1 = -1
b1 = 1
degree = 2

p = aproximate2(f, a1, b1, degree)
x = np.linspace(-1, 1)
xp = np.linspace(min(x), max(x))
yp = np.polyval(p, xp)

plt.title('Exercise 3')
plt.plot(x, f(x), label='Real function')
plt.plot(xp, yp, label='Aprox')
plt.legend()
plt.show()

Er = (np.linalg.norm(f(x) - yp)) / (np.linalg.norm(f(x)))
print('Er = ', Er)

print('\n-------------------------------------------\n')
#%% Exercise 4
f = lambda x : np.cos(np.arctan(x)) - (np.exp(x**2)*np.log(x + 2))
a1 = -1
b1 = 1
degree = 4

p = aproximate2(f, a1, b1, degree)
x = np.linspace(-1, 1)
xp = np.linspace(min(x), max(x))
yp = np.polyval(p, xp)

plt.title('Exercise 4')
plt.plot(x, f(x), label='Real function')
plt.plot(xp, yp, label='Aprox')
plt.legend()
plt.show()

Er = (np.linalg.norm(f(x) - yp)) / (np.linalg.norm(f(x)))
print('Er = ', Er)