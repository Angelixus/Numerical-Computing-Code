# coins=coins.convert('L') Convert image to Black and White

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import sympy as sym

"""
We get the approximate derivative values for each x point we have on the array (based on the definition of derivative by limits and h)
"""
def derivatives(f, a, b, h):
    x = np.arange(a+h, b, h) # arange vs linspace: arange creates a vector of equaly spaced numbers but it works until the previous value of the end (until b-1 point)
    df_forward = (f(x + h) - f(x)) / h
    df_centered = (f(x + h) - f(x - h)) / (2*h)
    df_backward = (f(x) - f(x - h)) / h

    return (x, df_forward, df_backward, df_centered)

def executeEx1(h):
    f = lambda x : np.exp(x)
    df = lambda x: np.exp(x)
    a = 0.
    b = 1.

    result = derivatives(f, a, b, h)
    error_forward = abs(result[1]-df(result[0])) # Our aproximation - the exact value
    EaForward = norm(error_forward)
    ErForward = EaForward / norm(df(result[0]))

    error_backward = abs(result[2] - df(result[0]))
    EaBackward = norm(error_backward)
    ErBackward = EaBackward / norm(df(result[0]))

    error_center = abs(result[3] - df(result[0]))
    EaCenter = norm(error_center)
    ErCenter = EaCenter / norm(df(result[0]))

    print('GLOBAL ERRORS')
    print('h: ' + str(h) + '\n')
    print('E(df_f): %.6e\n' %ErForward)
    print('E(df_b): %.6e\n' %ErBackward)
    print('E(df_c): %.6e\n' %ErCenter)

    plt.title(r'Derivatives of e$^x$ with h = ' + str(h))
    plt.plot(result[0], df(result[0]), 'r--', label='Exact Derivative')
    plt.plot(result[0], result[1], 'b', label='Forward Derivative')
    plt.plot(result[0], result[2], 'crimson', label='Backward Derivative')
    plt.plot(result[0], result[3], 'g', label='Centered Derivative')
    plt.legend()
    plt.show()

    plt.title('Errors for h = ' + str(h))
    plt.plot(result[0], error_forward, 'b', label='Forward')
    plt.plot(result[0], error_backward, 'darkorange', label='backward')
    plt.plot(result[0], error_center, 'g', label='centered')
    plt.legend()
    plt.show()

def executeEx2(h):
    f = lambda x : 1 / x
    df = lambda x: -1 / (x**2)
    a = 0.2
    b = 1.2

    result = derivatives_a(f, a, b, h)
    error_forward = abs(result[1]-df(result[0])) # Our aproximation - the exact value
    EaForward = norm(error_forward)
    ErForward = EaForward / norm(df(result[0]))

    print('GLOBAL ERRORS')
    print('h: ' + str(h) + '\n')
    print('E(df_f): %.6e\n' %ErForward)

    plt.title(r'Derivatives of 1/x with h = ' + str(h))
    plt.plot(result[0], df(result[0]), 'r', linewidth=4, label='Exact Derivative')
    plt.plot(result[0], result[1], 'b--', label='Forward Derivative')
    plt.legend()
    plt.show()

    result = derivatives_b(f, a, b, h)
    error_forward = abs(result[1]-df(result[0])) # Our aproximation - the exact value
    EaForward = norm(error_forward)
    ErForward = EaForward / norm(df(result[0]))

    print('GLOBAL ERRORS')
    print('h: ' + str(h) + '\n')
    print('E(df_f): %.6e\n' %ErForward)

    plt.title(r'Derivatives of 1/x with h = ' + str(h))
    plt.plot(result[0], df(result[0]), 'r', linewidth=4, label='Exact Derivative')
    plt.plot(result[0], result[1], 'b--', label='Forward Derivative')
    plt.legend()
    plt.show()

def executeEx3():
    f = lambda x: np.sin(2*np.pi*x)
    x = sym.Symbol('x', real=True)
    f_sim = sym.sin(2*sym.pi*x)
    df_sim=sym.diff(f_sim, x)
    d2f_sim = sym.diff(df_sim, x)

    lambdaFuncDer = sym.lambdify([x],d2f_sim, 'numpy')
    lambdaFunc = sym.lambdify([x],f_sim, 'numpy')
    a = 0
    b = 1
    h=0.01

    result = derivative2(lambdaFunc, a, b, h)
    error_center = abs(result[1]-lambdaFuncDer(result[0])) # Our aproximation - the exact value
    EaCenter = norm(error_center)
    ErCenter = EaCenter / norm(lambdaFuncDer(result[0]))

    print('GLOBAL ERRORS')
    print('h: ' + str(h) + '\n')
    print('E(df_f): %.6e\n' %ErCenter)

    plt.title(r'Second derivative of sin(2πx) with h = ' + str(h))
    plt.plot(result[0], lambdaFuncDer(result[0]), 'r', linewidth=4, label='Exact Derivative')
    plt.plot(result[0], result[1], 'b--', label='Centered Derivative')
    plt.legend()
    plt.show()

def derivativesO1(f, a, b, h):
    df_f = (f(a + h) - f(a)) / h
    df_b = (f(b) - f(b - h)) / h
    return (df_f, df_b)

def derivativesO2(f, a, b, h):
    df_f = (-3 * f(a) + 4*f(a + h) - f(a + 2*h)) / (2*h)
    df_b = (f(b - 2*h) - 4*f(b - h) + 3*f(b)) / (2*h)
    return (df_f, df_b)

"""
Use order 1 functions to compute the borders
"""
def derivatives_a(f, a, b, h):
    x = np.arange(a, b+h, h) # We use b+h in order to have [a, b] in order to be able to compute the forward and backward at a and b respectively

    df_centered = derivatives(f, a - h, b + h, h)[3]
    df_centered[0] = derivativesO1(f, a, b, h)[0]# Progressive order 1
    df_centered[-1] = derivativesO1(f, a, b, h)[1]# Regressive order 1
    return x, df_centered

"""
Use order 2 functions to compute the borders
"""
def derivatives_b(f, a, b, h):
    x = np.arange(a, b+h, h) # We use b+h in order to have [a, b] in order to be able to compute the forward and backward at a and b respectively

    df_centered = derivatives(f, a - h, b + h, h)[3]
    df_centered[0] = derivativesO2(f, a, b, h)[0]# Progressive order 2
    df_centered[-1] = derivativesO2(f, a, b, h)[1]# Regressive order 2
    return x, df_centered

def derivative2(f,a,b,h):
    x = np.arange(a+h, b, h)
    df_centered = (f(x + h) - 2*f(x) + f(x - h)) / (h**2)
    return x, df_centered

#%% Exercise 1
executeEx1(0.1)
executeEx1(0.01)

#%% Exercise 2
executeEx2(0.01)
#%% Exercise 3
executeEx3()