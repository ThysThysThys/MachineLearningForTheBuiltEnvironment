"""
This example shows a simple implementation of the gradient descent algorithm for minimizing
a simple function: f(x) = x^2 - 4x + 1.
"""

import numpy as np

def gradient_descent(start, function, gradient, learn_rate, max_iter, tol=0.001):
    """
    Performs gradient descent to minimize a given function.

    Parameters:
    start (float): The starting point for the algorithm.
    function (callable): The function to minimize.
    gradient (callable): The gradient of the function.
    learn_rate (float): The learning rate (step size).
    max_iter (int): The maximum number of iterations.
    tol (float): The tolerance for stopping the algorithm.

    Returns:
    float: The point at which the function is minimized.
    """
    x = start  # Initialize the starting point
    for it in range(max_iter):
        diff = learn_rate * gradient(x)  # Calculate the step size
        if np.abs(diff) < tol:  # Check if the step size is smaller than the tolerance
            break  # If yes, stop the algorithm
        print("iteration =", it, "\t\tx =", "{:.5f}".format(x), "\t\tf(x) =", "{:.3f}".format(function(x)))
        x = x - diff  # Update the current point
    return x

# Define the function f(x) = x^2 - 4x + 1
def func(x):
    """
    The function to minimize.

    Parameters:
    x (float): The input value.

    Returns:
    float: The function value at x.
    """
    return x**2 - 4*x + 1

# Define the gradient of the function, which is f'(x) = 2x - 4
def gradient_func(x):
    """
    The gradient of the function.

    Parameters:
    x (float): The input value.

    Returns:
    float: The gradient value at x.
    """
    return 2*x - 4

# Run the gradient descent algorithm starting from x = 9
gradient_descent(9, func, gradient_func, 0.1, 100)