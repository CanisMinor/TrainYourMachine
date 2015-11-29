import numpy as np
import math


def heaviside_step(z):
    '''
    function: heaviside_step
    ------------------------
    calculates the Heaviside step function for value z

    :param: z: input value

    :return: value of step function at z
    '''

    if z <= 0.0:
        return 0.0
    else:
        return 1.0


def sigmoid(z):
    '''
    function: sigmoid
    -----------------
    evaluates the sigmoid function on value z

    :param z: value
    :return: sigmoid function value at z
    '''

    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    '''
    function: sigmoid_prime(z)
    --------------------------
    evaluates the first derivative of the sigmoid function at value z

    :param z: value
    :return: first derivative of sigmoid function at z
    '''

    return sigmoid(z) * (1.0 - sigmoid(z))


def rectified_linear(z):
    '''
    function: rectified_linear(z)
    --------------------------
    evaluates the rectified linear function at z

    :param z: value
    :return: value of rectified linear function at z
    '''

    if z <= 0:
        return 0.0
    else:
        return z


def rectified_linear_prime(z):
    '''
    function: rectified_linear_prime(z)
    --------------------------
    evaluates the gradient of the rectified linear function at z

    :param z: value
    :return: gradient of rectified linear function at z
    '''

    if z <= 0:
        return 0.0
    else:
        return 1.0


def tanh(z):
    '''
    function: tanh(z)
    --------------------------
    evaluates the hyperbolic tangent at z

    :param z: value
    :return: hyperbolic tangent at z
    '''

    return (2.0 / (1.0 + math.exp(-2.0 * z))) - 1.0

def tanh_prime(z):
    '''
    function: tanh(z)
    --------------------------
    evaluates the gradient of the hyperbolic tangent at z

    :param z: value
    :return: gradient of hyperbolic tangent at z
    '''

    return 1.0 - (tanh(z) * tanh(z))


def arctan(z):
    '''
    function: arctan(z)
    --------------------------
    evaluates the arcus tangent at z

    :param z: value
    :return: arcus tangent at z
    '''

    return math.atan(z)

def arctan_prime(z):
    '''
    function: arctan_prime(z)
    --------------------------
    evaluates the gradient of the arcus tangent at z

    :param z: value
    :return: gradient of arcus tangent at z
    '''

    return 1.0 / ((z * z) + 1.0)