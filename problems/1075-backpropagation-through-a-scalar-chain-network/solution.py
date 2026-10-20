import numpy as np
import math

def backprop_chain(a0: float, y: float, w: list, b: list) -> dict:
    """
    Compute gradients of C = (a_L - y)^2 with respect to each weight and bias
    in a chain network where every layer has a single sigmoid neuron.

    Args:
        a0: input activation (scalar)
        y: target value (scalar)
        w: list of L weights
        b: list of L biases

    Returns:
        Dictionary with keys 'dW' and 'dB', each a list of L floats rounded to 6 places.
    """
    # Your code here
    L = len(w)

    # Forward pass
    a = [a0]
    z = []

    for l in range(L):
        z_l = w[l] * a[l] + b[l]
        a_l = 1 / (1 + math.exp(-z_l))

        z.append(z_l)
        a.append(a_l)

    # Backward pass
    dW = [0.0] * L
    dB = [0.0] * L

    # dC/da_L
    delta = 2 * (a[L] - y)

    for l in range(L - 1, -1, -1):

        # sigmoid derivative
        sigmoid_derivative = a[l + 1] * (1 - a[l + 1])

        # dC/dz_l
        dz = delta * sigmoid_derivative

        # dC/dw_l = dC/dz_l * dz_l/dw_l
        dW[l] = dz * a[l]

        # dC/db_l = dC/dz_l
        dB[l] = dz

        # Propagate to previous activation
        delta = dz * w[l]

    return {
        "dW": [round(x, 6) for x in dW],
        "dB": [round(x, 6) for x in dB]
    }
