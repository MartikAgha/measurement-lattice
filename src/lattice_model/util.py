#!/usr/bin/python3
import numpy as np

def tau(x, y, z):
    """
    Qubit Naming Function: Converts coordinates into node keys.
    :param x: x-axis node coord
    :param y: y-axis node coord
    :param z: z-axis node coord
    :return:
    """
    return "{}:{}:{}".format(x, y, z)

def fermi_scaling_function(x, a, b):
    """
    Fermi-like scaling function for fitting.
    :param x: Value of probability.
    :param a: Argument displacement.
    :param b: Argument scaler.
    :return:
    """
    return 1/(np.exp((a - x)/b) + 1)