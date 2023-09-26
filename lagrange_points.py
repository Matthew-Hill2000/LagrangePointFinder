# Standard Library
import sys
from typing import Union, Callable
from pathlib import Path

# Third Party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import sympy as sym
import scipy.optimize as spo


def effective_potential(xy):
    """
    Returns the potential given two masses
    """

    mu = 0.2
    
    x = xy[0]
    y = xy[1]

    m1 = 1 - mu
    m2 = mu

    r1_2 = (x + m2)**2 + y**2 
    r2_2 = (x - m1)**2 + y**2

    U = -1/2 * (x**2 + y**2) - m1/np.sqrt(r1_2) - m2/np.sqrt(r2_2)
    return -1*U

def main():
    xy_start = [0.3, 0.2]

    result = spo.minimize(effective_potential, xy_start, options={"disp": True})

    print("code")
    if result.success:
        print("Success!")
        xy = result.x
        x = xy[0]
        y = xy[1]
        print(f"x = {x}, y = {y} U = {result.fun}")

main()