# File cec2017.py
# Copyright 2018 Łukasz Neumann <fuine@riseup.net>
# Based on: https://CRAN.R-project.org/package=cec2017
# Distributed under GPL 3 or later

################################################################################
#                                 CEC 2017                                     #
################################################################################
# Purpose   : Evaluate a CEC-2017 benchamark function on a user-defined para-  #
#             meter set                                                        #
################################################################################
# i: integer in [1, 30], with the number of the CEC2017 benchmark function to  #
#    be evalauated                                                             #
# x: numeric, with the parameter set to be evaluated in the benchmark function #
#    Its length MUST be in [2, 10, 20, 30, 50, 100]                            #
################################################################################

from ctypes import cdll, c_double, c_int, POINTER, c_char_p
import numpy as np
import os

c_double_p = POINTER(c_double)


def cec2017(i, x):
    assert isinstance(i, int)
    if i < 1 or i > 30:
        exit("Invalid argument: 'i' should be an integer between 1 and 30 !")

    try:
        sh = x.shape
    except AttributeError:
        exit("x must be a numpy array")
    if len(sh) == 1:
        row = 1
        col = sh[0]
    else:
        row = sh[0]
        col = sh[1]

    if col not in [2, 10, 20, 30, 50, 100]:
        exit("Invalid argument: Only 2, 10, 20, 30, 50 and 100 dimensions/variables are allowed !")

    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "libcec2017C.so"))
    libc = cdll.LoadLibrary(path)
    cec2017 = libc.cec2017
    cec2017.argtypes = [c_char_p, c_int, c_double_p, c_int, c_int, c_double_p]

    x = x.astype(float, order='C')

    f = np.zeros(row)
    f = f.astype(float, order='C')

    cec2017(
        #  root_path("data", "cec2017").encode(),
        os.path.join('..', 'data', 'cec2017').encode(),
        i,
        x.ctypes.data_as(c_double_p),
        row,
        col,
        f.ctypes.data_as(c_double_p))

    #  __import__('pdb').set_trace()
    return f[0]


if __name__ == "__main__":
    print(cec2017(1, np.zeros(2)))
