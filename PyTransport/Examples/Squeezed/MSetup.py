####################################### Setup file for the double quadratic potential evolved with SIGMA Matrices ######################################################

import sympy as sym # we import the sympy package
import math         # we import the math package (not used here, but has useful constants such math.pi which might be needed in other cases)
import sys          # we import the sys module used below 
import os
from pathlib import Path
############################################################################################################################################

# if using an integrated environment we recommend restarting the python console after running this script to make sure updates are found 
from PyTransport.PyTransPy import PyTransSetup # the above commands allows python to find 

PyTransSetup.pathSet()
############################################################################################################################################

nF = 1  # number of fields needed to define the double quadratic potential
nP = 5  # number of parameters needed to define the double quadratic potential
f = sym.symarray('f', nF)   # an array representing the nF fields present for this model
p = sym.symarray('p', nP)   # an array representing the nP parameters needed to define this model (that we might wish to change)
                            # if we don't wish to change them they could be typed explicitly in the potential below

V = p[0]*( p[1] + p[2]*( sym.log(sym.cosh(p[3]*f[0])) + (p[3]+p[4])*f[0] ))  # this is the potential written in sympy notation

# The last argument is for wheter the sympy's simplify is used to on derivatives of the potential and field geometric quantities.
# Caution is recommended as simpy's simplify is known to have bugs. Simplification can increase the speed of numerically evolutions, but at the cost of compiling more slowly.

PyTransSetup.potential(V,nF,nP,True)    # writes this potential and its derivatives into C++ file potential.h when run 
PyTransSetup.compileName("Matt", False)       # this compiles a python module using the C++ code, including the edited potential potential.h file, called PyTransDQuadSigma
                                        # and places it in the location folder ready for use

############################################################################################################################################