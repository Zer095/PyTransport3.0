####################################### Setup file for a generic model ######################################################
import sympy as sym # we import the sympy package
import math         # we import the math package (not used here, but has useful constants such math.pi which might be needed in other cases)
import sys          # we import the sys module used for import
import os           # we import the os module used for import
from pathlib import Path
############################################################################################################################################

# if using an integrated environment we recommend restarting the python console after running this script to make sure updates are found 

location = os.path.dirname(__file__)          # get the current folder
location = Path(location)                      # transform it in a path 
location = location.parent.absolute()                       # each .parent.absolute() goes back to the parent folder up to NewPyTransport folder
location = os.path.join(location, 'PyTransPy')            # this should be the location of the PyTransPy folder
sys.path.append(location)   # we add this location to the python PATH

import PyTransSetup # the above commands allows python to find it

PyTransSetup.pathSet()      # add all the needed files to the python PATH
############################################################################################################################################

nF = 2  # number of fields needed to define the double quadratic potential
nP = 2  # number of parameters needed to define the double quadratic potential
f = sym.symarray('f', nF)   # an array representing the nF fields present for this model
p = sym.symarray('p', nP)   # an array representing the nP parameters needed to define this model (that we might wish to change)
                            # if we don't wish to change them they could be typed explicitly in the potential below

V = p[0]*f[0] + p[1]*f[1]  # this is the potential written in sympy notation

# The last argument is for wheter the sympy's simplify is used to on derivatives of the potential and field geometric quantities.
# Caution is recommended as simpy's simplify is known to have bugs. Simplification can increase the speed of numerically evolutions, but at the cost of compiling more slowly.

PyTransSetup.potential(V,nF,nP,True)    # writes this potential and its derivatives into C++ file potential.h when run 
PyTransSetup.compileName("Test")       # this compiles a python module using the C++ code, including the edited potential potential.h file, called PyTransDQuadSigma
                                        # and places it in the location folder ready for use

############################################################################################################################################