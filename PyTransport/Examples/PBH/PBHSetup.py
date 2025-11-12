####################################### Setup file for the Step Potential for PBHs ###########################################
import sympy as sym
import numpy as np
import math
import sys
import os
from pathlib import Path
############################################################################################################################################
from PyTransport.PyTransPy import PyTransSetup

### Sets potential and compiles PyTransport, users may prefer to do this only once in a separate file (or comment after running below once) ###
nF=1
nP=6
f=sym.symarray('f',nF)
p=sym.symarray('p',nP)

## example step
V = p[0]*(1.0 - p[1]*( sym.tanh( (f[0]-p[2])/p[3]) ) ) * (1.0 - p[4] * f[0]**2/(1+f[0]/p[5]))

PyTransSetup.potential(V,nF,nP) # differentiates this potential and writes this potential and derivatives into c file when run (can be a 
                               # little slow, and so one may not wish to run if recompiling to alater other properties such as tols) 

PyTransSetup.compileName("PBH") # this compiles the module with the new potential and places it in the location folder, and adds this folder to the path ready for use
############################################################################################################################################

