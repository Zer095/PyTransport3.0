import sympy as sym
import math
import numpy as np
from gravipy import *
from pathlib import Path

#################################################################
from PyTransport.PyTransPy import PyTransSetup, PyTransScripts

PyTransSetup.pathSet()
#############################################################

nF = 2
nP = 11
f = sym.symarray('f', nF) 
p = sym.symarray('p', nP)

h1=p[1] * ((9 * p[6] ** 2) / (f[0] ** 2) - sym.log(1 + (9. * p[6] ** 2) / (f[0] ** 2)))

V = (4 * math.pi * p[8] * pow(h1,-1) *
    (p[0] * sym.sqrt((h1/ 9.) *
                      (f[0] ** 2 + p[6] ** 2) ** 2 + (math.pi * p[5] ** 2 * p[2]) ** 2) - p[9] * p[0] * math.pi * p[5] ** 2 * p[2]) + 4.*math.pi**2* p[5]**2*p[0]*p[2]*p[8]*p[9]* (5.625* (f[0]**2/p[6]**2-2.0)*((f[0]**2)/(9*p[6]**2)) -1.4905 + 11.25 * sym.log(1+ (f[0]**2)/(9*p[6]**2)) + 2 * ((9 * p[3] * p[6] ** 2) / (f[0] ** 2) - p[3]  * sym.log(1 + (9 * p[6] ** 2) / (f[0] ** 2)))
+ p[4] * sym.cos(f[1]) * (2 * (6. + ((3 * p[6]) ** 2) / (f[0] ** 2) - 2 * (2. + 3. * (f[0] ** 2) / ((3 * p[6]) ** 2)) * sym.log(1. + ((3 * p[6]) ** 2) / (f[0] ** 2))))   +(1./2.)*p[7] * (2. + (f[0] ** 2) / (3* p[6] ** 2)) * sym.cos(f[1])
 ))-p[10] 

G = sym.Matrix([
    [4 * sym.pi * p[8] * p[0] * ((f[0] ** 2 + 6 * p[6] ** 2) / (f[0] ** 2 + 9 * p[6] ** 2)) *
     sym.sqrt((p[1] * ((9 * p[6] ** 2) / (f[0] ** 2) - sym.log(1 + (9. * p[6] ** 2) / (f[0] ** 2)))) / 9.0 *
              (f[0] ** 2 + p[6] ** 2) ** 2 + (sym.pi * p[5] ** 2 * p[2]) ** 2), 0],
    [0, 4 * sym.pi * p[8] * p[0] * ((f[0] ** 2 + 6 * p[6] ** 2) / 6.0) *
     (sym.sqrt((p[1] * ((9 * p[6] ** 2) / (f[0] ** 2) - sym.log(1 + (9. * p[6] ** 2) / (f[0] ** 2)))) / 9.0) *
      (f[0] ** 2 + p[6] ** 2) ** 2 + (sym.pi * p[5] ** 2 * p[2]) ** 2)]
])

PyTransSetup.potential(V,nF,nP,False,G)
PyTransSetup.compileName("Dibya", True)