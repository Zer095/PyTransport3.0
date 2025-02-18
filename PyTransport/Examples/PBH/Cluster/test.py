####################################### PyTransPyStep simple example of basic functions ###########################################
from matplotlib import pyplot as plt
import time
from pylab import *
import numpy as np
from scipy import interpolate 
import sympy as sym
import subprocess
import os
from pathlib import Path
############################################################################################################################################

############################################################################################################################################

#This file contains simple examples of using the PyTransport package for the single field example of Chen et al.
#It assumes the StepExampleSeptup file has been run to install a Step version of PyTransPyStep
#It is recommended you restart the kernel to insure any updates to PyTransPyStep are imported 
location = os.path.dirname(__file__) # this should be the current location of the file
location = Path(location)
location = location.parent.absolute().parent.absolute().parent.absolute().parent.absolute()
location = os.path.join(location, 'PyTransport')
sys.path.append(location)  # we add this location to the python path

import PyTransSetup
PyTransSetup.pathSet()  # his add sets the other paths that PyTransport uses

import PyTransStepPBH as PyT;  # import module
import PyTransScripts as PyS;
###########################################################################################################################################
inter = int(sys.argv[1]) - 1             # Argument passed by the cluster - it's the job index
###########################################################################################################################################
tols = np.array([10**-17,10**-17])      # Tolerance required for the numerical integrator

nF = PyT.nF()                           # Get the number of fields
nP = PyT.nP()                           # Get the number of parameters

fields = np.array([.0008])              # Initial field value
params = np.zeros(nP)                   # Set up the parameters
params[0]=1.0; params[1]=0.003; params[2]=0.005; params[3]=0.0005; params[4] = .06; params[5] = 0.001

V = PyT.V(fields,params)                # Initial value of the potential
dV = PyT.dV(fields,params)              # Intial value of the derivative of the potential

initial = np.array([fields,-dV/np.sqrt(3.*V)])  # Initial condition with slow-roll conditions

################################## Calculate the Background evolution ########################################
Nstart = 0.0                                # Initial time 
Nend = 50.0                                 # End time
t = np.linspace(Nstart, Nend, 1000)         # Time steps
back = PyT.backEvolve(t, initial, params, tols, True)       # Background evolution
print(f'Nend = {back[-1,0]}')

################################## Set up parameters for the spectrums  ########################################
NB = 5.0
Nexits = np.linspace(6, 46, 1001)                # Array with NExits for each mode k

kouts = []
for Ne in Nexits:
    k = PyS.kexitN(Ne, back, params, PyT)
    kouts.append(k)

kouts = np.array(kouts)                       # K mode for each NExits


k = kouts[inter]
Nstart, backExit = PyS.ICsBE(NB, k, back, params, PyT)
times = np.linspace(Nstart, back[-1,0], 1000)

k1 = k; k2 = k; k3 = k

#################################################################################################################
print('MPP Two-point run')
# MPP Two-point run
rho = PyT.MPP2(times, k, backExit, params, tols)
twoPtM = PyT.MPPSigma(times, k, backExit, params, rho, True, -1)
pzM = twoPtM[-1,1]
print('PyT Two-point run')
# PyT Two-point run
twoPtT = PyT.sigEvolve(times, k, backExit, params,tols,True)
pzT = twoPtT[-1,1]
print('MPP Three-point run')
# MPP Three-point run
rho3 = PyT.MPP3(times, k1,k2,k3,backExit, params, tols)
threePtM = PyT.MPPAlpha(times, k1, k2, k3, backExit, params, rho3, True)
zM = threePtM[:,1:5]
fnlM = np.abs( ( 5.0/6.0*zM[-1,3] )/(zM[-1,1]*zM[-1,2]  + zM[-1,0]*zM[-1,1] + zM[-1,0]*zM[-1,2]) )
print('PyT Three-point run')
# PyT Three-point run
threePtT = PyT.alphaEvolve(times, k1, k2, k3, backExit, params, tols, True)
zT = threePtT[:,1:5]
fnlT = np.abs( ( 5.0/6.0*zT[-1,3] )/(zT[-1,1]*zT[-1,2]  + zT[-1,0]*zT[-1,1] + zT[-1,0]*zT[-1,2]) )

# Save file
np.savetxt('zzz'+str(inter)+'.txt',(Nexits[inter], k, pzM, pzT, fnlM, fnlT), newline='\n')