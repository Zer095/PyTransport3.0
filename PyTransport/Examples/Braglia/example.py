####################################### Example of Standard PyTransport #######################################

'''
In this example, we'll show how to use PyTransport to compute inflationary observables.
We'll use the model defined in xSetup.py. Remember to setup the model by running the command: python xSetup.py

The fundamental steps are:
    - Import packages;
    - Import PyTransport;
    - Initialize model.
These steps are needed for every script.

Then we'll show:
    - How to run a background evolution, and how to plot it.
      This is needed for any further calculation.
    - How to run a phase-space 2pt evolution, 
      how to compute the curvature perturbation power spectrum,
      how to plot it. 
    - How to run a tensor 2pt evolution,
    - How to run a phase-space 3pt evolution,
      how to compute the reduced bispectrum f_NL and how to plot it.

In this example, we we'll use the Transport equation. 
For an example on how to use the MPP formalism, see MPPExample.py.
'''

################################################### Import ###################################################
from matplotlib import pyplot as plt     # Standard package for plotting
import time                              # Package to track the running time of our algorithm
import numpy as np                       # Package needed to define arrays
###############################################################################################################

############################################## Import PyTransport ##############################################
from PyTransport.PyTransPy import PyTransScripts as PyS     # Import the PyTransScript module

import PyTransLNC as PyT                       # Import the model. Be sure the name is the same that appears in .compileName3 in xSetup.py
###############################################################################################################
##################################### Set initial values ######################################################

nF = PyT.nF()
nP = PyT.nP()

fields = np.array([7., 7.31])
params = np.zeros(nP)

params[0] = 1.
params[1] = np.sqrt(6.); params[2] = params[0]/500.; params[3] = 7.8

V = PyT.V(fields, params)
dV = PyT.dV(fields, params)
initial = np.concatenate((fields, np.array([0.,0.])))

#####################################################################################################
################################# Run Background ####################################################

Nstart = 0.
Nend = 85.6
t = np.linspace(Nstart, Nend, 10**4)
tols = np.array([10e-25,10e-25])
back = PyT.backEvolve(t, initial, params, tols, True)
# Plot background evolution
fig1 = plt.figure(1)
plt.plot(back[:,0], back[:,1], 'r')
plt.plot(back[:,0], back[:,2], 'g')
print(back[-1,0])
######################################################################################################
################################# Set 2pt/3pt parameters #############################################

NExit = back[-1,0] - 50.
NB = 5.0
k = PyS.kexitN(NExit, back, params, PyT)
fact = (k**3)/(2*np.pi**2)
alpha = 0.
beta = 1/3

tols = np.array([1e-14,1e-14])
NstartA, backExitMinusA = PyS.ICs(NB, k, back, params, PyT)
talp=np.linspace(NstartA, back[-1,0], 1000)

print(f'Time_step 3pt = {talp[1] - talp[0]}')

######################################################################################################
############################################## 3pt run ###############################################

#threePtMPP = PyT.rhoEvolve2(talp, k, k , k, backExitMinusA, params, tols, True)

start = time.time()
rho3 = PyT.MPP3(talp, k, k, k,backExitMinusA, params, tols)
threePtMPP = PyT.MPPAlpha(talp, k, k, k, backExitMinusA, params, rho3, True)
print(f'MPP time = {time.time() - start}')
MPPalpha = threePtMPP[:,5:]

start = time.time()
threePtPyT = PyT.alphaEvolve(talp, k, k, k, backExitMinusA, params, tols, True)
PyTalpha = threePtPyT[:,5:]
print(f'PyT time = {time.time() - start}')

print(f'MPP Pz = {fact*threePtMPP[-1, 1]}')
print(f'PyT Pz = {fact*threePtPyT[-1, 1]}')

print(f'MPP FNL = {(5./6 * threePtMPP[-1, 4])/(3*threePtMPP[-1, 1]**2)}')
print(f'PyT FNL = {(5./6 * threePtPyT[-1, 4])/(3*threePtPyT[-1, 1]**2)}')

fig2 = plt.figure(2, figsize=(10,8))
for i in range(2):
    for j in range(2):
      plt.plot(threePtMPP[:,0], np.abs(MPPalpha[:,2*nF*i + j]), label='MPP')
      plt.plot(threePtPyT[:,0], np.abs(PyTalpha[:,2*nF*i + j]), linestyle='dashed', label = 'PyT')
plt.yscale('log')
plt.legend()
plt.grid()

######################################################################################################
plt.show()

