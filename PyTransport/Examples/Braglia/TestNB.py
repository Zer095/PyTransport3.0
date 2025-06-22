################################################### Import ###################################################
from matplotlib import pyplot as plt     # Standard package for plotting
import time                              # Package to track the running time of our algorithm
import numpy as np                       # Package needed to define arrays
import sys
###############################################################################################################

############################################## Import PyTransport ##############################################
from PyTransport.PyTransPy import PyTransScripts as PyS     # Import the PyTransScript module

import PyTransLNC as PyT                       # Import the model. Be sure the name is the same that appears in .compileName3 in xSetup.py
###############################################################################################################
inter = int(sys.argv[1]) - 1
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

#####################################################################################################

NExit = back[-1,0] - 50.0
k = PyS.kexitN(NExit, back, params, PyT)
fact = (k**3.)/(2.*np.pi**2.)

##################################################################################################################################
tolerances = np.linspace(5,18,27)
nbs = np.linspace(1, 14, 27)
values = []
for nb in nbs:
    for tol in tolerances:
        values.append([nb,tol])

print(f"len = {len(values)}")

tsm = []    # Time-sigma MPP
tst = []    # Time-sigma Transport 
pzm = []    # Final Pz MPP
pzt = []    # Final Pz Transport
tam = []    # Time-alpha MPP
tat = []    # Time-alpha Transport
golsm = 0   # Golden-tolerance sigma MPP
golst = 0   # Golden-tolerance sigma Transport
fnlm = []   # Final Fnl MPP
fnlt = []   # Final Fnl Transport

NB = values[inter][0]
t = values[inter][1]

exponent = -1.0*t
tols = np.array([10**exponent, 10**exponent])
Nstart, backExitMinus = PyS.ICs(NB, k, back, params, PyT) 
talp=np.linspace(Nstart, back[-1,0], 1000)


start = time.time()
rho = PyT.MPP2(talp, k, backExitMinus, params, tols)
twoPtMPP = PyT.MPPSigma(talp, k, backExitMinus, params, rho, True, - 1)
tsm = time.time()- start

start = time.time()
twoPtTrans = PyT.sigEvolve(talp, k, backExitMinus, params, tols, True)
tst = time.time()-start

start = time.time()
rho3 = PyT.MPP3(talp, k, k, k,backExitMinus, params, tols)
threePtMPP = PyT.MPPAlpha(talp, k, k, k, backExitMinus, params, rho3, True)
tam = time.time() - start

start = time.time()
threePtPyT = PyT.alphaEvolve(talp, k, k, k, backExitMinus, params, tols, True)
tat = time.time() - start

pzm = fact*threePtMPP[-1, 1]
pzt = fact*threePtPyT[-1, 1]

fnlm = (5./6 * threePtMPP[-1, 4])/(3*threePtMPP[-1, 1]**2)
fnlt = (5./6 * threePtPyT[-1, 4])/(3*threePtPyT[-1, 1]**2)

np.savetxt('zzz'+str(inter)+'.out', (NB, exponent, tsm, pzm, tst, pzt, tam, fnlm, tat, fnlt), newline='\n')