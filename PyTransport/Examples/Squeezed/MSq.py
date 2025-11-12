######### Testing script for Axion-Quartic model ####################################
# The purpose of this script is to compare the results obtained with the transport equation approach 
# to the Gamma Matrices formalism, with different NB

###############################################################################################################################
########################## Set up ##################################

from matplotlib import pyplot as plt
from pylab import *  # contains some useful stuff for plotting
import math 
import numpy as np
import sys 
import csv
import os
from pathlib import Path
############################################################################################################################################

#This file contains simple examples of using the PyTransport package for the heavy field example of Langlois.
#It assumes the PyTransAxQrt file has been run to install a AxQrt version of PyTransPy
#It is recommended you restart the kernel to insure any updates to PyTransAxQrt are imported 

from PyTransport.PyTransPy import PyTransSetup
PyTransSetup.pathSet()  # his add sets the other paths that PyTransport uses

import PyTransMatt as PyT;  # import module
from PyTransport.PyTransPy import PyTransScripts as PyS;

########################## Set initial values ##################################

nF=PyT.nF() # gets number of fields (useful check)
nP=PyT.nP() # gets number of parameters needed (useful check)

fields = np.array([0.075])

pvalue = np.zeros(nP)
pvalue[0] = 4*10**-12; pvalue[1] = 1.; pvalue[2] = 5*10**-7; pvalue[3] = 4*10**-3; pvalue[4] = 2.

V = PyT.V(fields,pvalue) # calculate potential from some initial conditions
dV=PyT.dV(fields,pvalue) # calculate derivatives of potential (changes dV to derivatives)
print(f"dV = {dV}")
initial = np.concatenate((fields,-dV/np.sqrt(3* V))) # set initial conditions using slow roll expression
############################################################################################################################################
################################## run the background fiducial run #########################################################################
Nstart = 0.0
Nend = 30.0
t=np.linspace(Nstart, Nend, 1000) # array at which output is returned
tols = np.array([10**-25,10**-25])
back = PyT.backEvolve(t, initial, pvalue,tols,True) # The output is read into the back numpy array 
print(f"Nend = {back[-1,0]}")
############################################################################################################################################

NExit = back[-1,0] - 20.0
print(f"Nexit = {NExit}")
k = PyS.kexitN(NExit, back, pvalue, PyT)
tols = np.array([10**-17,10**-17])

NB = 6.0
alpha = 0.0
beta_max = 1 - 2*10E-8
beta_min = 1./3.

betas = list(np.linspace(beta_min, beta_max, 400))
ratios = []
#ratios = [1/3, 0.1, 0.05, 0.01, 0.005, 0.001]
# Times
tam = []
tat = []
# fNL
fnlm = []
fnlt = []
for beta in betas:
    start_loop = time.time()
    # betas.append(beta)
    k1 = (k/2.)*(1 - beta); k2 = (k/4.)*(1+alpha+beta) ; k3 = (k/4.)*(1-alpha+beta)
    kM = np.min(np.array([k1,k2,k3]))
    kMax = np.max(np.array([k1,k2,k3]))
    print(f"Run number: {betas.index(beta)}")
    # print(f"r = {r}")
    print(f"beta = {beta}")
    print(f"k = {k}")
    print(f"k's = {k1,k2,k3}")
    print(f"ratio = {kM/kMax}")
    ratios.append(kM/k)
    NstartA, backExitMinusA = PyS.ICsBM(NB, kM, back, pvalue, PyT)
    talp=np.linspace(NstartA,back[-1,0], 1000)

    # Three-point run MPP
    start = time.time()
    rho3 = PyT.MPP3(talp, k1,k2,k3,backExitMinusA, pvalue, tols)
    threePtMPP = PyT.MPPAlpha(talp, k1, k2, k3, backExitMinusA, pvalue, rho3, True, 999)
    tam.append(time.time()-start)
    zG = threePtMPP[1:5]
    fnlG = 5.0/6.0*zG[3]/(zG[1]*zG[2]  + zG[0]*zG[1] + zG[0]*zG[2])
    fnlm.append(abs(fnlG))
    print(f'MPP fNL = {np.abs(fnlG)}')
    # Three-point run Transport
    start = time.time()
    threePtTrans = PyT.alphaEvolve(talp,k1,k2,k3,backExitMinusA,pvalue,tols,True)
    tat.append(time.time() - start)
    zS = threePtTrans[:,1:5]
    fnlS = 5.0/6.0*zS[:,3]/(zS[:,1]*zS[:,2]  + zS[:,0]*zS[:,1] + zS[:,0]*zS[:,2])
    fnlt.append(abs(fnlS[-1]))
    print(f'Trans fNL = {np.abs(fnlS[-1])}')

    print(f"Total time = {time.time() - start_loop}")
    print("-------------------------------------------------------------------")



# Write on files
with open('files/testSq.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(betas)):
        row = [betas[i], tam[i], tat[i], fnlm[i], fnlt[i]]
        writer.writerow(row)