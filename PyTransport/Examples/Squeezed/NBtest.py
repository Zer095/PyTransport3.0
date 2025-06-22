####################################### PyTransPyStep simple example of basic functions ###########################################
from matplotlib import pyplot as plt
import time
import imp  
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
from PyTransport.PyTransPy import PyTransScripts as PyS
from PyTransport.PyTransPy import PyTransSetup

import PyTransMatt as PyT;  # import module

###########################################################################################################################################
nF = PyT.nF()
nP = PyT.nP()

fields = np.array([.075])
params = np.zeros(nP)
params[0]=4.*10**-12; params[1]=1.; params[2]=5*10**-7; params[3]=4*10**3;params[4] = 2.

V = PyT.V(fields,params)
dV = PyT.dV(fields,params)

initial = np.array([fields,-dV/np.sqrt(3.*V)])
############################################################################################################################################
file = open("NBout.txt", 'w')


############################################################################################################################################
Nstart = 0.0
Nend = 30.0
t=np.linspace(Nstart, Nend, 1000)

tols = np.array([10**-25,10**-25])
back = PyT.backEvolve(t, initial, params,tols, True)
print(back[-1,0])

fig1 = plt.figure(1)
plt.plot(back[:,0], back[:,1], 'r')
plt.plot(back[:,0], back[:,2], 'g')

###########################################################################################################################################

tols = np.array([10**-17,10**-17])
NB = 5.0
NExit = back[-1,0] - 10
k = PyS.kexitN(NExit, back, params, PyT)
print(f'k = {k}')
NstartS, backExitS = PyS.ICsBE(NB, k, back, params, PyT)
tsig = np.linspace(NstartS, back[-1,0], 1000)
fact = (k**3.)/(2.*np.pi**2.)

alpha=0.0
beta =1.0/3.0
k1 = (k/2.) - beta*(k/2.) ; k2 = (k/4.)*(1+alpha+beta) ; k3 = (k/4.)*(1-alpha+beta)
kM = np.min(np.array([k1,k2,k3]))
NstartA, backExitMinusA = PyS.ICsBM(NB, kM, back, params, PyT)
talp=np.linspace(NstartA, back[-1,0], 1000)

###########################################################################################################################################

rho = PyT.MPP2(tsig, k, backExitS, params, tols)
twoPt = PyT.MPPSigma(tsig, k, backExitS, params, rho, True, -1)
sigma = twoPt[:, 1 + 1 + 2*nF:]

rho3 = PyT.MPP3(talp, k1,k2,k3,backExitMinusA, params, tols)
threePtMPP = PyT.MPPAlpha(talp, k1, k2, k3, backExitMinusA, params, rho3, True)
alphaMPP = threePtMPP[:,1+4+2*nF+6*(2*nF*2*nF):]

############################################################################################################################################

nbs = np.linspace(1,10,19)
tol = np.linspace(5,17,25)

values = []
for i in range(len(nbs)):
    for j in range(len(tol)):
        values.append([nbs[i],tol[j]])

tsm = []
tst = []
pzm = []
pzt = []

tam = []
tat = []
fnlm = []
fnlt = []

for value in values:
    NB = value[0]
    exponent = -1.0*value[1]

    print(NB, exponent)
    tols = np.array([10**exponent, 10**exponent])

    NstartS, backExitS = PyS.ICsBE(NB, k, back, params, PyT)
    tsig = np.linspace(NstartS, back[-1,0], 1000)

    # MPP Two-point 
    start = time.time()
    rhoM = PyT.MPP2(tsig, k, backExitS, params, tols)
    twoPtM = PyT.MPPSigma(tsig, k, backExitS, params, rho, True, -1)
    sigmaM = twoPtM[:, 1 + 1 + 2*nF:]
    tsm.append(time.time() - start)
    pzm.append(fact*np.abs(twoPtM[-1,1]))

    # Trans Two-point
    start = time.time()
    twoPtT = PyT.sigEvolve(tsig, k, backExitS, params,tols,True)
    sigmaT = twoPtT[:,1+1+2*nF:]
    tst.append(time.time()-start)
    pzt.append(fact*np.abs(twoPtT[-1,1]))

    # MPP Three-point
    start = time.time()
    NstartA, backExitMinusA = PyS.ICsBM(NB, kM, back, params, PyT)
    talp=np.linspace(NstartA, back[-1,0], 1000)
    rho3 = PyT.MPP3(talp, k1,k2,k3,backExitMinusA, params, tols)
    threePtM = PyT.MPPAlpha(talp, k1, k2, k3, backExitMinusA, params, rho3, True)
    alphaM = threePtM[:,1+4+2*nF+6*(2*nF*2*nF):]
    tam.append(time.time()-start)
    zG = threePtMPP[-1,1:5]
    fnlG = ( 5.0/6.0*zG[3] )/( zG[1]*zG[2]  + zG[0]*zG[1] + zG[0]*zG[2] )
    fnlm.append(np.abs(fnlG))

    # Trans Three-point
    start = time.time()
    threePtT = PyT.alphaEvolve(talp, k1, k2, k3, backExitMinusA, params, tols, True)
    alphaT = threePtT[:,1+4+2*nF+6*(2*nF*2*nF):]
    tat.append(time.time()-start)
    zS = threePtT[:,1:5]
    fnlS = ( 5.0/6.0*zS[:,3] )/(zS[:,1]*zS[:,2]  + zS[:,0]*zS[:,1] + zS[:,0]*zS[:,2])
    fnlt.append(np.abs(threePtT[-1]))

    # Write on file
    string = f"{NB}, {exponent}, {tsm[-1]}, {pzm[-1]}, {tam[-1]}, {fnlm[-1]}\n"
    file.write(string)



############################################################################################################################################
nexit = r"$\Delta N$ =" + f"{back[-1,0]-NExit}"
phi_str = r"$\phi$"
pi_str = r"$\pi_{\phi}$"

labelTwo = [phi_str+"-"+phi_str, phi_str+'-'+pi_str, pi_str+"-"+phi_str, pi_str+"-"+pi_str]
colors = ["#b30000", "#7c1158", "#4421af", "#1a53ff", "#0d88e6", "#00b7c7", "#5ad45a", "#8be04e", "#ebdc78"]
clr = ["#b30000", "#4421af", "#00b7c7", "#5ad45a"]

deltaNTwo = back[-1,0] -  twoPt[:,0]

labels = [phi_str+"-"+phi_str+"-"+phi_str,
          phi_str+"-"+phi_str+'-'+pi_str, 
          phi_str+'-'+pi_str+"-"+phi_str,
          phi_str+'-'+pi_str+'-'+pi_str,
          pi_str+"-"+phi_str+"-"+phi_str,
          pi_str+"-"+phi_str+"-"+pi_str,
          pi_str+"-"+pi_str+"-"+phi_str,
          pi_str+"-"+pi_str+"-"+pi_str]

deltaNThree = back[-1,0] -  talp

############################################################################################################################################

fig2 = plt.figure(2)
for ii in range(2):
    for jj in range(2):
        plt.plot(deltaNTwo, np.abs(sigma[:, 2*nF*ii + jj]),color=clr[2*nF*ii+jj], label=labelTwo[2*nF*ii+jj])
plt.title("MPP Two-point correlation function "+nexit )
plt.gca().invert_xaxis()
plt.xlabel(r"$\Delta N$")
plt.ylabel(r'$\Sigma$', rotation=0)
plt.grid()
plt.yscale('log')
plt.legend()

fig3 = plt.figure(3)
for ii in range(2*nF):
    for jj in range(2*nF):
        for kk in range(2*nF):
            plt.plot(deltaNThree, np.abs(alphaMPP[:,2*nF*2*nF*ii + 2*nF*jj + kk]), color=colors[2*nF*2*nF*ii+2*nF*jj+kk], label=labels[2*nF*2*nF*ii+2*nF*jj+kk])
plt.title("MPP Three-point correlation function "+nexit )
plt.gca().invert_xaxis()
plt.xlabel(r"$\Delta N$")
plt.ylabel(r'$\alpha$', rotation=0)
plt.grid()
plt.yscale('log')
plt.legend()

############################################################################################################################################
file.close()
plt.show()