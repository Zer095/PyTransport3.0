from matplotlib import pyplot as plt
import time
from pylab import *
import numpy as np
from scipy import interpolate 
from scipy import optimize
from scipy.interpolate import UnivariateSpline
import sympy as sym
import subprocess
import os
from pathlib import Path
############################################################################################################################################

############################################################################################################################################

#This file contains a script meant to use on Apoctrita using the PyTransport package for the single field example of Chen et al.
#It assumes the MSetup.py file has been run to install the Matt model
#It is recommended you restart the kernel to insure any updates to PyTransMatt are imported 

from PyTransport.PyTransPy import PyTransSetup
PyTransSetup.pathSet()  # his add sets the other paths that PyTransport uses

import PyTransMatt as PyT;  # import module
from PyTransport.PyTransPy import PyTransScripts as PyS;
###########################################################################################################################################
inter = int(sys.argv[1]) - 1 + 4000
###########################################################################################################################################

nF = PyT.nF()
nP = PyT.nP()
pvalue = np.zeros(nP)

pvalue[0] = 4*10**-12
pvalue[1] = 1
pvalue[2] = 0.5*10**-6
pvalue[3] = 4*10**3
pvalue[4] = 2

fields = np.array([75*10**-3])

V = PyT.V(fields,pvalue)
dV = PyT.dV(fields, pvalue)

initial = np.concatenate((fields, -dV/np.sqrt(3*V)))
###########################################################################################################################################
# Background evolution
tols = np.array([10**-20, 10**-20])
NB = 4.0
Nstart = 0
Nend = 30.

t = np.linspace(Nstart,Nend,10000)

back = PyT.backEvolve(t, initial, pvalue, tols, False)
DeltaN = back[-1,0] - back[:,0]

###########################################################################################################################################
# calculate H(N)
H = []
for i in range(np.size(back[:,0])):

    Hv = PyT.H(back[i,1:3], pvalue)
    H.append(Hv)

H = np.array(H)
H_func = UnivariateSpline(back[:,0], H, s=0)

# calculate epsilon(N)
eps = 0.5*back[:,2]**2/H**2
eps_func = UnivariateSpline(back[:,0], eps, s=0)

# calculate eta(N)
deps = eps_func.derivative()
eta = deps([back[:,0]])/eps_func(back[:,0])
eta_func = UnivariateSpline(back[:,0], eta, s=0)

# calculate xi = eta'/eta
deta = eta_func.derivative()
xi = deta(back[:,0])/eta_func(back[:,0])

# calculate pi
pi = back[:,2]/H
pi_func = UnivariateSpline(back[:,0], pi, s=0)

# calculate phi
phi = back[:,1]
phi_func = UnivariateSpline(back[:,0], phi, s=0)

# Define when the transitions are happening

target = np.array([[18.19139567, 18.9934972],[20.37801688, 25.49594705]])

# Fields and velocity at the end of USR

tUSR = np.linspace(target[0,1], target[1,0])
Ne = target[1,0]
pie = pi_func(Ne)
phie = phi_func(Ne)

# Potential slow-roll parameters (x = phi)
def norm_pot(x):
    return pvalue[1] + pvalue[2]*(np.log(np.cosh(pvalue[3]*x)) + (pvalue[3] + pvalue[4])*x)

def epsilon_V(x):

    p0 = pvalue[1]

    p1 = pvalue[2]

    p2 = pvalue[3]

    p3 = pvalue[4]

    return (p1**2 * (p2 + p3 +p2 * np.tanh(p2 * x))**2) / (2*(p0 + p1*((p2+p3)*x + np.log(np.cosh(p2*x))))**2)

def eta_V(x):

    p0 = pvalue[1]

    p1 = pvalue[2]

    p2 = pvalue[3]

    p3 = pvalue[4]

    return (p1*p2**2 / np.cosh(p2*x)**2) / (p0+p1*((p2+p3)*x + np.log(np.cosh(p2*x))))

# Define h and s as in Cai et al. Here x is the time
def h(x):

    return 6*np.sqrt(2*epsilon_V(phi_func(x)))/pi_func(x)

def s(x):

    return np.sqrt(9-12*eta_V(phi_func(x)))


NCMBexit = 20.0


def kexitNH(NexitArr, Harr, back, params, PyT, exact=False):

    logk = interpolate.splev(NexitArr, interpolate.splrep(back[:,0], back[:,0] + np.log(Harr), s=1e-15), der=0)

    return np.exp(logk)

###########################################################################################################################################
tols = np.array([10**-12, 10**-12])

Numk = 5000 # Number of k's used to 

Ncross = np.linspace(back[-1,0] - NCMBexit, back[-1,0], Numk)

kvalues = kexitNH(Ncross, H, back, pvalue, PyT)

kpeak = 5.018093148060832505e+02

# kL = 0.001* kpeak
kL= 0.0004 * kpeak 

# print(np.log10(kL/kpeak))
# print('------------------------')
# # print(np.log10(117.15245880038552))
# print(np.log10(1.153958169334419495e+02))
# print('------------------------')

#factor = np.logspace(-3.398, 1, Numk)
#factor = np.logspace(np.log10(kL/kpeak), 4, Numk)
factor = np.logspace(np.log10(kL/kpeak), np.log10(1.153958169334419495e+02),  Numk)

print('------------------------')
print(f"factor[0] = {factor[0]}, factor[-1] = {factor[-1]}")

kS = kpeak * factor[int(inter)]

k1 = kL

k2 = kS

k3 = kS

print(f'Factor = {factor[inter]}, Ratio = {kS/kL}')
print(f'k1 = {k1}, k2 = {k2}, k3 = {k3}')
print('------------------------')

kmin = np.min(np.array([k1,k2,k3]))
Nstart, backExitMinus = PyS.ICsBE(NB, kmin, back, pvalue, PyT)
talp = np.linspace(Nstart, back[-1,0], num=1000)

NstartS, backExitMinusS = PyS.ICsBE(NB, kS, back, pvalue, PyT)
tsig = np.linspace(NstartS, back[-1,0], num=1000)
fac = (kS**3)/(2*np.pi**2)

print('Start PyT 2pt Run')
pzT = []
try:
    twoPtT = PyT.sigEvolve(tsig, kS, backExitMinusS, pvalue, tols, True)
    pzT = fac*twoPtT[:,1]
except TypeError:
    pzT.append(0.)

print(f'PyT Pz = {pzT[-1]}')

print('------------------------')

print('Start MPP 2pt Run')
pzM = []
try:
    rho = PyT.MPP2(tsig, kS, backExitMinusS, pvalue, tols)
    twoPtM = PyT.MPPSigma(tsig, kS, backExitMinusS, pvalue, rho, True, -1)
    pzM = fac*twoPtM[:,1]
except TypeError:
    pzM.append(0.)

print(f'MPP Pz = {pzM[-1]}')

print('------------------------')

print('Start PyT 3pt Run')
fNLT = []
try:
    BzzzT = PyT.alphaEvolve(talp, k1, k2, k3, backExitMinus, pvalue, tols, False)
    fNLT = 5/6 * BzzzT[:,4]/(BzzzT[:,1]*BzzzT[:,2] + BzzzT[:,1]*BzzzT[:,3] + BzzzT[:,2]*BzzzT[:,3])
except TypeError:
    fNLT.append(0.)

print(f'PyT fNL = {fNLT[-1]}')

print('------------------------')

print('Start MPP 3pt Run')

FNLM = []
try:
    rho3 = PyT.MPP3(talp, k1, k2, k3, backExitMinus, pvalue, tols)
    BzzzM = PyT.MPPAlpha(talp, k1, k2, k3, backExitMinus, pvalue, rho3, True)
    fNLM = 5/6 * BzzzM[:,4]/(BzzzM[:,1]*BzzzM[:,2] + BzzzM[:,1]*BzzzM[:,3] + BzzzM[:,2]*BzzzM[:,3])
except TypeError:
    FNLM.append(0.)

print(f'MPP fNL = {fNLM[-1]}')

print(f'Saving file on { "zzz"+str(inter)+".out" }')

np.savetxt('zzz'+str(inter)+'.out', (factor[int(inter)], kS, pzT[-1], pzM[-1], fNLT[-1], fNLM[-1]), newline='\n')

print(f'End run inter = {inter}')