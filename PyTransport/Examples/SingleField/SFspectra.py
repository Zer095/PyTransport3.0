import numpy as np
import matplotlib.pyplot as plt
import csv


from PyTransport.PyTransPy import PyTransSetup
from PyTransport.PyTransPy import PyTransScripts as PyS

import PyTransSF as PyT

############################################## Set initial values ##################################################################

nF=PyT.nF() # gets number of fields (useful check)
nP=PyT.nP() # gets number of parameters needed (useful check)


pvalue = np.zeros(nP)
pvalue[0]=8.02713424*10**-6; pvalue[1]=0.0018; pvalue[2]=14.84; pvalue[3]=0.022;

fields = np.array([17.0])

V = PyT.V(fields,pvalue) # calculate potential from some initial conditions
dV=PyT.dV(fields,pvalue) # calculate derivatives of potential (changes dV to derivatives)

initial = np.array([fields,-dV/np.sqrt(3.*V)]) # set initial conditions to be in slow roll

############################################## Background run ##################################################################
Nstart = 0.0
Nend = 73.5
t = np.linspace(Nstart, Nend, 1000)
tols = np.array([10**-25,10**-25])
back = PyT.backEvolve(t, initial, pvalue, tols, True)

############################################## Set PyT parameters ##################################################################
NB = 6.0
tols = np.array([10**-10,10**-10])

PyT_fnlOut = np.array([])
PyT_pzOut= np.array([])
MPP_fnlOut = np.array([])
MPP_pzOut= np.array([])
kOut = np.array([])
PhiOut = np.array([])
NOut = np.array([])

for i in range(401):
    NExit = 10 + i*.05
    NOut = np.append(NOut, NExit)
    k = PyS.kexitN(NExit, back, pvalue, PyT)
    kOut = np.append(kOut, k)

zzOut, zzzOut, times = PyS.eqSpectra(kOut, back, pvalue, NB, tols, PyT)

PyT_fnlOut = 5./6*zzzOut/(3.0*zzOut**2)
PyT_pzOut = (kOut**3/(2*np.pi**2))*zzOut

zzOut, zzzOut, times = PyS.MPPSpectra(kOut, back, pvalue, NB, tols, PyT)

MPP_fnlOut = 5./6*zzzOut/(3.0*zzOut**2)
MPP_pzOut = (kOut**3/(2*np.pi**2))*zzOut

with open('Data/SFspectra.csv', 'w') as file:
    csvwriter = csv.writer(file)
    for i in range(401):
        row = [NOut[i], kOut[i], PyT_pzOut[i], MPP_pzOut[i], PyT_fnlOut[i], MPP_fnlOut[i]]
        csvwriter.writerow(row)