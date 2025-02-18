import numpy as np
import matplotlib.pyplot as plt
import csv


from PyTransport.PyTransPy import PyTransSetup
from PyTransport.PyTransPy import PyTransScripts as PyS

import PyTransAxQ as PyT

############################################## Set initial values ##################################################################

nF=PyT.nF() # gets number of fields (useful check)
nP=PyT.nP() # gets number of parameters needed (useful check)

fields = np.array([23.5,.5-0.001])

pvalue = np.zeros(nP)
pvalue[0]=2.690422*10**-15; pvalue[1]=1.; pvalue[2]=25.0**2.0*pvalue[0]/4.0/np.pi**2

V = PyT.V(fields,pvalue) # calculate potential from some initial conditions
dV=PyT.dV(fields,pvalue) # calculate derivatives of potential (changes dV to derivatives)
initial = np.concatenate((fields,np.array([0.,0.]))) # set initial conditions using slow roll expression

############################################## Background run ##################################################################
Nstart = 0.0
Nend = 70.0
t = np.linspace(Nstart, Nend, 1000)
tols = np.array([10**-25,10**-25])
back = PyT.backEvolve(t, initial, pvalue, tols, True)

############################################## Set PyT parameters ##################################################################
NB = 6.0
tols = np.array([10**-12,10**-12])

PyT_fnlOut = np.array([])
PyT_pzOut= np.array([])
MPP_fnlOut = np.array([])
MPP_pzOut= np.array([])
kOut = np.array([])
PhiOut = np.array([])
NOut = np.array([])

for i in range(600):
    NExit = 30 + i*.05
    NOut = np.append(NOut, NExit)
    k = PyS.kexitN(NExit, back, pvalue, PyT)
    kOut = np.append(kOut, k)

zzOut, zzzOut, times = PyS.eqSpectra(kOut, back, pvalue, NB, tols, PyT)

PyT_fnlOut = 5./6*zzzOut/(3.0*zzOut**2)
PyT_pzOut = (kOut**3/(2*np.pi**2))*zzOut

zzOut, zzzOut, times = PyS.MPPSpectra(kOut, back, pvalue, NB, tols, PyT)

MPP_fnlOut = 5./6*zzzOut/(3.0*zzOut**2)
MPP_pzOut = (kOut**3/(2*np.pi**2))*zzOut

with open('Data/QAspectra.csv', 'a') as file:
    csvwriter = csv.writer(file)
    for i in range(401):
        row = [NOut[i], kOut[i], PyT_pzOut[i], MPP_pzOut[i], PyT_fnlOut[i], MPP_fnlOut[i]]
        csvwriter.writerow(row)