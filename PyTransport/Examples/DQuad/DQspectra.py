import numpy as np
import matplotlib.pyplot as plt
import csv


from PyTransport.PyTransPy import PyTransSetup
from PyTransport.PyTransPy import PyTransScripts as PyS

import PyTransDQ as PyT

############################################## Set initial values ##################################################################

fields = np.array([12.0, 12.0]) # we set up a numpy array which contains the values of the fields

nP=PyT.nP()   # the .np function gets the number of parameters needed for the potential -- this can be used as a useful crosscheck
pvalue = np.zeros(nP) 
pvalue[0]=1.702050*10.0**(-6.0); pvalue[1]=9.0*pvalue[0] # we set up numpy array which contains values of the parameters

nF=PyT.nF() # use number of fields routine to get number of fields (can be used as a useful cross check)

V = PyT.V(fields,pvalue) # calculate potential from some initial conditions
dV=PyT.dV(fields,pvalue) # calculate derivatives of potential

initial = np.concatenate((fields, -dV/np.sqrt(3* V))) # sets an array containing field values and there derivative in cosmic time 
                                                      # (set using the slow roll equation)

############################################## Background run ##################################################################
Nstart = 0.0
Nend = 73.5
t = np.linspace(Nstart, Nend, 1000)
tols = np.array([10**-18,10**-18])
back = PyT.backEvolve(t, initial, pvalue, tols, True)

############################################## Set PyT parameters ##################################################################
NB = 6.0
tols_PyT = np.array([10**-12,10**-12])
tols_MPP = np.array([10**-13,10**-13])

PyT_fnlOut = np.array([])
PyT_pzOut= np.array([])
MPP_fnlOut = np.array([])
MPP_pzOut= np.array([])
kOut = np.array([])
PhiOut = np.array([])
NOut = np.array([])

# Store NOut and kOut
for i in range(601):
    NExit = 30 + i*.05
    NOut = np.append(NOut, NExit)
    k = PyS.kexitN(NExit, back, pvalue, PyT)
    kOut = np.append(kOut, k)

# Compute zz and zzz in equilateral configuration with PyTransport
zzOut, zzzOut, times = PyS.eqSpectra(kOut, back, pvalue, NB, tols_PyT, PyT)

PyT_fnlOut = 5./6*zzzOut/(3.0*zzOut**2)
PyT_pzOut = (kOut**3/(2*np.pi**2))*zzOut

# Compute zz and zzz in equilateral configuration with MPP
zzOut, zzzOut, times = PyS.MPPSpectra(kOut, back, pvalue, NB, tols_MPP, PyT)

MPP_fnlOut = 5./6*zzzOut/(3.0*zzOut**2)
MPP_pzOut = (kOut**3/(2*np.pi**2))*zzOut

with open('Data/DQspectra.csv', 'a') as file:
    csvwriter = csv.writer(file)
    for i in range(601):
        row = [NOut[i], kOut[i], PyT_pzOut[i], MPP_pzOut[i], PyT_fnlOut[i], MPP_fnlOut[i]]
        csvwriter.writerow(row)