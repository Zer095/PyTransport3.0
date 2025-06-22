import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import itertools 


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

fig = plt.figure(1, figsize=(10,8))
plt.title(r'$\phi plot$')
plt.plot(back[:,0], back[:, 1])
plt.plot(back[:,0], back[:,3])
plt.savefig('Plots/DQ_Background_phi.pdf', format='pdf',bbox_inches='tight')

fig = plt.figure(2, figsize=(10,8))
plt.title(r'$\chi plot$')
plt.plot(back[:,0], back[:, 2])
plt.plot(back[:,0], back[:, 4])
plt.savefig('Plots/DQ_Background_chi.pdf', format='pdf',bbox_inches='tight')

plt.show()