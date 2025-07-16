
import matplotlib.pyplot as plt
import csv
import numpy as np

from PyTransport.PyTransPy import PyTransSetup
from PyTransport.PyTransPy import PyTransScripts as PyS

import PyTransLNC as PyT

##################################### Set initial values ######################################################

nF = PyT.nF()
nP = PyT.nP()


fields = np.array([7., 7.31])
pvalue = np.zeros(nP)

pvalue[0] = 1.0
pvalue[1] = np.sqrt(6.); pvalue[2] = 1/500.0; pvalue[3] = 7.8

V = PyT.V(fields, pvalue)
dV = PyT.dV(fields, pvalue)
initial = np.concatenate((fields, np.array([0,0])))

################################# Run Background ####################################################

Nstart = 0.
Nend = 85.6
t = np.linspace(Nstart, Nend, 10**4)
tols = np.array([10e-10,10e-10])
back = PyT.backEvolve(t, initial, pvalue, tols, True)

############################################## Set PyT parameters ##################################################################
NB = 4.0
tols = np.array([10**-10,10**-10])

PyT_pzOut= np.array([])

kOut = np.array([])
PhiOut = np.array([])
NOut = np.array([])

for i in range(801):
    NExit = 5 + i*0.1
    NOut = np.append(NOut, NExit)
    k = PyS.kexitN(NExit, back, pvalue, PyT)
    kOut = np.append(kOut, k)

zzOut, times = PyS.pSpectraMPP(kOut, back, pvalue, NB, tols, PyT)

PyT_pzOut = (kOut**3/(2*np.pi**2))*zzOut

# Plot parameters
sz_tit = 24
sz_leg = 20
sz_lab = 18

# Plot Pz vs k
fig1 = plt.figure(1, figsize=(10,8))
plt.plot(kOut, np.abs(PyT_pzOut), label='PyT')
# plt.vlines(x=k_cmb/kOut[0], ymax=max(MPP_Pz_norm), ymin=min(MPP_Pz_norm), color='black',linestyle='dashed', label=r'$k_{CMB}$')
#plt.title(r'$P_{\zeta}$ spectrum, PyT vs MPP', fontsize = sz_tit)
plt.grid()
plt.legend(fontsize=sz_leg)
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$k/k_0$', fontsize = sz_lab)
plt.ylabel(r'$P_{\zeta}(k)$/$P_{\zeta}(k_0)$', fontsize = sz_lab)
plt.savefig('plots/LNC_spectra_pz_dibya.pdf', format='pdf',bbox_inches='tight')

plt.show()