import numpy as np
import matplotlib.pyplot as plt
import csv
import time

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
Nend = 75.0
t = np.linspace(Nstart, Nend, 1000)
tols = np.array([10**-25,10**-25])
back = PyT.backEvolve(t, initial, pvalue, tols, True)

############################################## Set 2pt and 3pt parameters ##################################################################
tols = np.array([10**-10,10**-10])
NB = 6.0
Nexit = back[-1, 0] - 50.0
k = PyS.kexitN(Nexit, back, pvalue, PyT)
print(f'k = {k}')
scaling = (k**3)/(2*np.pi**2)

Nstart, backExit = PyS.ICsBE(NB, k, back, pvalue, PyT)
tsig = np.linspace(Nstart, back[-1, 0], 1000)

##############################################  2pt and 3pt runs ##################################################################

print('Start 2pt PyT run')
start = time.time()
PyT_twoPt = PyT.sigEvolve(tsig, k, backExit, pvalue, tols, True)
print(f'End 2pt PyT run, Pz = {scaling*PyT_twoPt[-1, 1]}, time = {time.time()-start}')
print('---------------------------------------------------------')

print('Start 2pt MPP run')
start = time.time()
rho = PyT.MPP2(tsig, k, backExit, pvalue, tols)
MPP_twoPt = PyT.MPPSigma(tsig, k, backExit, pvalue, rho, True)
print(f'End 2pt MPP run, Pz = {scaling*MPP_twoPt[-1, 1]}, time = {time.time()-start}')
print('---------------------------------------------------------')

print('Start 3pt PyT run')
start = time.time()
PyT_3pt = PyT.alphaEvolve(tsig, k, k, k, backExit, pvalue, tols, True)
PyT_fnl = 5.6*PyT_3pt[-1, 4]/(PyT_3pt[-1,1]*PyT_3pt[-1, 2] + PyT_3pt[-1,1]*PyT_3pt[-1, 3] + PyT_3pt[-1,3]*PyT_3pt[-1, 2])
print(f'End 3pt PyT run, fnl = {PyT_fnl}, time = {time.time()-start}')
print('---------------------------------------------------------')

print('Start 3pt MPP run')
start = time.time()
rho3 = PyT.MPP3(tsig, k, k, k, backExit, pvalue, tols)
MPP_3pt = PyT.MPPAlpha(tsig, k, k, k, backExit, pvalue, rho3, True)
MPP_fnl = 5.6*MPP_3pt[-1, 4]/(MPP_3pt[-1,1]*MPP_3pt[-1, 2] + MPP_3pt[-1,1]*MPP_3pt[-1, 3] + MPP_3pt[-1,3]*MPP_3pt[-1, 2])
print(f'End 3pt MPP run, fnl = {MPP_fnl}, time = {time.time()-start}')
print('---------------------------------------------------------')

##############################################  Set plot parameters ##################################################################

# Useful strings for titles and labels
nexit = r"$\Delta N$ = " + f"{int(back[-1,0] - Nexit)}"
phi_str = r"$\phi$"
pi_str = r"$\pi_{\phi}$"

# Labels for 2pt and 3pt correlation functions
labels2 = [r"$\langle$" + phi_str+phi_str + r"$\rangle$", 
           r"$\langle$" + phi_str+pi_str+ r"$\rangle$",
           r"$\langle$" +pi_str+phi_str+ r"$\rangle$", 
           r"$\langle$" +pi_str+pi_str+ r"$\rangle$"]

labels3 = [r"$\langle$"+phi_str+phi_str+phi_str+r"$\rangle$",
          r"$\langle$"+phi_str+phi_str+pi_str+r"$\rangle$", 
          r"$\langle$"+phi_str+pi_str+phi_str+r"$\rangle$",
          r"$\langle$"+phi_str+pi_str+pi_str+r"$\rangle$",
          r"$\langle$"+pi_str+phi_str+phi_str+r"$\rangle$",
          r"$\langle$"+pi_str+phi_str+pi_str+r"$\rangle$",
          r"$\langle$"+pi_str+pi_str+phi_str+r"$\rangle$",
          r"$\langle$"+pi_str+pi_str+pi_str+r"$\rangle$"]

# Colors
clr = ["#B30000", "#1A53FF", "#5AD45A", "#ED7F2C"]

# Fontsizes
titsz = 24
legsz = 20
labsz = 20
ticksz = 18

##############################################  Plots ##################################################################

# Plot 2pt
PyT_sigma = PyT_twoPt[:, 2 + 2*nF:]
MPP_sigma = MPP_twoPt[:, 2+2*nF:]
ind = [0,1,3]
fig1, ax = plt.subplots(figsize=(10,8))
lines = []
for i in range(3):
    l1 = plt.plot(tsig, np.abs(PyT_sigma[:, ind[i]]), color = clr[i], label = labels2[i])
    l2 = plt.plot(tsig, np.abs(MPP_sigma[:, ind[i]]), color = 'black', label = labels2[i], linestyle='dashed')
    lines.append([l1[0],l2[0]])

#print(f'Handles = {[l[0][0].get_label() for l in lines]}')
plt.xlabel(r"$N$", fontsize=labsz)
plt.ylabel(r'$\Sigma$', rotation=0, fontsize=labsz)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.grid()
plt.yscale('log')
leg1 = ax.legend(lines[0], ['PyT', 'MPP'], loc='upper right', fontsize=legsz)
ax.add_artist(leg1)
ax.legend(handles=[l[0] for l in lines], loc='lower right', fontsize=legsz)
plt.savefig('Plots/SF_Confront2pt.pdf', format='pdf',bbox_inches='tight')


# Plot 3pt
fig2 = plt.figure(2, figsize=(10,8))
PyT_Alpha = PyT_3pt[:, 1+4+2*nF+6*(2*nF*2*nF):]
MPP_Alpha = MPP_3pt[:, 1+4+2*nF+6*(2*nF*2*nF):]
ind = [0, 1, 6, 7]
for i in range(4):
    plt.plot(tsig, np.abs(MPP_Alpha[:,ind[i]]), label = 'Mpp '+ labels3[ind[i]], color=clr[i], linestyle = 'dashed')
    plt.plot(tsig, np.abs(PyT_Alpha[:,ind[i]]), label = 'PyT '+ labels3[ind[i]], color=clr[i])
plt.xlabel(r"$ N$", fontsize=labsz)
plt.ylabel(r'$\alpha$', rotation=0, fontsize=labsz)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.grid()
plt.yscale('log')
plt.legend(fontsize=legsz)
plt.savefig('Plots/QA_Confront3pt.pdf', format='pdf',bbox_inches='tight')
##############################################  Show ##################################################################
plt.show()
