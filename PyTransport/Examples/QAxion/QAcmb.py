import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import itertools


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

############################################## Set 2pt and 3pt parameters ##################################################################
tols = np.array([10**-10,10**-17])
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
PyT_fnl = 5/6*PyT_3pt[-1, 4]/(PyT_3pt[-1,1]*PyT_3pt[-1, 2] + PyT_3pt[-1,1]*PyT_3pt[-1, 3] + PyT_3pt[-1,3]*PyT_3pt[-1, 2])
print(f'End 3pt PyT run, fnl = {PyT_fnl}, time = {time.time()-start}')
print('---------------------------------------------------------')

print('Start 3pt MPP run')
start = time.time()
rho3 = PyT.MPP3(tsig, k, k, k, backExit, pvalue, tols)
MPP_3pt = PyT.MPPAlpha(tsig, k, k, k, backExit, pvalue, rho3, True)
MPP_fnl = 5/6*MPP_3pt[-1, 4]/(MPP_3pt[-1,1]*MPP_3pt[-1, 2] + MPP_3pt[-1,1]*MPP_3pt[-1, 3] + MPP_3pt[-1,3]*MPP_3pt[-1, 2])
print(f'End 3pt MPP run, fnl = {MPP_fnl}, time = {time.time()-start}')
print('---------------------------------------------------------')

# Compute ns
Nstart1, backExit = PyS.ICsBE(NB, k+0.01, back, pvalue, PyT)
tsig1 = np.linspace(Nstart1, back[-1, 0], 1000)
rho1 = PyT.MPP2(tsig1, k+0.01, backExit, pvalue, tols)
MPP_twoPt1 = PyT.MPPSigma(tsig1, k+0.01, backExit, pvalue, rho, True)


zz1a = MPP_twoPt[-1,1]
zz2a = MPP_twoPt1[-1,1]
n_s = (np.log(zz2a)-np.log(zz1a))/(np.log(k+.01)-np.log(k))
print(f'nS = {np.abs(n_s)}')
##############################################  Set plot parameters ##################################################################

# Useful strings for titles and labels
nexit = r"$\Delta N$ = " + f"{int(back[-1,0] - Nexit)}"
phi_str = r"$\phi$"
pi_str = r"$\pi_{\phi}$"

# # Labels for 2pt and 3pt correlation functions
# labels2 = [r"$\langle$" + phi_str+phi_str + r"$\rangle$", 
#            r"$\langle$" + phi_str+pi_str+ r"$\rangle$",
#            r"$\langle$" +pi_str+phi_str+ r"$\rangle$", 
#            r"$\langle$" +pi_str+pi_str+ r"$\rangle$"]

# labels3 = [r"$\langle$"+phi_str+phi_str+phi_str+r"$\rangle$",
#           r"$\langle$"+phi_str+phi_str+pi_str+r"$\rangle$", 
#           r"$\langle$"+phi_str+pi_str+phi_str+r"$\rangle$",
#           r"$\langle$"+phi_str+pi_str+pi_str+r"$\rangle$",
#           r"$\langle$"+pi_str+phi_str+phi_str+r"$\rangle$",
#           r"$\langle$"+pi_str+phi_str+pi_str+r"$\rangle$",
#           r"$\langle$"+pi_str+pi_str+phi_str+r"$\rangle$",
#           r"$\langle$"+pi_str+pi_str+pi_str+r"$\rangle$"]

phi_str = r"$\phi$"
chi_str = r"$\chi$"
pi_phi_str = r"$\pi_{\phi}$"
pi_chi_str = r"$\pi_{\chi}$"

# Set of all operators
fields = [phi_str, chi_str, pi_phi_str, pi_chi_str]

# Generate all unique 2-point correlation function labels
labels2 = [r"$\langle $" + A + B + r"$ \rangle$" for A, B in itertools.combinations_with_replacement(fields, 2)]

# Generate all unique 3-point correlation function labels
labels3 = [r"$\langle $" + A + B + C + r"$ \rangle$" for A, B, C in itertools.combinations_with_replacement(fields, 3)]


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
    l2 = plt.plot(tsig, np.abs(MPP_sigma[:, ind[i]]), color = 'black', label = labels2[i], linestyle=(0, (10, 5)))
    lines.append([l1[0],l2[0]])
plt.vlines(x=Nexit, ymax=10**-9, ymin=10**-28, color='gray', linestyle='dashed')
plt.xlabel(r"$N$", fontsize=labsz)
plt.ylabel(r'$\Sigma$', rotation=0, fontsize=labsz)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(left=tsig[0], right=tsig[-1])
plt.ylim(bottom=10**-28, top=10**-9)
plt.grid()
plt.yscale('log')
leg1 = ax.legend(lines[0], ['PyT', 'MPP'], loc='upper left', fontsize=legsz, framealpha=1.0)
ax.add_artist(leg1)
ax.legend(handles=[l[0] for l in lines], loc='lower right', fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/QA_Confront2pt.pdf', format='pdf',bbox_inches='tight')


# Plot 3pt
fig2, ax1 = plt.subplots(figsize=(10,8))
PyT_Alpha = PyT_3pt[:, 1+4+2*nF+6*(2*nF*2*nF):]
MPP_Alpha = MPP_3pt[:, 1+4+2*nF+6*(2*nF*2*nF):]
ind = [0, 1, 6, 7]
lines = []
for i in range(4):
    l1 = plt.plot(tsig, np.abs(PyT_Alpha[:,ind[i]]), label = labels3[ind[i]], color=clr[i])
    l2 = plt.plot(tsig, np.abs(MPP_Alpha[:,ind[i]]), label = labels3[ind[i]], color=clr[i], linestyle = (0, (10, 5)))
    lines.append([l1[0], l2[0]])
plt.vlines(x=Nexit, ymax=10**-26, ymin=10**-50, color='gray', linestyle='dashed')
plt.xlabel(r"$ N$", fontsize=labsz)
plt.ylabel(r'$\alpha$', rotation=0, fontsize=labsz)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(left=tsig[0], right=tsig[-1])
plt.ylim(bottom=10**-50, top=10**-26)
plt.grid()
plt.yscale('log')
leg2 = ax1.legend(lines[0], ['PyT', 'MPP'], loc='lower right', fontsize=legsz, framealpha=1.0)
ax1.add_artist(leg2)
ax1.legend(handles=[l[0] for l in lines], loc='upper center', fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/QA_Confront3pt.pdf', format='pdf',bbox_inches='tight')
##############################################  Show ##################################################################
plt.show()

np.savetxt('Data/cmbdata.txt', (Nexit, k, scaling*PyT_twoPt[-1, 1], scaling*MPP_twoPt[-1, 1], PyT_fnl, MPP_fnl), newline='\n')
