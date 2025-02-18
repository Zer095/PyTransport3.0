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

############################################## Set 2pt and 3pt parameters ##################################################################
tols = np.array([10**-12,10**-12])
NB = 6.0
Nexit = back[-1, 0] - 50.0
k = PyS.kexitN(Nexit, back, pvalue, PyT)
print(f'k = {k}')
k = 597659.9124137812
print(f'New k = {k}')
scaling = (k**3)/(2*np.pi**2)

Nstart, backExit = PyS.ICsBE(NB, k, back, pvalue, PyT)
tsig = np.linspace(Nstart, back[-1, 0], 1000)

##############################################  2pt and 3pt runs ##################################################################
tols = np.array([10**-12,10**-12])
print('Start 2pt PyT run')
start = time.time()
PyT_twoPt = PyT.sigEvolve(tsig, k, backExit, pvalue, tols, True)
print(f'End 2pt PyT run, Pz = {scaling*PyT_twoPt[-1, 1]}, time = {time.time()-start}')
print('---------------------------------------------------------------------------')

tols = np.array([10**-13,10**-13])
print('Start 2pt MPP run')
start = time.time()
rho = PyT.MPP2(tsig, k, backExit, pvalue, tols)
MPP_twoPt = PyT.MPPSigma(tsig, k, backExit, pvalue, rho, True)
print(f'End 2pt MPP run, Pz = {scaling*MPP_twoPt[-1, 1]}, time = {time.time()-start}')
print('---------------------------------------------------------------------------')

tols = np.array([10**-13,10**-13])
print('Start 3pt PyT run')
start = time.time()
PyT_3pt = PyT.alphaEvolve(tsig, k, k, k, backExit, pvalue, tols, True)
PyT_fnl = (5/6*PyT_3pt[-1, 4])/(PyT_3pt[-1,1]*PyT_3pt[-1, 2] + PyT_3pt[-1,1]*PyT_3pt[-1, 3] + PyT_3pt[-1,3]*PyT_3pt[-1, 2]) 
print(f'End 3pt PyT run, fnl = {PyT_fnl}, time = {time.time()-start}')
print('---------------------------------------------------------------------------')

tols = np.array([10**-13,10**-13])
print('Start 3pt MPP run')
start = time.time()
rho3 = PyT.MPP3(tsig, k, k, k, backExit, pvalue, tols)
MPP_3pt = PyT.MPPAlpha(tsig, k, k, k, backExit, pvalue, rho3, True)
MPP_fnl = (5/6*MPP_3pt[-1, 4])/(MPP_3pt[-1,1]*MPP_3pt[-1, 2] + MPP_3pt[-1,1]*MPP_3pt[-1, 3] + MPP_3pt[-1,3]*MPP_3pt[-1, 2])
print(f'End 3pt MPP run, fnl = {MPP_fnl}, time = {time.time()-start}')
print('---------------------------------------------------------------------------')

##############################################  Set plot parameters ##################################################################

# Useful strings for titles and labels
nexit = r"$\Delta N$ = " + f"{int(back[-1,0] - Nexit)}"
phi_str = r"$\phi$"
chi_str = r"$\chi$"
pi_str = r"$\pi_{\phi}$"
pi_str1 = r"$\pi_{\chi}$"

# Labels for 2pt and 3pt correlation functions
labels2 = [r"$\langle$" + phi_str+phi_str + r"$\rangle$", 
           r"$\langle$" + phi_str+chi_str+ r"$\rangle$",
           r"$\langle$" +chi_str+phi_str+ r"$\rangle$", 
           r"$\langle$" +chi_str+chi_str+ r"$\rangle$"]

labels3 = [r"$\langle$"+phi_str+phi_str+phi_str+r"$\rangle$",
          r"$\langle$"+phi_str+phi_str+pi_str+r"$\rangle$", 
          r"$\langle$"+phi_str+pi_str+phi_str+r"$\rangle$",
          r"$\langle$"+phi_str+pi_str+pi_str+r"$\rangle$",
          r"$\langle$"+pi_str+phi_str+phi_str+r"$\rangle$",
          r"$\langle$"+pi_str+phi_str+pi_str+r"$\rangle$",
          r"$\langle$"+pi_str+pi_str+phi_str+r"$\rangle$",
          r"$\langle$"+pi_str+pi_str+pi_str+r"$\rangle$"]


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

# # Print results
# print("2-Point Correlation Labels:")
# print(labels2)
# print("\n3-Point Correlation Labels:")
# print(labels3)

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
MPP_sigma = MPP_twoPt[:, 2 + 2*nF:]
ind = [0,1,2,3]
fig1, ax = plt.subplots(figsize=(10,8))
lines = []
for i in range(4):
    l1 = plt.plot(tsig, np.abs(PyT_sigma[:, ind[i]]), color = clr[i], label = labels2[ind[i]])
    l2 = plt.plot(tsig, np.abs(MPP_sigma[:, ind[i]]), color = 'black', label = labels2[ind[i]], linestyle=(0, (10, 5)))
    lines.append([l1[0],l2[0]])
plt.vlines(x=Nexit, ymax=10**-18, ymin=10**-50, color='gray', linestyle='dashed')
#plt.text(30, 10**-42, f'NB = {6.0}\n'+r'PyT $\epsilon$ = -12' +'\n' +r'MPP $\epsilon$ = -13', fontsize=10, bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
#print(f'Handles = {[l[0][0].get_label() for l in lines]}')
plt.xlabel(r"$N$", fontsize=labsz)
plt.ylabel(r'$\Sigma$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(left=tsig[0], right=tsig[-1])
plt.ylim(bottom=10**-50, top=10**-18)
plt.grid()
plt.yscale('log')
leg1 = ax.legend( lines[0], ['PyT', 'MPP'], loc='upper right', bbox_to_anchor=(40, 10**-42), bbox_transform=ax.transData, fontsize=legsz, framealpha=1.0)
ax.add_artist(leg1)
ax.legend(handles=[l[0] for l in lines], loc='lower center', fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/DQ_Confront2pt.pdf', format='pdf',bbox_inches='tight')

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
plt.vlines(x=Nexit, ymax=10**-40, ymin=10**-86, color='gray', linestyle='dashed')
#plt.text(30, 10**-76, f'NB = {6.0}\n'+r'PyT $\epsilon$ = -13'+'\n'+r'MPP $\epsilon$ = -13', fontsize=10, bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
plt.xlabel(r"$ N$", fontsize=labsz)
plt.ylabel(r'$\alpha$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(left=tsig[0], right=tsig[-1])
plt.ylim(bottom=10**-86, top=10**-40)
plt.grid()
plt.yscale('log')
leg2 = ax1.legend(lines[0], ['PyT', 'MPP'], loc='upper right', fontsize=legsz, framealpha=1.0)
ax1.add_artist(leg2)
ax1.legend(handles=[l[0] for l in lines], loc='lower center', fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/DQ_Confront3pt.pdf', format='pdf',bbox_inches='tight')
############################################## Show ##################################################################
plt.show()
##############################################  ##################################################################
np.savetxt('Data/cmbdata.txt', (Nexit, k, scaling*PyT_twoPt[-1, 1], scaling*MPP_twoPt[-1, 1], PyT_fnl, MPP_fnl), newline='\n')