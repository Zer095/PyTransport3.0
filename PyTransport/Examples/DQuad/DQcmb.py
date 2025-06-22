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
plt.plot(back[:,0], back[:, 1])
plt.plot(back[:,0], back[:,3])

fig = plt.figure(2, figsize=(10,8))
plt.plot(back[:,0], back[:, 2])
plt.plot(back[:,0], back[:, 4])

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
labels2 = [r"$\langle $" + A + B + r"$ \rangle$" for A, B in itertools.product(fields, repeat=2)]

# Generate all unique 3-point correlation function labels
labels3 = [r"$\langle $" + A + B + C + r"$ \rangle$" for A, B, C in itertools.product(fields, repeat=3)]

# Colors
#clr = ["#B30000", "#1A53FF", "#5AD45A", "#ED7F2C"]

# Fontsizes
titsz = 24
legsz = 20
labsz = 20
ticksz = 18

##############################################  Plots ##################################################################
# Plot 2pt
PyT_sigma = PyT_twoPt[:, 2 + 2*nF:]
MPP_sigma = MPP_twoPt[:, 2 + 2*nF:]

#ind = [0,1,2,3]
#ind = [4,5,6,7]
#ind = [8,9,10,11]
#ind = [12,13,14,15]
#ind = [0, 5, 6, 10]
#ind = [0, 1, 2, 3, 5, 6, 7, 10, 11, 15]
ind = [0, 1, 2, 3, 10]
fig1, ax = plt.subplots(figsize=(10,8))
lines = []
for i in range(len(ind)):
    l1 = plt.plot(tsig, np.abs(PyT_sigma[:, ind[i]]), label = labels2[ind[i]])
    l2 = plt.plot(tsig, np.abs(MPP_sigma[:, ind[i]]), color = 'black', label = labels2[ind[i]], linestyle=(0, (10, 5)))
    lines.append([l1[0],l2[0]])
plt.vlines(x=Nexit, ymax=10**-10, ymin=10**-70, color='gray', linestyle='dashed')
plt.xlabel(r"$N$", fontsize=labsz)
plt.ylabel(r'$\Sigma$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(left=tsig[0], right=tsig[-1])
plt.ylim(bottom=10**-70, top=10**-10)
plt.grid()
plt.yscale('log')
leg1 = ax.legend( lines[0], ['PyT', 'MPP'], loc='upper right', bbox_to_anchor=(36, 10**-10), bbox_transform=ax.transData, fontsize=legsz, framealpha=1.0)
ax.add_artist(leg1)
ax.legend(handles=[l[0] for l in lines], loc='lower right', bbox_to_anchor=(37.5, 10**-70), bbox_transform=ax.transData, fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/DQ_Confront2pt.pdf', format='pdf',bbox_inches='tight')

# Plot 3pt
fig2, ax1 = plt.subplots(figsize=(10,8))
PyT_Alpha = PyT_3pt[:, 1+4+2*nF+6*(2*nF*2*nF):]
MPP_Alpha = MPP_3pt[:, 1+4+2*nF+6*(2*nF*2*nF):]
# ind = [0,1,2,3,5,6,7,10,11,15,21,22,23,26,27,31,42,43,47,63]
ind = [0,1,2,3,6,10,11,26,42,43]
lines = []
for i in range(len(ind)):
    l1 = plt.plot(tsig, np.abs(PyT_Alpha[:,ind[i]]), label = labels3[ind[i]])
    l2 = plt.plot(tsig, np.abs(MPP_Alpha[:,ind[i]]), label = labels3[ind[i]], color='black', linestyle = (0, (10, 5)))
    lines.append([l1[0], l2[0]])
plt.vlines(x=Nexit, ymax=10**-30, ymin=10**-96, color='gray', linestyle='dashed')
plt.xlabel(r"$ N$", fontsize=labsz)
plt.ylabel(r'$\alpha$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(left=tsig[0], right=tsig[-1])
plt.ylim(bottom=10**-96, top=10**-30)
plt.grid()
plt.yscale('log')
leg2 = ax1.legend(lines[0], ['PyT', 'MPP'], loc='upper right', fontsize=legsz, framealpha=1.0)
ax1.add_artist(leg2)
ax1.legend(handles=[l[0] for l in lines], loc='lower center',  bbox_to_anchor=(85, 10**-70), bbox_transform=ax.transData,fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/DQ_Confront3pt.pdf', format='pdf',bbox_inches='tight')

# #################################################################################################################################

ind = [5, 6, 7, 11, 15]
fig1, ax = plt.subplots(figsize=(10,8))
lines = []
for i in range(len(ind)):
    l1 = plt.plot(tsig, np.abs(PyT_sigma[:, ind[i]]), label = labels2[ind[i]])
    l2 = plt.plot(tsig, np.abs(MPP_sigma[:, ind[i]]), color = 'black', label = labels2[ind[i]], linestyle=(0, (10, 5)))
    lines.append([l1[0],l2[0]])
plt.vlines(x=Nexit, ymax=10**-10, ymin=10**-70, color='gray', linestyle='dashed')
plt.xlabel(r"$N$", fontsize=labsz)
plt.ylabel(r'$\Sigma$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(left=tsig[0], right=tsig[-1])
plt.ylim(bottom=10**-70, top=10**-10)
plt.grid()
plt.yscale('log')
leg1 = ax.legend( lines[0], ['PyT', 'MPP'], loc='upper right', bbox_to_anchor=(36, 10**-10), bbox_transform=ax.transData, fontsize=legsz, framealpha=1.0)
ax.add_artist(leg1)
ax.legend(handles=[l[0] for l in lines], loc='lower right', bbox_to_anchor=(37.5, 10**-70), bbox_transform=ax.transData, fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/DQ_Confront2pt2.pdf', format='pdf',bbox_inches='tight')


# Plot 3pt
fig2, ax1 = plt.subplots(figsize=(10,8))
PyT_Alpha = PyT_3pt[:, 1+4+2*nF+6*(2*nF*2*nF):]
MPP_Alpha = MPP_3pt[:, 1+4+2*nF+6*(2*nF*2*nF):]
ind = [5,7,15,21,22,23,27,31,47,63]
lines = []
for i in range(len(ind)):
    l1 = plt.plot(tsig, np.abs(PyT_Alpha[:,ind[i]]), label = labels3[ind[i]])
    l2 = plt.plot(tsig, np.abs(MPP_Alpha[:,ind[i]]), label = labels3[ind[i]], color='black', linestyle = (0, (10, 5)))
    lines.append([l1[0], l2[0]])
plt.vlines(x=Nexit, ymax=10**-30, ymin=10**-96, color='gray', linestyle='dashed')
plt.xlabel(r"$ N$", fontsize=labsz)
plt.ylabel(r'$\alpha$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(left=tsig[0], right=tsig[-1])
plt.ylim(bottom=10**-96, top=10**-30)
plt.grid()
plt.yscale('log')
leg2 = ax1.legend(lines[0], ['PyT', 'MPP'], loc='upper right', fontsize=legsz, framealpha=1.0)
ax1.add_artist(leg2)
ax1.legend(handles=[l[0] for l in lines], loc='lower center',  bbox_to_anchor=(85, 10**-70), bbox_transform=ax.transData,fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/DQ_Confront3pt2.pdf', format='pdf',bbox_inches='tight')

# Create one figure with two vertically-stacked subplots that share the y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(18, 8))

# Define common y-axis limits
y_min, y_max = 10**-70, 10**-10

# --- First panel ---
ind1 = [0, 1, 2, 3, 10]
lines1 = []
for i in range(len(ind1)):
    l1 = ax1.plot(tsig, np.abs(PyT_sigma[:, ind1[i]]), label=labels2[ind1[i]])
    l2 = ax1.plot(tsig, np.abs(MPP_sigma[:, ind1[i]]), color='black',
                  label=labels2[ind1[i]], linestyle=(0, (10, 5)))
    lines1.append([l1[0], l2[0]])
ax1.vlines(x=Nexit, ymax=y_max, ymin=y_min, color='gray', linestyle='dashed')
ax1.set_xlabel(r"$N$", fontsize=labsz)
ax1.set_ylabel(r'$\Sigma$', rotation=0, fontsize=labsz, labelpad=10)
ax1.tick_params(axis='both', labelsize=ticksz)
ax1.set_xlim(tsig[0], tsig[-1])
ax1.set_ylim(y_min, y_max)
ax1.grid()
ax1.set_yscale('log')
# Adjust legend as needed; here using sample positions
ax1.legend(handles=[l[0] for l in lines1],
           loc='lower right', bbox_to_anchor=(40, y_min),
           bbox_transform=ax1.transData, fontsize=legsz, framealpha=1.0)

# --- Second panel ---
ind2 = [5, 6, 7, 11, 15]
lines2 = []
for i in range(len(ind2)):
    l1 = ax2.plot(tsig, np.abs(PyT_sigma[:, ind2[i]]), label=labels2[ind2[i]])
    l2 = ax2.plot(tsig, np.abs(MPP_sigma[:, ind2[i]]), color='black',
                  label=labels2[ind2[i]], linestyle=(0, (10, 5)))
    lines2.append([l1[0], l2[0]])
ax2.vlines(x=Nexit, ymax=y_max, ymin=y_min, color='gray', linestyle='dashed')
ax2.set_xlabel(r"$N$", fontsize=labsz)
ax2.tick_params(axis='both', labelsize=ticksz)
ax2.set_xlim(tsig[0], tsig[-1])
ax2.set_ylim(y_min, y_max)
ax2.grid()
ax2.set_yscale('log')
leg2 = ax2.legend(lines2[0], ['PyT', 'MPP'],
                  loc='upper right', fontsize=legsz, framealpha=1.0)
ax2.add_artist(leg2)
ax2.legend(handles=[l[0] for l in lines2],
           loc='lower right', bbox_to_anchor=(40, y_min),
           bbox_transform=ax2.transData, fontsize=legsz, framealpha=1.0)

plt.tight_layout()
plt.savefig('Plots/DQ_Confront2pt_all.pdf', format='pdf',bbox_inches='tight')



# Create one figure with two subplots arranged horizontally that share the y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(18, 8))

# Extract the common data
PyT_Alpha = PyT_3pt[:, 1+4+2*nF+6*(2*nF*2*nF):]
MPP_Alpha = MPP_3pt[:, 1+4+2*nF+6*(2*nF*2*nF):]

# Define common y-axis limits
y_min, y_max = 10**-96, 10**-30

###############################################################################
# --- First Panel: Using indices from the second block (10 lines) ---
ind_panel1 = [0, 1, 2, 3, 6, 10, 11, 26, 42, 43]
lines_panel1 = []
for i in range(len(ind_panel1)):
    l1 = ax1.plot(tsig, np.abs(PyT_Alpha[:, ind_panel1[i]]),
                  label=labels3[ind_panel1[i]])
    l2 = ax1.plot(tsig, np.abs(MPP_Alpha[:, ind_panel1[i]]), color='black',
                  linestyle=(0, (10, 5)), label=labels3[ind_panel1[i]])
    lines_panel1.append([l1[0], l2[0]])

ax1.vlines(x=Nexit, ymax=y_max, ymin=y_min, color='gray', linestyle='dashed')
ax1.set_xlabel(r"$ N$", fontsize=labsz)
ax1.set_ylabel(r'$B$', rotation=0, fontsize=labsz, labelpad=10)
ax1.tick_params(axis='both', labelsize=ticksz)
ax1.set_xlim(tsig[0], tsig[-1])
ax1.set_ylim(y_min, y_max)
ax1.grid()
ax1.set_yscale('log')

# Get handles and labels from the plotted lines (using the first artist in each pair)
handles1 = [l[0] for l in lines_panel1]
labels1  = [handle.get_label() for handle in handles1]

# Split the handles/labels into two groups of 5
handles1_first = handles1[:5]
labels1_first  = labels1[:5]
handles1_second = handles1[5:]
labels1_second  = labels1[5:]

# Create two adjacent legends for the first panel.
leg1a = ax1.legend(handles=handles1_first, labels=labels1_first,
                   loc='lower center', bbox_to_anchor=(30, y_min),
                   bbox_transform=ax1.transData, fontsize=legsz, framealpha=1.0)
ax1.add_artist(leg1a)
leg1b = ax1.legend(handles=handles1_second, labels=labels1_second,
                   loc='lower center', bbox_to_anchor=(45, y_min),
                   bbox_transform=ax1.transData, fontsize=legsz, framealpha=1.0)

###############################################################################
# --- Second Panel: Using indices from the first block (10 lines) ---
ind_panel2 = [5, 7, 15, 21, 22, 23, 27, 31, 47, 63]
lines_panel2 = []
for i in range(len(ind_panel2)):
    l1 = ax2.plot(tsig, np.abs(PyT_Alpha[:, ind_panel2[i]]),
                  label=labels3[ind_panel2[i]])
    l2 = ax2.plot(tsig, np.abs(MPP_Alpha[:, ind_panel2[i]]), color='black',
                  linestyle=(0, (10, 5)), label=labels3[ind_panel2[i]])
    lines_panel2.append([l1[0], l2[0]])

ax2.vlines(x=Nexit, ymax=y_max, ymin=y_min, color='gray', linestyle='dashed')
ax2.set_xlabel(r"$ N$", fontsize=labsz)
ax2.tick_params(axis='both', labelsize=ticksz)
ax2.set_xlim(tsig[0], tsig[-1])
ax2.set_ylim(y_min, y_max)
ax2.grid()
ax2.set_yscale('log')

# In the second panel, you already have an upper-right legend for the PyT vs. MPP style:
leg2_upper = ax2.legend(lines_panel2[0], ['PyT', 'MPP'], loc='upper right', 
                        fontsize=legsz, framealpha=1.0)
ax2.add_artist(leg2_upper)

# Get handles and labels for the lower legend from the second panel
handles2 = [l[0] for l in lines_panel2]
labels2  = [handle.get_label() for handle in handles2]

# Split into two groups of 5
handles2_first = handles2[:5]
labels2_first  = labels2[:5]
handles2_second = handles2[5:]
labels2_second  = labels2[5:]

# Create two adjacent legends for the second panel.
leg2a = ax2.legend(handles=handles2_first, labels=labels2_first,
                   loc='lower center', bbox_to_anchor=(25, y_min),
                   bbox_transform=ax2.transData, fontsize=legsz, framealpha=1.0)
ax2.add_artist(leg2a)
leg2b = ax2.legend(handles=handles2_second, labels=labels2_second,
                   loc='lower center', bbox_to_anchor=(40, y_min),
                   bbox_transform=ax2.transData, fontsize=legsz, framealpha=1.0)

plt.tight_layout()
plt.savefig('Plots/DQ_Confront3pt_all.pdf', format='pdf', bbox_inches='tight')
plt.show()

############################################## Show ##################################################################
plt.show()
##############################################  ##################################################################
np.savetxt('Data/cmbdata.txt', (Nexit, k, scaling*PyT_twoPt[-1, 1], scaling*MPP_twoPt[-1, 1], PyT_fnl, MPP_fnl), newline='\n')