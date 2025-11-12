####################################### PyTransPyStep simple example of basic functions ###########################################
from matplotlib import pyplot as plt
import time
import csv
from pylab import *
import numpy as np
from scipy import interpolate 
from scipy.interpolate import UnivariateSpline
import sympy as sym
import subprocess
import os
from pathlib import Path
############################################################################################################################################
from PyTransport.PyTransPy import PyTransSetup
from PyTransport.PyTransPy import PyTransScripts as PyS
import PyTransPBH as PyT;  # import module
###########################################################################################################################################
tols = np.array([10**-17,10**-17])      # Tolerance required for the numerical integrator

nF = PyT.nF()                           # Get the number of fields
nP = PyT.nP()                           # Get the number of parameters

fields = np.array([.0008])              # Initial field value
params = np.zeros(nP)                   # Set up the parameters
params[0]=1.0; params[1]=0.003; params[2]=0.005; params[3]=0.0005; params[4] = .06; params[5] = 0.001

V = PyT.V(fields,params)                # Initial value of the potential
dV = PyT.dV(fields,params)              # Intial value of the derivative of the potential

initial = np.array([fields,-dV/np.sqrt(3.*V)])  # Initial condition with slow-roll conditions

####################################### Calculate the Background evolution ########################################
Nstart = 0.0                                # Initial time 
Nend = 50.0                                 # End time
t = np.linspace(Nstart, Nend, 100000)         # Time steps
back = PyT.backEvolve(t, initial, params, tols, False)       # Background evolution
print(f'Nend = {back[-1,0]}')               # Print End of inflation

# Compute epsilon and eta for each value of the fields
fields = back[:,1:]
epsilons = np.ones(len(t))
etas = np.ones(len(t))
for i in range(len(t)):
    field = fields[i]
    epsilons[i] = np.log10(PyT.Ep(field, params))
    etas[i] = -2.0*PyT.Eta(field, params)

H = []
for i in range(np.size(back[:,0])):
    Hv = PyT.H(back[i,1:3], params)
    H.append(Hv)

H = np.array(H)
H_func = UnivariateSpline(back[:,0], H, s=0)

# calculate epsilon(N)
eps = 0.5*back[:,2]**2/H**2
eps_func = UnivariateSpline(back[:,0], eps, s=0)
#print(eps_func(back[:,0]))
print('Epsilon func')

# calculate eta(N)
deps = eps_func.derivative()
eta = deps(back[:,0])/eps_func(back[:,0])
eta_func = UnivariateSpline(back[:,0], eta, s=0)
#print(eta_func(back[:,0]))
print('Eta func')

pot = []
fields = back[:,1]
for i in range(np.size(back[:,0])):
    field = np.array([back[i,1]])
    V = PyT.V(field,params) 
    pot.append(V)

fig9 = plt.figure(9, figsize=(10,8))
plt.plot(t, pot)
plt.savefig('Plots/PBH_Pot.pdf', format='pdf',bbox_inches='tight')

 
######################################## Read file  ################################################################

Ns = []                 # NExit of each mode k
Ks = []                 # K modes for each NExit
MPP_Pz = []             # MPP Pz
PyT_Pz = []             # PyT Pz
MPP_fNL = []            # MPP f_NL
PyT_fNL = []             # PyT f_NL

file_path = os.path.join( os.path.dirname(__file__),  'Data/Total.csv')
file = open(file_path, 'r')

csvFile = csv.reader(file)
next(csvFile, None)
for line in csvFile:
    Ns.append(float(line[0]))
    Ks.append(float(line[1]))
    factor = (float(line[1])**3.)/(2.*np.pi**2.)
    MPP_Pz.append(factor*float(line[2]))
    PyT_Pz.append(factor*float(line[3]))
    MPP_fNL.append(np.abs(float(line[4])))
    PyT_fNL.append(np.abs(float(line[5])))

###########################################################################################################################################
tols = np.array([10**-12,10**-12]) 
######################################## Two point run  ################################################################
NB = 5.0
NExit = 6.0
k = PyS.kexitN(NExit, back, params, PyT)
factor = (k**3.)/(2.*np.pi**2.)

NstartS, backExitS = PyS.ICsBE(NB, k, back, params, PyT)
tsig = np.linspace(NstartS, back[-1,0], 1000)

# MPP Two-point run
rho = PyT.MPP2(tsig, k, backExitS, params, tols)
twoPtM = PyT.MPPSigma(tsig, k, backExitS, params, rho, True, -1)
sigmaM = twoPtM[:, 1 + 1 + 2*nF:]
pzM = factor*twoPtM[:,1]
print('End MPP 2pt')

# PyT Two-point run
twoPtT = PyT.sigEvolve(tsig, k, backExitS, params,tols,True)
sigmaT = twoPtT[:,1+1+2*nF:]
pzT = factor*twoPtT[:,1]
print('End PyT 2pt')
######################################## Three point run  ################################################################

kM = np.min(np.array([k,k,k]))
NstartA, backExitMinusA = PyS.ICsBE(NB, kM, back, params, PyT)
talp=np.linspace(NstartA, back[-1,0], 1000)

# MPP Three point run
rho3 = PyT.MPP3(talp, k,k,k,backExitMinusA, params, tols)
threePtMPP = PyT.MPPAlpha(talp, k, k, k, backExitMinusA, params, rho3, True)
alphaMPP = threePtMPP[:,1+4+2*nF+6*(2*nF*2*nF):]
zM = threePtMPP[:,1:5]
fnlM = np.abs(( 5.0/6.0*zM[:,3] )/(zM[:,1]*zM[:,2]  + zM[:,0]*zM[:,1] + zM[:,0]*zM[:,2]))

print('End MPP 3pt')


# PyT Three point run
threePtTrans = PyT.alphaEvolve(talp, k, k, k, backExitMinusA, params, tols, True)
alphaTrans = threePtTrans[:,1+4+2*nF+6*(2*nF*2*nF):]
zT = threePtTrans[:,1:5]
fnlT = np.abs(( 5.0/6.0*zT[:,3] )/(zT[:,1]*zT[:,2]  + zT[:,0]*zT[:,1] + zT[:,0]*zT[:,2]))

print('End PyT 3pt')

print(f'MPP 3pt end = {alphaMPP[-1,1]}, PyT 3pt end = {alphaTrans[-1,1]}')
###########################################################################################################################################

# Add the current run to the lists
Ns.insert(0,NExit)
Ks.insert(0, k)
MPP_Pz.insert(0, np.abs(pzM[-1]))
PyT_Pz.insert(0, np.abs(pzT[-1]))
MPP_fNL.insert(0, np.abs(fnlM[-1]))
PyT_fNL.insert(0, np.abs(fnlT[-1]))

###########################################################################################################################################
####################################################### Plots   ###########################################################################

# Plot parameters and useful strings

nexit = r"$\Delta N$ = " + f"{int(back[-1,0] - NExit)}"
phi_str = r"$\phi$"
pi_str = r"$\pi_{\phi}$"

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
colors = ["#b30000", "#7c1158", "#4421af", "#1a53ff", "#0d88e6", "#00b7c7", "#5ad45a", "#8be04e", "#ebdc78"]
clr = ["#B30000", "#1A53FF", "#5AD45A", "#ED7F2C"]

titsz = 24
legsz = 20
labsz = 20
ticksz = 18
# pot = []
# for i in range(len(t)):
#     field = np.array(back[i,1])
#     V = PyT.V(field,params) 
#     pot.append(V)

# fig9 = plt.figure(9, figsize=(10,8))
# plt.plot(t, pot)
# plt.show()
# plt.savefig('Plots/PBH_Pot.pdf', format='pdf',bbox_inches='tight')

# Plot background field evolution
fig1 = plt.figure(1, figsize=(10,8))
plt.plot(back[:,0], back[:,1], color=clr[0], label=r'$\phi$')
plt.plot(back[:,0], back[:,2], color=clr[1], label=r'$\dot{\phi}$')
#plt.title("Background evolution", fontsize = titsz)
plt.xlabel(r"$N$", fontsize=labsz)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.grid()
plt.legend(fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/PBH_BackEvo.pdf', format='pdf',bbox_inches='tight')

# Plot background eta and epsilon evolution
fig2 = plt.figure(2, figsize=(10,8))
plt.plot(back[:,0], np.log10(eps_func(back[:,0])), color = clr[0], label = r'$\log_{10}(\epsilon$)')
plt.plot(back[:,0], eta_func(back[:,0]), color = clr[1], label = r'$\eta$')
#plt.title(r'$\epsilon$ $\text{and}$ $\eta$ evolution', fontsize = titsz)
plt.xlabel(r"$N$", fontsize = labsz)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.grid()
plt.legend(fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/PBH_EpsilonEta.pdf', format='pdf',bbox_inches='tight')

# Plot two-point correlation function
# deltaN = back[-1,0] -  twoPtT[:,0]
# fig3 = plt.figure(3, figsize=(10,8))
# ind = [0,1,3]
# x_ticks = [50, 44, 40, 30, 20, 10, 0]
# for i in ind:
#         plt.plot(deltaN, np.abs(sigmaT[:, i]), color=clr[i], label='PyT ' +labels2[i])
#         plt.plot(deltaN, np.abs(sigmaM[:, i]), color='black', label='MPP '+labels2[i], linestyle='dashed')
# #plt.title("Two-point correlation function "+nexit, fontsize=titsz )
# plt.vlines(x=44, ymax=10**4, ymin=10**-24, linestyle='dashed', color='gray')
# plt.gca().invert_xaxis()
# plt.xlabel(r"$\Delta N$", fontsize=labsz)
# plt.ylabel(r'$\Sigma$', rotation=0, fontsize=labsz)
# plt.xticks(fontsize=ticksz, ticks=x_ticks)
# plt.yticks(fontsize=ticksz)
# plt.grid()
# plt.yscale('log')
# plt.legend(fontsize=legsz, loc='upper right')
# plt.savefig('plot/PBH_Confront2pt.pdf', format='pdf',bbox_inches='tight')

deltaN = back[-1,0] -  twoPtT[:,0]
ind = [0,1,3]
fig3, ax = plt.subplots(figsize=(10,8))
lines = []
for i in range(3):
    l1, = plt.plot(twoPtT[:,0], np.abs(sigmaT[:, ind[i]]), color=clr[i], label=labels2[i])
    l2, = plt.plot(twoPtT[:,0], np.abs(sigmaM[:, ind[i]]), color='black', label=labels2[i], linestyle=(0, (10, 5)))
    lines.append([l1,l2])
#plt.title("Two-point correlation function "+nexit, fontsize=titsz )
plt.vlines(x=NExit, ymax=10**4, ymin=10**-24, linestyle='dashed', color='gray')
#plt.gca().invert_xaxis()
plt.xlabel(r"$N$", fontsize=labsz)
plt.ylabel(r'$\Sigma$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.ylim(10**-24, 10**2)
plt.xlim(twoPtT[0,0], twoPtT[-1,0])
plt.grid()
plt.yscale('log')
leg1 = ax.legend(lines[0], ['PyT', 'MPP'], loc='upper left', fontsize=legsz, framealpha=1.0)
ax.add_artist(leg1)
ax.legend(handles=[l[0] for l in lines], loc='lower left', fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/PBH_Confront2pt.pdf', format='pdf',bbox_inches='tight')

# Plot Three-point correlation function
fig4 = plt.figure(4, figsize=(10,8))
ind = [0, 1, 6, 7]
deltaN = back[-1,0] -  talp
x_ticks = [50, 44, 40, 30, 20, 10, 0]
for i in range(4):
    plt.plot(deltaN, np.abs(alphaTrans[:,ind[i]]), label = 'PyT '+ labels3[ind[i]], color=clr[i])
    plt.plot(deltaN, np.abs(alphaMPP[:,ind[i]]), label = 'Mpp '+ labels3[ind[i]], color=clr[i], linestyle = (0, (10, 5)))
#plt.title("Comparison Three-point correlation functions "+nexit, fontsize=titsz )
plt.vlines(x=44, ymax=10**10, ymin=10**-34, linestyle='dashed', color='gray')
plt.gca().invert_xaxis()
plt.xlabel(r"$\Delta N$", fontsize=labsz)
plt.ylabel(r'$\alpha$', rotation=0, fontsize=labsz)
plt.xticks(fontsize=ticksz, ticks=x_ticks)
plt.yticks(fontsize=ticksz)
plt.grid()
plt.yscale('log')
plt.legend(fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/PBH_Confront3pt.pdf', format='pdf',bbox_inches='tight')

k_ratios = []
PyT_Pz_norm = []
MPP_Pz_norm = []
for i in range(len(Ks)):
    k_ratios.append(Ks[i]/Ks[0])
    PyT_Pz_norm.append(PyT_Pz[i]/PyT_Pz[0])
    MPP_Pz_norm.append(MPP_Pz[i]/MPP_Pz[0])


# Plot Pz spectrum
fig5 = plt.figure(5, figsize=(10,8))
plt.plot(k_ratios, PyT_Pz_norm, color=colors[0], label='PyT') 
plt.plot(k_ratios, MPP_Pz_norm, color='black', label='MPP', linestyle=(0, (10, 5)))
#plt.title(r"$P_{\zeta}$ spectrum", fontsize=titsz)
#plt.gca().invert_xaxis()
plt.xlabel(r"$k/k_0$", fontsize=labsz)
plt.ylabel(r'$\frac{\mathcal{P}_{\zeta}(k)}{\mathcal{P}_{\zeta}(k_0)}$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(k_ratios[0], k_ratios[-1])
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/PBH_SpectraPz.pdf', format='pdf',bbox_inches='tight')

# Plot fNL spectrum
fig6 = plt.figure(6, figsize=(10,8))
plt.plot(k_ratios, PyT_fNL, color=colors[0], label='PyT') 
plt.plot(k_ratios, MPP_fNL, color='black', label='MPP', linestyle=(0, (10, 5)))
#plt.title(r"$f_{NL}$ spectrum", fontsize=titsz)
#plt.gca().invert_xaxis()
plt.xlabel(r"$k/k_0$", fontsize=labsz)
plt.ylabel(r'$f_\text{NL}$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(k_ratios[0], k_ratios[-1])
plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.legend(fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/PBH_SpectraFnL.pdf', format='pdf',bbox_inches='tight')

# Plot background eta and epsilon evolution
fig7 = plt.figure(7, figsize=(10,8))
plt.plot(back[:,0], etas, color = clr[1], label = r'$\eta$')
plt.plot(back[:,0], epsilons, color = clr[0], label = r'$\log_{10}(\epsilon$)', linestyle='dashed')
#plt.title(r'$\epsilon$ $\text{and}$ $\eta$ evolution', fontsize = titsz)
plt.xlabel(r"$N$", fontsize = labsz)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.ylim(-10, 10)
plt.xlim(0, 50)
plt.grid()
plt.legend(fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/PBH_EpsilonEta1.pdf', format='pdf',bbox_inches='tight')

# Plot 3pt correlation function with 2 legends
fig8, ax = plt.subplots(figsize=(10,8))
PyT_lines = []
MPP_lines = []
lines = []
ind = [0, 1, 6, 7]
deltaN = back[-1,0] -  talp
for i in range(4):
    l2, = plt.plot(talp, np.abs(alphaTrans[:,ind[i]]), label = labels3[ind[i]], color=clr[i])
    l1, = plt.plot(talp, np.abs(alphaMPP[:,ind[i]]), label = labels3[ind[i]], color=clr[i], linestyle = (0, (10, 5)))
    PyT_lines.append(l1)
    MPP_lines.append(l2)
    lines.append([l1,l2])
#plt.title("Comparison Three-point correlation functions "+nexit, fontsize=titsz )
plt.vlines(x=NExit, ymax=10**6, ymin=10**-34, linestyle='dashed', color='gray')
# plt.gca().invert_xaxis()
plt.xlabel(r"$N$", fontsize=labsz)
plt.ylabel(r'$B$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(talp[0], talp[-1])
plt.ylim(10**-34,10**6)
plt.grid()
plt.yscale('log')
legend_1 = ax.legend(lines[0], ["MPP", "PyT"], loc='upper right', fontsize=legsz, labelcolor='black', framealpha=1.0)
ax.add_artist(legend_1)
ax.legend(handles=[l[1] for l in lines], loc='lower center', bbox_to_anchor=(15, 10**-32), bbox_transform=ax.transData, fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/PBH_New3pt.pdf', format='pdf',bbox_inches='tight')


ind = [0,1,3]
fig9, ax = plt.subplots(figsize=(10,8))
lines = []
PyT_sigma = sigmaT
MPP_sigma = sigmaM
for i in range(len(ind)):
    l1 = plt.plot(tsig, np.abs(PyT_sigma[:, ind[i]]), label = labels2[ind[i]])
    l2 = plt.plot(tsig, np.abs(MPP_sigma[:, ind[i]]), color = 'black', label = labels2[ind[i]], linestyle=(0, (10, 5)))
    lines.append([l1[0],l2[0]])
plt.vlines(x=NExit, ymax=10**1, ymin=10**-30, color='gray', linestyle='dashed')
plt.xlabel(r"$N$", fontsize=labsz)
plt.ylabel(r'$\Sigma$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(left=tsig[0], right=tsig[-1])
plt.ylim(bottom=10**-30, top=10**1)
plt.grid()
plt.yscale('log')
leg1 = ax.legend( lines[0], ['PyT', 'MPP'], loc='upper right', fontsize=legsz, framealpha=1.0)
ax.add_artist(leg1)
ax.legend(handles=[l[0] for l in lines], loc='lower right', bbox_to_anchor=(10, 10**-70), bbox_transform=ax.transData, fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/PBH_All2pt.pdf', format='pdf',bbox_inches='tight')

# Plot 3pt
fig10, ax1 = plt.subplots(figsize=(10,8))
PyT_Alpha = threePtTrans[:, 1+4+2*nF+6*(2*nF*2*nF):]
MPP_Alpha = threePtMPP[:, 1+4+2*nF+6*(2*nF*2*nF):]
ind = [0,1,3,7]
lines = []
for i in range(len(ind)):
    l1 = plt.plot(tsig, np.abs(PyT_Alpha[:,ind[i]]), label = labels3[ind[i]])
    l2 = plt.plot(tsig, np.abs(MPP_Alpha[:,ind[i]]), label = labels3[ind[i]], color='black', linestyle = (0, (10, 5)))
    lines.append([l1[0], l2[0]])
plt.vlines(x=NExit, ymax=10**-40, ymin=10**6, color='gray', linestyle='dashed')
plt.xlabel(r"$ N$", fontsize=labsz)
plt.ylabel(r'$\alpha$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(left=tsig[0], right=tsig[-1])
plt.ylim(bottom=10**-40, top=10**6)
plt.grid()
plt.yscale('log')
leg2 = ax1.legend(lines[0], ['PyT', 'MPP'], loc='upper right', fontsize=legsz, framealpha=1.0)
ax1.add_artist(leg2)
ax1.legend(handles=[l[0] for l in lines], loc='lower center',  bbox_to_anchor=(10, 10**-70), bbox_transform=ax.transData,fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/DQ_All3pt.pdf', format='pdf',bbox_inches='tight')

plt.show()