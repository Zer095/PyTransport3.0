from matplotlib import pyplot as plt
import time
from pylab import *
import numpy as np
from scipy import interpolate 
from scipy import optimize
from scipy.interpolate import UnivariateSpline
import sympy as sym
import csv
import os
from pathlib import Path
############################################################################################################################################

############################################################################################################################################

#This file contains simple examples of using the PyTransport package for the single field example of Chen et al.
#It assumes the StepExampleSeptup file has been run to install a Step version of PyTransPyStep
#It is recommended you restart the kernel to insure any updates to PyTransPyStep are imported 

from PyTransport.PyTransPy import PyTransSetup
from PyTransport.PyTransPy import PyTransScripts as PyS

import PyTransMatt as PyT;  # import module

###########################################################################################################################################
nF = PyT.nF()
nP = PyT.nP()
pvalue = np.zeros(nP)

pvalue[0] = 4*10**-12
pvalue[1] = 1
pvalue[2] = 0.5*10**-6
pvalue[3] = 4*10**3
pvalue[4] = 2

fields = np.array([75*10**-3])

V = PyT.V(fields,pvalue)
dV = PyT.dV(fields, pvalue)

initial = np.concatenate((fields, -dV/np.sqrt(3*V)))
###########################################################################################################################################
# Background evolution
tols = np.array([10**-20, 10**-20])
NB = 4.0
Nstart = 0
Nend = 30.

t = np.linspace(Nstart,Nend,10000)

back = PyT.backEvolve(t, initial, pvalue, tols, False)
DeltaN = back[-1,0] - back[:,0]

# Compute epsilon and eta for each value of the fields
fields = back[:,1:]
epsilons = np.ones(len(t))
etas = np.ones(len(t))
for i in range(len(t)):
    field = fields[i]
    epsilons[i] = np.log(PyT.Ep(field, pvalue))
    etas[i] = -2.0*PyT.Eta(field, pvalue)



H = []
for i in range(np.size(back[:,0])):
    Hv = PyT.H(back[i,1:3], pvalue)
    H.append(Hv)

H = np.array(H)
H_func = UnivariateSpline(back[:,0], H, s=0)

# calculate epsilon(N)
eps = 0.5*back[:,2]**2/H**2
eps_func = UnivariateSpline(back[:,0], eps, s=0)
#print(eps_func(back[:,0]))

# calculate eta(N)
deps = eps_func.derivative()
eta = deps([back[:,0]])/eps_func(back[:,0])
eta_func = UnivariateSpline(back[:,0], eta, s=0)
#print(eta_func(back[:,0]))
###########################################################################################################################################

######################################## Read file  ################################################################

factors = []
Ks = []                 # K modes for each NExit
MPP_Pz = []
PyT_Pz = []
MPP_fNL = []            # MPP f_NL
PyT_fNL = []            # PyT f_NL

file_path = os.path.join( os.path.dirname(__file__),  'cluster/SQTotal.csv')
file = open(file_path, 'r')

csvFile = csv.reader(file)
next(csvFile, None)
for line in csvFile:
    # factors = [float(line[0])]
    # Ks.append(float(line[1]))
    # PyT_Pz.append(np.abs(float(line[2])))
    # MPP_Pz.append(np.abs(float(line[3])))
    PyT_fNL.append(np.abs(float(line[4])))
    MPP_fNL.append(np.abs(float(line[5])))

file_path = os.path.join( os.path.dirname(__file__),  'Data/SQzz.csv')
file = open(file_path, 'r')
csvFile = csv.reader(file)
for line in csvFile:
    factors.append(float(line[0]))
    Ks.append(float(line[1]))
    PyT_Pz.append(np.abs(float(line[2])))
    MPP_Pz.append(np.abs(float(line[3])))

factors = factors[:len(MPP_fNL)]
Ks = Ks[:len(MPP_fNL)]
PyT_Pz = PyT_Pz[:len(MPP_fNL)]
MPP_Pz = MPP_Pz[:len(MPP_fNL)]


file_path = os.path.join( os.path.dirname(__file__),  'Data/SQ4.csv')
file = open(file_path, 'r')
csvFile = csv.reader(file)
new_factors = []
new_ks = []
new_PyT_fnl = []
new_MPP_fnl = []
for line in csvFile:
    new_factors.append(float(line[0]))
    new_ks.append(float(line[1]))
    new_PyT_fnl.append(np.abs(float(line[2])))
    new_MPP_fnl.append(np.abs(float(line[3])))

###########################################################################################################################################

######################################## Two point run  ################################################################
NB = 5.0
NExit = 6.0
k = PyS.kexitN(NExit, back, pvalue, PyT)
factor = (k**3.)/(2.*np.pi**2.)
tols = np.array([10**-10, 10**-10])
NstartS, backExitS = PyS.ICsBE(NB, k, back, pvalue, PyT)
tsig = np.linspace(NstartS, back[-1,0], 1000)

# MPP Two-point run
rho = PyT.MPP2(tsig, k, backExitS, pvalue, tols)
twoPtM = PyT.MPPSigma(tsig, k, backExitS, pvalue, rho, True, -1)
sigmaM = twoPtM[:, 1 + 1 + 2*nF:]
pzM = factor*twoPtM[:,1]

print('End MPP Two-point')

# PyT Two-point run
twoPtT = PyT.sigEvolve(tsig, k, backExitS, pvalue,tols,True)
sigmaT = twoPtT[:,1+1+2*nF:]
pzT = factor*twoPtT[:,1]

print('End PyT Two-point')

######################################## Three point run  ################################################################

kM = np.min(np.array([k,k,k]))
NstartA, backExitMinusA = PyS.ICsBE(NB, kM, back, pvalue, PyT)
talp=np.linspace(NstartA, back[-1,0], 1000)

# MPP Three point run
rho3 = PyT.MPP3(talp, k,k,k,backExitMinusA, pvalue, tols)
threePtMPP = PyT.MPPAlpha(talp, k, k, k, backExitMinusA, pvalue, rho3, True)
alphaMPP = threePtMPP[:,1+4+2*nF+6*(2*nF*2*nF):]
zM = threePtMPP[:,1:5]
fnlM = np.abs(( 5.0/6.0*zM[:,3] )/(zM[:,1]*zM[:,2]  + zM[:,0]*zM[:,1] + zM[:,0]*zM[:,2]))
print('End MPP Three-point')

# PyT Three point run
threePtTrans = PyT.alphaEvolve(talp, k, k, k, backExitMinusA, pvalue, tols, True)
alphaTrans = threePtTrans[:,1+4+2*nF+6*(2*nF*2*nF):]
zT = threePtTrans[:,1:5]
fnlT = np.abs(( 5.0/6.0*zT[:,3] )/(zT[:,1]*zT[:,2]  + zT[:,0]*zT[:,1] + zT[:,0]*zT[:,2]))
print('End PyT Three-point')
###########################################################################################################################################
####################################################### fNL Spectrum   ###########################################################################
 
factors = []
kpeak = 5.018093148060832505e+02
kL= 0.0004*kpeak 
for ks in Ks:
    factors.append(ks/kL)

factors = np.array(factors)
#print(f'factors[0] = {factors[0]}, factors[-1] = {factors[-1]}')
###########################################################################################################################################
####################################################### Maldacena Consistency Relation ###########################################################################

PyT_logPz = np.log(np.array(PyT_Pz))
MPP_logPz = np.log(np.array(MPP_Pz))

print(f'Len[factors] = {len(factors)}, Len[PyT_logPz] = {len(PyT_logPz)}')

PyT_logPz_func = UnivariateSpline(factors, PyT_logPz, s=10e-10)
MPP_logPz_func = UnivariateSpline(factors, MPP_logPz, s=10e-10)

PyT_ns = np.multiply(PyT_logPz_func.derivative()(factors), factors)
MPP_ns = np.multiply(MPP_logPz_func.derivative()(factors), factors)


x = sym.Symbol('x', real=True)
f_sym = -5/6*(sym.tanh( 0.001*(x/kL) ) - 1) 

f_k = np.array([f_sym.subs(x, k) for k in Ks])

#print(f'f(k0) = {f_k[0]}, f(k_1) = {f_k[-1]}')

Ns = []
for k in Ks:
    N, b = PyS.ICsBE(0, k, back, pvalue, PyT)
    Ns.append(N)
Ns = np.array(Ns)
#print(f'Ns[0] = {Ns[0]}, Ns[-1] = {Ns[-1]}')

eps_k = eps_func(Ns)

# PyT_THfNL = (5/12)*(PyT_ns - 2*np.multiply(f_k, eps_k))
# MPP_THfNL = (5/12)*(MPP_ns - 2*np.multiply(f_k, eps_k))

# PyT_THfNL = UnivariateSpline(factors, np.abs((5/12)*(PyT_ns  - 2*np.multiply(f_k, eps_k))), s = 1e-10)
# MPP_THfNL = UnivariateSpline(factors, np.abs((5/12)*(MPP_ns  - 2*np.multiply(f_k, eps_k))), s = 1e-10)

PyT_THfNL = UnivariateSpline(factors, (5/12)*(PyT_ns), s = 1e-10)
MPP_THfNL = UnivariateSpline(factors, (5/12)*(MPP_ns), s = 1e-10)

PyT_THfNL = PyT_THfNL(factors)
MPP_THfNL = MPP_THfNL(factors)

# Clean fNL

PyT_fNL_np = np.array(PyT_fNL)
#print(np.where(PyT_fNL_np == 0)[0][0])
PyT_index_0 = np.where(PyT_fNL_np == 0)[0][0]
PyT_index_1 = np.where(PyT_fNL_np == 0)[0][-1]


MPP_fNL_np = np.array(MPP_fNL)
#print(np.where(MPP_fNL_np == 0))

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

# Plot background field evolution
fig1 = plt.figure(1, figsize=(10,8))
plt.plot(back[:,0], back[:,1], color=clr[0], label=r'$\phi$')
plt.plot(back[:,0], back[:,2], color=clr[1], label=r'$\dot{\phi}$')
# plt.title("Background evolution", fontsize = titsz)
plt.xlabel(r"$N$", fontsize=labsz)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.grid()
plt.legend(fontsize=legsz)
plt.savefig('Plots/SQ_BackEvo.pdf', format='pdf',bbox_inches='tight')

# Plot background eta and epsilon evolution
fig2 = plt.figure(2, figsize=(10,8))
plt.plot(back[:,0], eta_func(back[:,0]), color = clr[1], label = r'$\eta$')
plt.plot(back[:,0], np.log10(eps_func(back[:,0])), color = clr[0], label = r'$\log_{10}(\epsilon$)', linestyle='dashed')
# plt.title(r'$\epsilon$ $\text{and}$ $\eta$ evolution', fontsize = titsz)
plt.xlabel(r"$N$", fontsize = labsz)
plt.xlim(back[0,0], back[-1,0])
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.grid()
plt.legend(fontsize=legsz)
plt.savefig('Plots/SQ_EpsilonEta.pdf', format='pdf',bbox_inches='tight')

# Plot two-point correlation function
# deltaN = back[-1,0] -  twoPtT[:,0]
# fig3 = plt.figure(3, figsize=(10,8))
# ind = [0,1,3]
# for i in ind:
#         plt.plot(twoPtT[:,0], np.abs(sigmaT[:, i]), color=clr[i], label='PyT ' +labels2[i])
#         plt.plot(twoPtM[:,0], np.abs(sigmaM[:, i]), color='black', label='MPP '+labels2[i], linestyle='dashed')
# plt.title("Two-point correlation function "+nexit, fontsize=titsz )
# plt.gca().invert_xaxis()
# plt.xlabel(r"$\Delta N$", fontsize=labsz)
# plt.ylabel(r'$\Sigma$', rotation=0, fontsize=labsz)
# plt.xticks(fontsize=ticksz)
# plt.yticks(fontsize=ticksz)
# plt.grid()
# plt.yscale('log')
# plt.legend(fontsize=legsz)
# plt.savefig('Plots/SQConfront2pt.pdf', format='pdf',bbox_inches='tight')

deltaN = back[-1,0] -  twoPtT[:,0]
ind = [0,1,3]
fig3, ax = plt.subplots(figsize=(10,8))
lines = []
for i in range(3):
    l1, = plt.plot(twoPtT[:,0], np.abs(sigmaT[:, ind[i]]), color=clr[i], label=labels2[i])
    l2, = plt.plot(twoPtT[:,0], np.abs(sigmaM[:, ind[i]]), color='black', label=labels2[i], linestyle=(0, (10, 5)))
    lines.append([l1,l2])
#plt.title("Two-point correlation function "+nexit, fontsize=titsz )
plt.vlines(x=int(NExit), ymax=10**14, ymin=10**-24, linestyle='dashed', color='gray')
#plt.gca().invert_xaxis()
plt.xlabel(r"$N$", fontsize=labsz)
plt.ylabel(r'$\Sigma$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(twoPtT[0,0],twoPtT[-1,0])
plt.ylim(10**-22,10**14)
plt.grid()
plt.yscale('log')
leg1 = ax.legend(lines[0], ['PyT', 'MPP'], loc='upper right', fontsize=legsz, framealpha=1.0)
ax.add_artist(leg1)
ax.legend(handles=[l[0] for l in lines], loc='lower right', fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/SQ_Confront2pt.pdf', format='pdf',bbox_inches='tight')

# Plot Three-point correlation function
fig4 = plt.figure(4, figsize=(10,8))
ind = [0, 1, 6, 7]
deltaN = back[-1,0] -  talp
for i in range(4):
    plt.plot(talp, np.abs(alphaMPP[:,ind[i]]), label = 'Mpp '+ labels3[ind[i]], color=clr[i], linestyle = 'dashed')
    plt.plot(talp, np.abs(alphaTrans[:,ind[i]]), label = 'PyT '+ labels3[ind[i]], color=clr[i])
# plt.title("Comparison Three-point correlation functions "+nexit, fontsize=titsz )
plt.gca().invert_xaxis()
plt.xlabel(r"$N$", fontsize=labsz)
plt.ylabel(r'$B$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(talp[0], talp[-1])
plt.ylim()
plt.grid()
plt.yscale('log')
plt.legend(fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/SQ_Confront3pt.pdf', format='pdf',bbox_inches='tight')

# Plot fNL spectrum
fig5 = plt.figure(5, figsize=(10,8))
plt.plot(factors, PyT_fNL, color=colors[0], label='PyT') 
plt.plot(factors, MPP_fNL, color='black', label='MPP', linestyle=(0, (10, 5)))
# plt.plot(factors, PyT_THfNL, color=clr[1], label='PyT Maldacena')
# plt.plot(factors, MPP_THfNL, color=clr[2], label=' MPP Maldacena', linestyle='dashed')
#plt.title(r"$f_{NL}$ spectrum", fontsize=titsz)
#plt.gca().invert_xaxis()
plt.xlabel(r"$k_{\rm short}$/$k_{\rm long}$", fontsize=labsz)
plt.ylabel(r'$f_{NL, SQ}$', rotation=0, fontsize=labsz, labelpad=10)
ax = plt.gca()
ax.yaxis.set_label_coords(-0.15, 0.5)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(factors[0],factors[-1])
plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.legend(fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/SQ_SpectraFnL.pdf', format='pdf',bbox_inches='tight')

k_ratios = []
PyT_Pz_norm = []
MPP_Pz_norm = []
for k in Ks:
    k_ratios.append(k/Ks[0])
    ind = Ks.index(k)
    PyT_Pz_norm.append(PyT_Pz[ind]/PyT_Pz[0])
    MPP_Pz_norm.append(MPP_Pz[ind]/MPP_Pz[0])

fig6 = plt.figure(6, figsize=(10,8))
plt.plot(factors, PyT_Pz_norm, color=colors[0], label='PyT') 
plt.plot(factors, MPP_Pz_norm, color='black', label='MPP', linestyle=(0, (10, 5)))
# plt.vlines(x=k_ratios[PyT_index_0], ymin=min(PyT_Pz_norm), ymax=max(PyT_Pz_norm), color='gray', linestyle='dashed')
# plt.vlines(x=k_ratios[PyT_index_1], ymin=min(PyT_Pz_norm), ymax=max(PyT_Pz_norm), color='gray', linestyle='dashed')
# plt.title(r"$f_{NL}$ spectrum", fontsize=titsz)
# plt.gca().invert_xaxis()
plt.xlabel(r"$k$/$k_0$", fontsize=labsz)
plt.ylabel(r'$\frac{\mathcal{P}_{\zeta}(k)}{\mathcal{P}_{\zeta}(k_0)}$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(k_ratios[0], k_ratios[-1])
plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.legend(fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/SQ_SpectraPz.pdf', format='pdf',bbox_inches='tight')

fig7, ax = plt.subplots(figsize=(10,8))
PyT_lines = []
MPP_lines = []
lines = []
ind = [0, 1, 6, 7]
deltaN = back[-1,0] -  talp
for i in range(4):
    l1, = plt.plot(talp, np.abs(alphaMPP[:,ind[i]]), label = labels3[ind[i]], color=clr[i], linestyle = (0, (10, 5)))
    l2, = plt.plot(talp, np.abs(alphaTrans[:,ind[i]]), label = labels3[ind[i]], color=clr[i])
    PyT_lines.append(l1)
    MPP_lines.append(l2)
    lines.append([l1,l2])
plt.vlines(x=int(NExit), ymax=10**18, ymin=10**-16, linestyle='dashed', color='gray')
# plt.title("Comparison Three-point correlation functions "+nexit, fontsize=titsz )
#plt.gca().invert_xaxis()
plt.xlabel(r"$N$", fontsize=labsz)
plt.ylabel(r'$B$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(talp[0], talp[-1])
plt.ylim(10**-16, 10**18)
plt.grid()
plt.yscale('log')
legend_1 = ax.legend(lines[0], ["MPP", "PyT"], loc='lower left', fontsize=legsz, labelcolor='black', framealpha=1.0)
ax.add_artist(legend_1)
ax.legend(handles=[l[1] for l in lines], loc='upper right', fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/SQ_Confront3pt.pdf', format='pdf',bbox_inches='tight')

fig10 = plt.figure(10, figsize=(10,8))
plt.plot(factors, np.abs(PyT_THfNL), color=clr[1], label='PyT Maldacena')
plt.plot(factors, PyT_fNL, color=colors[0], label='PyT', linestyle=(0, (10, 5))) 
# plt.plot(factors, MPP_fNL, color=clr[2], label='MPP', linestyle='dashed')
# plt.plot(factors, MPP_THfNL(factors), color=clr[3], label='MPP Maldacena')
plt.yscale('log')
plt.xscale('log')
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(factors[0], factors[-1])
plt.grid()
plt.legend(fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/SQ_PyTMaldacena.pdf', format='pdf',bbox_inches='tight')

fig11 = plt.figure(11, figsize=(10,8))
# plt.plot(factors, PyT_THfNL(factors), color=clr[1], label='PyT Maldacena')
# plt.plot(factors, PyT_fNL, color=colors[0], label='PyT', linestyle='dashed') 
plt.plot(factors, np.abs(MPP_THfNL), color=clr[3], label='MPP Maldacena')
plt.plot(factors, MPP_fNL, color=clr[2], label='MPP', linestyle=(0, (10, 5)))
plt.yscale('log')
plt.xscale('log')
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlim(factors[0], factors[-1])
plt.grid()
plt.legend(fontsize=legsz, framealpha=1.0, loc='lower left')
plt.savefig('Plots/SQ_MppMaldacena.pdf', format='pdf',bbox_inches='tight')

fig12 = plt.figure(12, figsize=(10,8))
plt.plot(factors, np.abs(PyT_THfNL), color=clr[1], label='PyT ')
plt.plot(factors, np.abs(MPP_THfNL), color=clr[3], label='MPP ', linestyle='dashed')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.savefig('Plots/Maldacena.pdf', format='pdf',bbox_inches='tight')

fig13 = plt.figure(13, figsize=(10,8))
plt.plot(new_factors, np.abs(new_PyT_fnl), color=clr[1], label='PyT ')
plt.plot(new_factors, np.abs(new_MPP_fnl), color=clr[3], label='MPP ', linestyle='dashed')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.savefig('Plots/NewfNL.pdf', format='pdf',bbox_inches='tight')


import matplotlib.pyplot as plt

# Create one figure with two horizontally-stacked subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# --- Left subplot: PyT Maldacena ---
ax1.plot(factors, np.abs(PyT_THfNL), color=clr[1], label='PyT Maldacena')
ax1.plot(factors, PyT_fNL, color=colors[0], label='PyT', linestyle=(0, (10, 5)))
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylabel(r'$f_{NL, SQ}$', fontsize=labsz, labelpad=10, rotation=0)
ax1.yaxis.set_label_coords(-0.15, 0.5)
ax1.set_xlabel(r"$k_{\rm short}$/$k_{\rm long}$", fontsize=labsz)
ax1.tick_params(axis='both', labelsize=ticksz)
ax1.set_xlim(factors[0], factors[-1])
ax1.grid(True)
ax1.legend(fontsize=legsz, framealpha=1.0)

# --- Right subplot: MPP Maldacena ---
ax2.plot(factors, np.abs(MPP_THfNL), color=clr[3], label='MPP Maldacena')
ax2.plot(factors, MPP_fNL, color=clr[2], label='MPP', linestyle=(0, (10, 5)))
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel(r"$k_{\rm short}$/$k_{\rm long}$", fontsize=labsz)
ax2.tick_params(axis='both', labelsize=ticksz)
ax2.set_xlim(factors[0], factors[-1])
ax2.grid(True)
ax2.legend(fontsize=legsz, framealpha=1.0, loc='lower left')
plt.tight_layout()
plt.savefig('Plots/Combined_SQ.pdf', format='pdf', bbox_inches='tight')



fig14 = plt.figure(figsize=(10,8))
plt.plot(twoPtT[:,0], np.abs(twoPtT[:,1]), color=colors[0], label='PyT') 
plt.plot(twoPtM[:,0], np.abs(twoPtM[:,1]), color='black', label='MPP', linestyle=(0, (10, 5)))
plt.vlines(x=NExit, ymin=min(np.abs(twoPtM[:,1])), ymax=max(np.abs(twoPtT[:,1])), color='gray', linestyle='dashed')
plt.xlabel(r"$N$", fontsize=labsz)
plt.ylabel(r'$\zeta$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.ylim(0.3*10**3, 10**7)
plt.xlim(twoPtT[0,0], twoPtT[-1,0])
plt.grid()
plt.yscale('log')
plt.legend(fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/Zeta_Evo.pdf', format='pdf',bbox_inches='tight')

fig15 = plt.figure(figsize=(10,8))
plt.plot(threePtMPP[:,0], np.abs(fnlM), color=colors[0], label=r'MPP $f_\text{NL}$') 
plt.vlines(x=NExit, ymin=10**-7, ymax=10**2, color='gray', linestyle='dashed')
plt.xlabel(r"$N$", fontsize=labsz)
plt.ylabel(r'$f_\text{NL}\left(k_\text{CMB}\right)$', rotation=0, fontsize=labsz, labelpad=10)
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.ylim(10**2, 10**-7)
plt.xlim(threePtMPP[0,0], threePtMPP[-1,0])
plt.grid()
plt.yscale('log')
plt.legend(fontsize=legsz, framealpha=1.0)
plt.savefig('Plots/fnl_evo.pdf', format='pdf',bbox_inches='tight')



###########################################################################################################################################
plt.show()