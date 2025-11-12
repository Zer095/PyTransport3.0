# This script is part of the analysis of the Axion Monodromy Inflationary model (2412.05762, 2505.19066)
# In this script we analize the slow-roll part of the solution
# Import packages

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

from PyTransport.PyTransPy import PyTransSetup
from PyTransport.PyTransPy import PyTransScripts as PyS

import PyTransAxQ as PyT
from helper import *
#plt.rcParams['text.usetex'] = True
###############################################################################################################
##################################### Set initial values ######################################################

nF = PyT.nF()
nP = PyT.nP()

# Free parameters/quantities
m_phi = 1.
phi_0 = 16.
lambda_ratio = 100       # should be >> 1
rho_mass_ratio = 409/4  # should be >> 1
alpha = 30              # should be >> 1
beta = 10e-2           # should be << 1

# Parameters derived by this quantities
f = (-2)/(alpha*phi_0)
a = -(beta/alpha)*(2/m_phi)
lmbd = (lambda_ratio/np.sqrt(alpha))*(2/phi_0)
m_rho = rho_mass_ratio*0.5*m_phi*phi_0**2
rho_0 = -1/(3*m_phi*m_rho*lmbd)

# Set parameters potential
params = np.zeros(nP)
params[0] = lmbd
params[1] = m_phi
params[2] = a
params[3] = f
params[4] = m_rho
params[5] = rho_0

print(f'Rho_0 = {rho_0}')

# vel_phi = -np.sqrt(2*m_phi/3.)
vel_phi = -1.*((m_phi*phi_0) / np.sqrt((3./2.)*m_phi*phi_0**2))
#rho = (vel_phi**2)/(2*lmbd*m_rho) + rho_0
rho = 0

fields = np.array([phi_0, rho])
V = PyT.V(fields,params) # calculate potential from some initial conditions
dV=PyT.dV(fields,params) # calculate derivatives of potential

vel = np.array([vel_phi, 0.])
initial = np.concatenate((fields, vel)) # sets an array containing field values and there derivative in cosmic time 
                                                      # (set using the slow roll equation)

####################################### Calculate PyT Background evolution ########################################
Nstart = 0.
Nend = 65.0
t = np.linspace(Nstart, Nend, 100000)
tols = np.array([10**-30,10**-30])
back = PyT.backEvolve(t, initial, params, tols, True)
print(f'Inflation ends at N = {back[-1,0]}')

####################################### Calculate Slow-Roll background evolution #################################

# Slow-roll solutions
vel_phi = -1.*((m_phi*phi_0) / np.sqrt((3./2.)*m_phi*phi_0**2))
#vel_phi = -np.sqrt(2*m_phi/3.)
# rho = (vel_phi**2)/(2*lmbd*m_rho) + rho_0
rho = 0
h = np.sqrt( (m_phi*phi_0**2)/6. +  (vel_phi**2)/6. + (0.**2)/6. + (m_rho*(rho - rho_0)**2)/6. )
SR_sol = [ [h, phi_0, rho, vel_phi , 0.] ] 
dn = back[1,0] - back[0,0]
h = np.sqrt( (m_phi*phi_0**2)/6. +  (vel_phi**2)/6. + (0.**2)/6. + (m_rho*(rho - rho_0)**2)/6. )
phi = phi_0
# for i in range(1,len(back[:,0])):
#     phi += (vel_phi*dn)/h
#     vel_phi = -1.*(m_phi*(phi))/(3.*h)
#     rho = (vel_phi**2)/(2*lmbd*m_rho) + rho_0
#     vel_rho = (rho - SR_sol[-1][2])/(h*dn)
#     # rho = 0.
#     # vel_rho = 0.
#     h = np.sqrt( (m_phi*phi**2)/6. +  (vel_phi**2)/6. + (vel_rho**2)/6. + (m_rho*(rho - rho_0)**2)/6. )

#     SR_sol.append([h, phi, rho, vel_phi, vel_rho])

# SR_sol = np.array(SR_sol)

# Analytical solutions
n = back[:,0]
phi = phi_0 - np.sqrt(m_phi*2/3)*(n/h)
h = phi*np.sqrt(m_phi/6.)
phi = phi_0 - m_phi*np.sqrt(2/3)*(n/h)
h = phi*np.sqrt(m_phi/6.)
v_phi = -np.sqrt(m_phi*2/3)*np.ones(len(n))
rho = (v_phi**2)/(2*lmbd*m_rho) + rho_0
v_rho = 0.*np.ones(len(n))

zipped = zip(h, phi, rho, v_phi, v_rho)

SR_sol = np.array([list(time_values) for time_values in zipped])

# Find the end of inflation
SR_epsilon = Epsilon(back[:,0], SR_sol[:,0])
n_max = np.where(SR_epsilon > 1)[0][0]
####################################### Calculate perturbed solutions #################################
# Unpack solutions
time = back[:n_max, 0]
h = SR_sol[:n_max,0]
phi = SR_sol[:n_max,1]
rho = SR_sol[:n_max,2]
vel_phi = SR_sol[:n_max,3]
vel_rho = SR_sol[:n_max,4]

# Unpack PyT solutions
PyT_h = PyT_H(back,params)[:n_max]
PyT_phi = back[:n_max,1]
PyT_rho = back[:n_max,2]
PyT_v_phi = back[:n_max,3]
PyT_v_rho = back[:n_max,4]

###############################################################################################################
# Build the perturbed solution

#phi solution
vel_phi_pert = (3*beta/alpha)*vel_phi*np.cos(phi/f - np.pi/2 - phi[0]/f)
phi_pert = -(3*beta*f/alpha)*np.sin(phi/f - phi[0]/f)

# rho solution
B = (-a/lmbd)/np.sqrt((m_rho - (vel_phi/f)**2)**2 + 9*h**2*(vel_phi/f)**2) # amplitude
delta = np.arcsin((m_rho - (vel_phi/f)**2)/np.sqrt((m_rho - (vel_phi/f)**2)**2 + 9*h**2*(vel_phi/f)**2)) #phase
rho_pert = B*np.cos(phi/f + delta )
vel_rho_pert = UnivariateSpline(time, rho_pert, s=0).derivative()(time)*h
vel_rho_pert = B*(vel_phi/f)*np.sin(phi/f + delta)
rho_pert = B*np.cos(phi/f + delta - np.pi/2 - (phi[0]/f + delta[0]) )
###############################################################################################################
# Full solitions
FA_vel_phi = vel_phi + vel_phi_pert
FA_phi = phi + phi_pert
FA_vel_rho = vel_rho + vel_rho_pert
FA_rho = rho + rho_pert
# Translate solutions

FA_vel_phi = FA_vel_phi + (PyT_v_phi[0] - FA_vel_phi[0])
# FA_vel_rho = FA_vel_rho + (PyT_v_rho[0] - FA_vel_rho[0])
# FA_rho = FA_rho + (PyT_rho[0] - FA_rho[0])
# Corrected solutions
n = np.where(back[:,0] > 10)[0][0]
delta_phi = back[n,1] - FA_phi[n]
delta_rho = back[n,2] - FA_rho[n]
delta_v_phi = back[n, 3] - FA_vel_phi[n]
delta_v_rho = back[n, 4] - FA_vel_rho[n]

c_time = back[n:,0]
c_phi = FA_phi + delta_phi
c_rho = FA_rho + delta_rho
c_v_phi = FA_vel_phi + delta_v_phi
c_v_rho = FA_vel_rho + delta_v_rho
###############################################################################################################
# Analytical Power-spectrum
SR_epsilon = SR_epsilon[:n_max]

Nstar = 5.
index_star = np.where(time > Nstar)[0][0]
N_out = []
k_out = []

# sol = zip(time, FA_phi, FA_rho, FA_vel_phi, FA_vel_rho)
# for i in range(400):
#     print(f'\n Run num {i} \n')
#     N = Nstar + i*0.15
#     N_out.append(N)
#     k = PyS.kexitN(N, back, params, PyT)
#     k_out.append(k)


######################## Plots #########################################
# φ SR-comparison
fig1, ax1 = plt.subplots(1,2, figsize=(10,6))
ax1[0].plot(time, PyT_phi, label=r'PyT $\phi$')
ax1[0].plot(time, FA_phi, label=r'FA Slow-roll $\phi$', linestyle='dashed')

# φ̇ comparison
ax1[1].plot(time, PyT_v_phi, label=r'PyT $\dot \phi$')
ax1[1].plot(time, FA_vel_phi, label=r'FA Slow-roll $\dot \phi$', linestyle='dashed')

# Inset for φ
axins0 = inset_axes(ax1[0], width="40%", height="40%", loc="lower left")
axins0.plot(time, PyT_phi)
axins0.plot(time, FA_phi, linestyle='dashed')
axins0.set_xlim(20, 22)
axins0.set_ylim(12.6, 13.4)  # Adjust as needed
axins0.set_ylim()
axins0.set_xticklabels([])
axins0.set_yticklabels([])
mark_inset(ax1[0], axins0, loc1=2, loc2=4, fc="none", ec="0.5")

# Inset for φ̇
axins1 = inset_axes(ax1[1], width="40%", height="40%", loc="upper left")
axins1.plot(time, PyT_v_phi)
axins1.plot(time, FA_vel_phi, linestyle='dashed')
axins1.set_xlim(20, 22)
axins1.set_ylim(-0.8260, -0.8050)  # Adjust as needed
axins1.set_xticklabels([])
axins1.set_yticklabels([])
mark_inset(ax1[1], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

# Labels and title
fig1.suptitle(r'$\phi$ and $\dot \phi$ solutions')
ax1[0].set_xlabel(r'$N$')
ax1[1].set_xlabel(r'$N$')
ax1[0].legend(loc='upper right')
ax1[1].legend(loc='upper right')

plt.savefig('../Plots/PERT_phi_sol.pdf', format='pdf', bbox_inches='tight')
##################################################################################################
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 6))

# ρ comparison
ax2[0].plot(time, PyT_rho, label=r'PyT $\rho$')
ax2[0].plot(time, FA_rho, label=r'FA Slow-roll $\rho$', linestyle='dashed')
# ρ̇ comparison
ax2[1].plot(time, PyT_v_rho, label=r'PyT $\dot \rho$')
ax2[1].plot(time, FA_vel_rho, label=r'FA Slow-roll $\dot \rho$', linestyle='dashed')

# Inset for ρ
axins0 = inset_axes(ax2[0], width="40%", height="40%", loc="lower left")
axins0.plot(time, PyT_rho)
axins0.plot(time, FA_rho, linestyle='dashed')
axins0.set_xlim(20, 22)
axins0.set_ylim(-2e-6, 1.5e-6)  # Adjust as needed
axins0.set_xticklabels([])
axins0.set_yticklabels([])
mark_inset(ax2[0], axins0, loc1=2, loc2=4, fc="none", ec="0.5")

# Inset for ρ̇
axins1 = inset_axes(ax2[1], width="40%", height="40%", loc="lower right")
axins1.plot(time, PyT_v_rho)
axins1.plot(time, FA_vel_rho, linestyle='dashed')
axins1.set_xlim(20, 22)
axins1.set_ylim(-0.0003, 0.0003)  # Adjust as needed
axins1.set_xticklabels([])
axins1.set_yticklabels([])
mark_inset(ax2[1], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

# Labels and title
fig2.suptitle(r'$\rho$ and $\dot \rho$ solutions')
ax2[0].set_xlabel(r'$N$')
ax2[1].set_xlabel(r'$N$')
ax2[0].legend(loc='lower right')
ax2[1].legend(loc='upper right')

plt.savefig('../Plots/PERT_rho_sol.pdf', format='pdf', bbox_inches='tight')

###################################################################################################
# fig11, ax11 = plt.subplots(1, 2, figsize=(10,6))

# # phi
# ax11[0].plot(c_time, back[n:,1], label=r'PyT $\phi$')
# ax11[0].plot(c_time, c_phi[n:], label=r'Corrected FA $\phi$', linestyle='dashed')
# # \dot \phi
# ax11[1].plot(c_time, back[n:,3], label=r'PyT $\dot \phi$')
# ax11[1].plot(c_time, c_v_phi[n:], label=r'Corrected FA $\dot \phi$', linestyle='dashed')

# fig11.suptitle(r'FA $\phi_0$ and $\dot \phi_0$ small window')
# ax11[0].set_xlabel(r'$N$')
# ax11[1].set_xlabel(r'$N$')
# ax11[0].set_xlim(10, 20)
# ax11[1].set_xlim(10, 20)
# ax11[0].set_ylim(12.0, 13.8)
# ax11[1].set_ylim(-0.8250, -0.8050)
# ax11[0].legend()
# ax11[1].legend()
# plt.savefig('../Plots/PERT_phi_reduced.pdf', format='pdf', bbox_inches='tight')
# ###################################################################################################
# fig12, ax12 = plt.subplots(1, 2, figsize=(10,6))

# # phi
# ax12[0].plot(c_time, back[n:,2], label=r'PyT $\rho$')
# ax12[0].plot(c_time, c_rho[n:], label=r'Corrected FA $\rho$', linestyle='dashed')
# # \dot \phi
# ax12[1].plot(c_time, back[n:,4], label=r'PyT $\dot \rho$')
# ax12[1].plot(c_time, c_v_rho[n:], label=r'Corrected FA $\dot \rho$', linestyle='dashed')

# fig12.suptitle(r'FA $\rho_0$ and $\dot \rho_0$ small window')
# ax12[0].set_xlabel(r'$N$')
# ax12[1].set_xlabel(r'$N$')
# ax12[0].set_xlim(10, 20)
# ax12[1].set_xlim(10, 20)
# ax12[0].set_ylim(-2e-6, 1e-6)
# ax12[1].set_ylim(-0.0003, 0.0003)
# ax12[0].legend()
# ax12[1].legend()
# plt.savefig('../Plots/PERT_rho_reduced.pdf', format='pdf', bbox_inches='tight')

###################################################################################################
plt.show()