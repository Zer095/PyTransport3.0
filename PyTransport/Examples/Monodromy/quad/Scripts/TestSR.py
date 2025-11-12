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

import PyTransAxQ0 as PyT
from helper import *
#plt.rcParams['text.usetex'] = True
###############################################################################################################
##################################### Set initial values ######################################################

nF = PyT.nF()
nP = PyT.nP()

# Free parameters/quantities
m_phi = 1.                  # Controls the energy scale
phi_0 = 17.                 # Controls inflation's duration
lambda_ratio = 100          # should be >> 1
rho_mass_ratio = 409/4      # should be >> 1
alpha = 30                  # should be >> 1
beta = 10e-2                # should be << 1

# Parameters derived by this quantities
f = (-2)/(alpha*phi_0)
a = -(beta/alpha)*(2/m_phi)
lmbd = (lambda_ratio/np.sqrt(alpha))*(2/phi_0)
m_rho = rho_mass_ratio*0.5*m_phi*phi_0**2
rho_0 = -1./(3.*m_phi*m_rho*lmbd)

# Set parameters potential
params = np.zeros(nP)
params[0] = lmbd
params[1] = m_phi
params[2] = a
params[3] = f
params[4] = m_rho
params[5] = rho_0


vel_phi = -np.sqrt(2.*m_phi/3.)
rho = (vel_phi**2)/(2*lmbd*m_rho) + rho_0
fields = np.array([phi_0, rho])
V = PyT.V(fields,params) # calculate potential from some initial conditions
dV=PyT.dV(fields,params) # calculate derivatives of potential
vel = np.array([vel_phi, 0.])
initial = np.concatenate((fields, vel)) # sets an array containing field values and there derivative in cosmic time 
                                                      # (set using the slow roll equation)

# print(f'Initial conditions (phi, rho, vel_phi, SR_v_rho) = {initial}')
####################################### Calculate PyT Background evolution ########################################
Nstart = 0.
Nend = 73.0
t = np.linspace(Nstart, Nend, 10000)
tols = np.array([10**-30,10**-30])
back = PyT.backEvolve(t, initial, params, tols, True)
print(f'Inflation ends at N = {back[-1,0]}')
####################################### Calculate Slow-Roll background evolution #################################

# Slow-roll solutions
vel_phi = -np.sqrt(2*m_phi/3.)
rho = (vel_phi**2)/(2*lmbd*m_rho) + rho_0
phi = phi_0
h = np.sqrt( (m_phi*phi_0**2)/6. )# +  (vel_phi**2)/6. + (0.**2)/6. + (m_rho*(rho - rho_0)**2)/6. )
SR_sol = [ [h, phi, rho, vel_phi , 0.] ] 
dn = back[1,0] - back[0,0]

# Analytical solutions
n = back[:,0]
h, phi, rho, v_phi, v_rho = get_sol(n, params, phi_0)
for i in range(1,len(n)):
    if not np.isnan(phi[-i]):
        indx = len(n) - i
        break

zipped = zip(h, phi, rho, v_phi, v_rho)


h, phi, rho, v_phi, v_rho = get_sol(n, params, phi_0)

zipped = zip(h, phi, rho, v_phi, v_rho)

SR_sol = np.array([list(time_values) for time_values in zipped])

SR_epsilon = Epsilon(back[:indx,0], SR_sol[:indx,0])

try:
    n_max = np.where(SR_epsilon > 1)[0][0]
except IndexError:
    n_max = len(back[:,0]) - 1

print(f'N indices = {n_max}, N = {back[n_max,0]}')

####################################### Calculate background quantities #################################
time = back[:n_max,0]
# Unpack SR solutions
SR_h = SR_sol[:n_max,0]
SR_phi = SR_sol[:n_max,1]
SR_rho = SR_sol[:n_max,2]
SR_v_phi = SR_sol[:n_max,3]
SR_v_rho = SR_sol[:n_max,4]

# Unpack PyT solutions
PyT_h = PyT_H(back,params)[:n_max]
PyT_phi = back[:n_max,1]
PyT_rho = back[:n_max,2]
PyT_v_phi = back[:n_max,3]
PyT_v_rho = back[:n_max,4]
###############################################################################################################

# Compute 3H^2 with both solutions
# SR solutions
SR_pot_phi, SR_pot_rho, SR_k_phi, SR_k_rho = Energy_SR(SR_sol, params, True, n_max)
SR_3h2 = 3*(SR_h**2)
# PyTransport sol
PyT_pot_phi, PyT_pot_rho, PyT_k_phi, PyT_k_rho = Energy_SR(back, params, False, n_max)
PyT_3h2 = 3*(PyT_H(back, params)**2)[:n_max]

###############################################################################################################
# Compute Epsilon with both solutions
SR_epsilon = Epsilon(time, SR_h)
PyT_epsilon = Epsilon(time, PyT_h)

###############################################################################################################
# Compute Epsilon with both solutions
# SR_eta = Eta(time, SR_epsilon)
# PyT_eta = Eta(time, PyT_epsilon)

####################################### Calculate parameters evolution #################################
# Slow roll
SR_alpha = (SR_v_phi*SR_h)/(SR_h*f)
SR_b = a/(m_phi*SR_phi*f)
SR_lmbd = (SR_v_phi*SR_h)/(lmbd*SR_h*np.sqrt(SR_alpha))

# PyTransport
PyT_alpha = PyT_v_phi/(PyT_h*f)
PyT_b = a/(m_phi*PyT_phi*f) 
PyT_lmbd = PyT_v_phi/(lmbd*PyT_h*np.sqrt(PyT_alpha))

####################################### Calculate discarded equation #################################
# Slow-roll
# phi equation
SR_phi_term1 = np.abs( UnivariateSpline(time, (SR_v_phi*SR_h)).derivative()(time)*SR_h )
SR_phi_term2 = np.abs( 3*SR_h*(SR_v_phi*SR_h) )
SR_phi_term3 = np.abs( m_phi*SR_phi )
SR_phi_term4 = np.abs( ((SR_v_phi*SR_h)*(SR_v_rho*SR_h))/lmbd )
SR_phi_term5 = np.abs( (m_phi*SR_phi*SR_rho)/lmbd )
# rho equation
SR_rho_term1 = np.abs( UnivariateSpline(time, (SR_v_rho*SR_h)).derivative()(time)*SR_h )
SR_rho_term2 = np.abs( 3*SR_h*((SR_v_rho*SR_h)*SR_h) )
SR_rho_term3 = np.abs( (SR_rho*(SR_v_phi*SR_h)**2)/(2*lmbd**2) )
SR_rho_term4 = np.abs( ((SR_v_phi*SR_h)**2)/(2*lmbd) )
SR_rho_term5 = np.abs( m_rho*(SR_rho - rho_0) )

# PyTransport
# phi equation
PyT_phi_term1 = np.abs(UnivariateSpline(time, PyT_v_phi).derivative()(time)*PyT_h )
PyT_phi_term2 = np.abs( 3*PyT_h*PyT_v_phi )
PyT_phi_term3 = np.abs( m_phi*PyT_phi )
PyT_phi_term4 = np.abs( (PyT_v_phi*PyT_v_rho)/lmbd )
PyT_phi_term5 = np.abs( (m_phi*PyT_phi*PyT_rho)/lmbd)
# rho equation
PyT_rho_term1 = np.abs( UnivariateSpline(time, PyT_v_rho).derivative()(time)*PyT_h )
PyT_rho_term2 = np.abs( 3*PyT_h*PyT_v_rho )
PyT_rho_term3 = np.abs((PyT_rho*PyT_v_phi**2)/(2*lmbd**2))
PyT_rho_term4 = np.abs((PyT_v_phi**2)/(2*lmbd))
PyT_rho_term5 = np.abs(m_rho*(PyT_rho - rho_0))

# Correction to the solution
n = np.where(time > 11.5)[0][0]
delta_v_rho = PyT_v_rho[n] - SR_v_rho[n]

n = np.where(time > 11.5)[0][0]
delta_phi = PyT_phi[n] - SR_phi[n]
delta_rho = PyT_rho[n] - SR_rho[n]
delta_v_phi = PyT_v_phi[n] - SR_v_phi[n]



# c_time = time[n:]
# c_phi = SR_phi + delta_phi
# c_rho = SR_rho + delta_rho
# c_v_phi = SR_v_phi + delta_v_phi
# c_v_rho = SR_v_rho + delta_v_rho

c_time = time[n:]
c_phi = PyT_phi - delta_phi
c_rho = PyT_rho - delta_rho
c_v_phi = PyT_v_phi - delta_v_phi
c_v_rho = PyT_v_rho - delta_v_rho

n_star = np.where(time > 13)[0][0]
# print(f'Delta(phi) = {abs( (c_phi[n_star] - SR_phi[n_star])/c_phi[n_star] )}')
# print(f'Delta(v_phi) = {abs( (c_v_phi[n_star] - SR_v_phi[n_star])/c_v_phi[n_star] )}')

# print(f'Delta(rho) = {abs((c_rho[n_star] - SR_rho[n_star])/c_rho[n_star])}')
# print(f'Delta(v_rho) = {abs((c_v_rho[n_star] - SR_v_rho[n_star])/c_v_rho[n_star])}')
####################################### Plots #######################################
# φ SR-comparison
fig1, ax1 = plt.subplots(1,2, figsize=(10,6))
ax1[0].plot(time, PyT_phi, label=r'PyT $\phi$')
ax1[0].plot(time, SR_phi, label=r'SR $\phi$', linestyle='dashed')

# φ̇ comparison
ax1[1].plot(time, PyT_v_phi/PyT_h, label=r'PyT $\phi^\prime$')
ax1[1].plot(time, SR_v_phi, label=r'SR $\phi^\prime$', linestyle='dashed')

# # Inset for φ
# axins0 = inset_axes(ax1[0], width="40%", height="40%", loc="lower left")
# axins0.plot(time, PyT_phi)
# axins0.plot(time, SR_phi, linestyle='dashed')
# axins0.set_xlim(20, 25)
# axins0.set_ylim(12, 13.4)  # Adjust as needed
# axins0.set_ylim()
# axins0.set_xticklabels([])
# axins0.set_yticklabels([])
# mark_inset(ax1[0], axins0, loc1=2, loc2=4, fc="none", ec="0.5")

# Inset for φ̇
# axins1 = inset_axes(ax1[1], width="40%", height="40%", loc="upper left")
# axins1.plot(time, PyT_v_phi)
# axins1.plot(time, SR_v_phi/SR_h, linestyle='dashed')
# axins1.set_xlim(20, 25)
# axins1.set_ylim(-0.8170, -0.8145)  # Adjust as needed
# axins1.set_xticklabels([])
# axins1.set_yticklabels([])
# mark_inset(ax1[1], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

# Labels and title
fig1.suptitle(r'$\phi$ and $\phi^\prime$ solutions')
ax1[0].set_xlabel(r'$N$')
ax1[1].set_xlabel(r'$N$')
ax1[0].legend(loc='upper right')
ax1[1].legend(loc='lower center')

plt.savefig('../Plots/SR_phi_sol.pdf', format='pdf', bbox_inches='tight')
##################################################################################################
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 6))

# ρ comparison
ax2[0].plot(time, PyT_rho, label=r'PyT $\rho$')
ax2[0].plot(time, SR_rho, label=r'SR $\rho$', linestyle='dashed')
# ρ̇ comparison
ax2[1].plot(time, PyT_v_rho/PyT_h, label=r'PyT $\rho^\prime$')
ax2[1].plot(time, SR_v_rho, label=r'SR $\rho^\prime$', linestyle='dashed')

# # Inset for ρ
# axins0 = inset_axes(ax2[0], width="40%", height="40%", loc="lower left")
# axins0.plot(time, PyT_rho)
# axins0.plot(time, SR_rho, linestyle='dashed')
# axins0.set_xlim(20, 25)
# axins0.set_ylim(-6e-7, 2e-7)  # Adjust as needed
# axins0.set_xticklabels([])
# axins0.set_yticklabels([])
# mark_inset(ax2[0], axins0, loc1=2, loc2=4, fc="none", ec="0.5")

# # Inset for ρ̇
# axins1 = inset_axes(ax2[1], width="40%", height="40%", loc="lower right")
# axins1.plot(time, PyT_v_rho)
# axins1.plot(time, SR_v_rho*SR_h, linestyle='dashed')
# axins1.set_xlim(20, 25)
# axins1.set_ylim(-2e-7, 2e-7)  # Adjust as needed
# axins1.set_xticklabels([])
# axins1.set_yticklabels([])
# mark_inset(ax2[1], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

# Labels and title
fig2.suptitle(r'$\rho$ and $\rho^\prime$ solutions')
ax2[0].set_xlabel(r'$N$')
ax2[1].set_xlabel(r'$N$')
ax2[0].legend(loc='lower right')
ax2[1].legend(loc='lower left')

plt.savefig('../Plots/SR_rho_sol.pdf', format='pdf', bbox_inches='tight')
##################################################################################################
fig3 = plt.figure(3, figsize=(10, 6))
plt.plot(time, SR_h, label=r'SR H')
plt.plot(time, PyT_h, label=r'PyT H', linestyle='dashed')
plt.legend()

##################################################################################################
fig4, ax4 = plt.subplots(1, 2, figsize=(10, 6))

# Solution manual
ax4[0].plot(time, SR_3h2, label=r'SR $3H^2$')
ax4[0].plot(time, SR_pot_phi, label=r'SR $V_\text{SR}(\phi_0)$', linestyle='dashed')
ax4[0].plot(time, SR_k_phi, label=r'SR $K(\phi^\prime)$', linestyle='dashed')
ax4[0].plot(time, SR_pot_rho, label=r'SR $W(\rho_o)$')
ax4[0].plot(time, SR_k_rho, label=r'SR $K(\rho^\prime)$', linestyle='dashed')
# Solution 
ax4[1].plot(time, PyT_3h2, label=r'PyT $3H^2$')
ax4[1].plot(time, PyT_pot_phi, label=r'PyT $V_\text{SR}(\phi_0)$', linestyle='dashed')
ax4[1].plot(time, PyT_k_phi/PyT_h**2, label=r'PyT $K(\phi^\prime)$', linestyle='dashed')
ax4[1].plot(time, PyT_pot_rho, label=r'PyT $W(\rho_o)$')
ax4[1].plot(time, PyT_k_rho/PyT_h**2, label=r'PyT $K(\rho^\prime)$', linestyle='dashed')

# # Inset for PyT
# axins0 = inset_axes(ax4[0], width="40%", height="40%", loc="upper right")
# axins0.plot(time, SR_3h2)
# axins0.plot(time, SR_pot_phi, linestyle='dashed')
# axins0.plot(time, SR_k_phi, linestyle='dashed')
# axins0.plot(time, SR_pot_rho)
# axins0.plot(time, SR_k_rho, linestyle='dashed')
# axins0.set_xlim(60, time[-1])
# axins0.set_ylim(-0.5, 1.5)  # Adjust as needed
# axins0.set_xticklabels([])
# axins0.set_yticklabels([])
# mark_inset(ax4[0], axins0, loc1=2, loc2=4, fc="none", ec="0.5")

# # Inset for SR
# axins1 = inset_axes(ax4[1], width="40%", height="40%", loc="upper right")
# axins1.plot(time, PyT_3h2)
# axins1.plot(time, PyT_pot_phi, linestyle='dashed')
# axins1.plot(time, PyT_k_phi, linestyle='dashed')
# axins1.plot(time, PyT_pot_rho, linestyle='dashed')
# axins1.plot(time, PyT_k_rho, linestyle='dashed')
# axins1.set_xlim(60, time[-1])
# axins1.set_ylim(-0.5, 10.)  # Adjust as needed
# axins1.set_xticklabels([])
# axins1.set_yticklabels([])
# mark_inset(ax4[1], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

# Settings
fig4.suptitle(r'$3H^2$ and components')
ax4[0].set_xlabel(r'$N$')
ax4[1].set_xlabel(r'$N$')
ax4[0].set_yscale('log')
ax4[1].set_yscale('log')
ax4[0].legend(loc="center left")
ax4[1].legend(loc="center left")

plt.savefig('../Plots/SR_3h2.pdf', format='pdf', bbox_inches='tight')
##################################################################################################

fig5, ax5 = plt.subplots(1, 2, figsize=(10, 6))

# Solution manual
ax5[0].plot(time, SR_epsilon, label=r'SR $\epsilon$')
ax5[0].plot(time, SR_k_phi, label=r'SR $K(\phi^\prime)$', linestyle='dashed')
ax5[0].plot(time, SR_k_rho, label=r'SR $K(\rho^\prime)$', linestyle='dashed')
# Solution 
ax5[1].plot(time, PyT_epsilon, label=r'PyT $\epsilon$')
ax5[1].plot(time, PyT_k_phi/PyT_h**2, label=r'PyT $K(\phi^\prime)$', linestyle='dashed')
ax5[1].plot(time, PyT_k_rho/PyT_h**2, label=r'PyT $K(\rho^\prime)$', linestyle='dashed')

# Settings
fig5.suptitle(r'$\epsilon$ and components')
ax5[0].set_xlabel(r'$N$')
ax5[1].set_xlabel(r'$N$')
# ax5[0].set_xlim(0, n_max)
ax5[0].set_ylim(-0.1,1.1)
# ax5[1].set_xlim(0, n_max)
# ax5[0].set_yscale('log')
# ax5[1].set_yscale('log')
ax5[0].legend()
ax5[1].legend()
plt.savefig('../Plots/SR_epsilon.pdf', format='pdf', bbox_inches='tight')

##################################################################################################
fig6, ax6 = plt.subplots(1,3,figsize=(10,6))
# alpha
ax6[0].plot(time, PyT_alpha, label=r'PyT $\alpha$')
ax6[0].plot(time, SR_alpha, label=r'SR $\alpha$', linestyle='dashed')
# beta
ax6[1].plot(time, PyT_b, label=r'PyT $b$')
ax6[1].plot(time, SR_b, label=r'SR $b$', linestyle='dashed')
# Lambda
ax6[2].plot(time, np.abs(PyT_lmbd), label=r'PyT $\Lambda$ condition')
ax6[2].plot(time, np.abs(SR_lmbd), label=r'SR $\Lambda$ condition', linestyle='dashed')
# Settings
fig6.suptitle(r"Parameter's evolution")
ax6[0].set_xlabel(r'$N$')
ax6[1].set_xlabel(r'$N$')
ax6[2].set_xlabel(r'$N$')
ax6[0].legend()
ax6[1].legend()
ax6[2].legend()
plt.savefig('../Plots/SR_parameters.pdf', format='pdf', bbox_inches='tight')

# ##################################################################################################

# # SR
# ax7[0].plot(time, PyT_term1, label=r'PyT $\frac{\dot\phi_0 \dot\rho_0}{\lambda}$')
# ax7[0].plot(time, SR_term1, label=r'SR $\frac{\dot\phi_0 \dot\rho_0}{\lambda}$', linestyle='dashed')

# # PyT
# ax7[1].plot(time, PyT_term2, label=r'PyT $V_\text{SR, \phi}\rho_0/\lambda$')
# ax7[1].plot(time, SR_term2, label=r'SR $V_\text{SR, \phi}\rho_0/\lambda$', linestyle='dashed')

# # Settings
# fig7.suptitle(r'$\phi$ Discarded equation terms')
# ax7[0].set_xlabel(r'$N$')
# ax7[1].set_xlabel(r'$N$')
# ax7[0].legend()
# ax7[1].legend()

# ###################################################################################################
# fig8, ax8 = plt.subplots(1,3,figsize=(10,6))

# ax8[0].plot(time, PyT_term3, label=r'PyT $\dot{\rho_0}$')
# ax8[0].plot(time, SR_term3, label=r'SR $\dot{\rho_0}$', linestyle='dashed')

# ax8[1].plot(time, PyT_term4, label=r'PyT $3H\dot\rho_0$')
# ax8[1].plot(time, SR_term4, label=r'SR $3H\dot\rho_0$', linestyle='dashed')

# ax8[2].plot(time, PyT_term5, label=r'PyT $\dot\phi_0^2/2 \lambda$')
# ax8[2].plot(time, SR_term5, label=r'SR $\dot\phi_0^2/2 \lambda$', linestyle='dashed')

# fig8.suptitle(r'$\phi$ Discarded equation terms')
# ax8[0].set_xlabel(r'$N$')
# ax8[1].set_xlabel(r'$N$')
# ax8[2].set_xlabel(r'$N$')
# ax8[0].legend()
# ax8[1].legend()
# ax8[2].legend()

###################################################################################################
fig7, ax7 = plt.subplots(1,2,figsize=(10,6))

# SR phi equations term
ax7[0].plot(time, SR_phi_term2, label=r'SR $3H\dot\phi_0$')
ax7[0].plot(time, SR_phi_term3, label=r"SR $V_\text{SR}'(\phi_0)$",linestyle='dashed')
ax7[0].plot(time, SR_phi_term1, label=r'SR $\ddot{\phi_0}$', marker='o', markersize=6, markevery=400)
ax7[0].plot(time, SR_phi_term4, label=r'SR $\frac{\dot\phi_0 \dot\rho_0}{\Lambda}$', marker='^', markersize=5, markevery=400)
ax7[0].plot(time, SR_phi_term5, label=r"SR $V_\text{SR}'\rho_0/\Lambda$")

# PyT
ax7[1].plot(time, PyT_phi_term2, label=r'PyT $3H\dot\phi_0$')
ax7[1].plot(time, PyT_phi_term3, label=r"PyT $V_\text{SR}'(\phi_0)$", linestyle='dashed')
ax7[1].plot(time, PyT_phi_term1, label=r'PyT $\ddot{\phi_0}$', marker='o', markersize=6, markevery=400)
ax7[1].plot(time, PyT_phi_term4, label=r'PyT $\frac{\dot\phi_0 \dot\rho_0}{\Lambda}$', marker='^', markersize=5, markevery=400)
ax7[1].plot(time, PyT_phi_term5, label=r"PyT $V_\text{SR}'\rho_0/\Lambda$")

# Settings
fig7.suptitle(r'$\phi_0$ Discarded equation terms')
ax7[0].set_xlabel(r'$N$')
ax7[1].set_xlabel(r'$N$')
ax7[0].legend()
ax7[1].legend()
plt.savefig('../Plots/SR_phi_equation.pdf', format='pdf', bbox_inches='tight')

###################################################################################################
fig8, ax8 = plt.subplots(1,2,figsize=(10,6))

# SR phi equations term
ax8[0].plot(time, SR_rho_term1, label=r'SR $\ddot{\rho_0}$')
ax8[0].plot(time, SR_rho_term2, label=r'SR $3H\dot\rho_0$', linestyle='dashed')
ax8[0].plot(time, SR_rho_term3, label=r"SR $(\rho_0\dot\phi_0^2)/(2\Lambda^2)$", linestyle='-.')
ax8[0].plot(time, SR_rho_term4, label=r'SR $\dot\phi_0^2/2 \Lambda$')
ax8[0].plot(time, SR_rho_term5, label=r"SR $W'(\rho_0)$", linestyle='dashed')

# PyT 
ax8[1].plot(time, PyT_rho_term1, label=r'PyT $\ddot{\rho_0}$')
ax8[1].plot(time, PyT_rho_term2, label=r'PyT $3H\dot\rho_0$')
ax8[1].plot(time, PyT_rho_term3, label=r"PyT $(\rho_0\dot\phi_0^2)/(2\Lambda^2)$", linestyle='dashed')
ax8[1].plot(time, PyT_rho_term4, label=r'PyT$\dot\phi_0^2/2 \Lambda$')
ax8[1].plot(time, PyT_rho_term5, label=r"PyT $W'(\rho_0)$", linestyle='dashed')

# Settings
fig8.suptitle(r'$\rho_0$ Discarded equation terms')
ax8[0].set_xlabel(r'$N$')
ax8[1].set_xlabel(r'$N$')
ax8[0].legend()
ax8[1].legend()
plt.savefig('../Plots/SR_rho_equation.pdf', format='pdf', bbox_inches='tight')

###################################################################################################
# fig9, ax9 = plt.subplots(1,2,figsize=(10,6))
# # phi difference
# ax9[0].plot(time, np.abs((PyT_phi - SR_phi)), label=r'$\Delta(\phi_0)$')
# # vel phi difference
# ax9[1].plot(time, np.abs( PyT_v_phi - SR_v_phi), label=r'$\Delta(\dot \phi_0)$')
# # Settings
# fig9.suptitle(r'$\phi_0$ and $\dot \phi_0$ absolute difference')
# ax9[0].set_xlabel(r'$N$')
# ax9[1].set_xlabel(r'$N$')
# ax9[0].legend()
# ax9[1].legend()
# ax9[0].set_yscale('log')
# ax9[1].set_yscale('log')
# plt.savefig('../Plots/SR_phi_difference.pdf', format='pdf', bbox_inches='tight')

###################################################################################################
# fig10, ax10 = plt.subplots(1,2,figsize=(10,6))
# # rho difference
# ax10[0].plot(time, np.abs( (PyT_rho - SR_rho)), label=r'$\Delta(\rho_0)$')
# # vel rho difference
# ax10[1].plot(time, np.abs(PyT_v_rho - SR_v_rho), label=r'$\Delta(\dot \rho_0)$')
# # Settings
# fig10.suptitle(r'$\rho_0$ and $\dot \rho_0$ absolute difference')
# ax10[0].set_xlabel(r'$N$')
# ax10[1].set_xlabel(r'$N$')
# ax10[0].legend()
# ax10[1].legend()
# ax10[0].set_yscale('log')
# ax10[1].set_yscale('log')
# plt.savefig('../Plots/SR_rho_difference.pdf', format='pdf', bbox_inches='tight')
###################################################################################################
fig11, ax11 = plt.subplots(1, 2, figsize=(10,6))

# phi
ax11[0].plot(c_time, c_phi[n:], label=r'Modified PyT $\phi$')
ax11[0].plot(c_time, SR_phi[n:], label=r'SR $\phi$', linestyle='dashed')
# \dot \phi
ax11[1].plot(c_time, c_v_phi[n:], label=r'Modified PyT $\dot \phi$')
ax11[1].plot(c_time, SR_v_phi[n:], label=r'SR $\phi^\prime$', linestyle='dashed')


fig11.suptitle(r'$\phi_0$ and $\dot \phi_0$ small window')
ax11[0].set_xlabel(r'$N$')
ax11[1].set_xlabel(r'$N$')
ax11[0].set_xlim(11.5, 21.5)
ax11[1].set_xlim(11.5, 21.5)
ax11[0].set_ylim(14.0, 15.7)
ax11[1].set_ylim(-0.16, -0.1)
ax11[0].legend()
ax11[1].legend()
plt.savefig('../Plots/SR_phi_reduced.pdf', format='pdf', bbox_inches='tight')

###################################################################################################
fig12, ax12 = plt.subplots(1, 2, figsize=(10,6))

# rho
ax12[0].plot(c_time, c_rho[n:], label=r'Modified PyT $\rho$')
ax12[0].plot(c_time, SR_rho[n:], label=r'SR $\rho$', linestyle='dashed')
# \dot \rho
ax12[1].plot(c_time, c_v_rho[n:], label=r'Modified PyT $\rho^\prime$')
ax12[1].plot(c_time, SR_v_rho[n:], label=r'SR $\rho^\prime$', linestyle='dashed')

fig12.suptitle(r'$\rho_0$ and $\dot \rho_0$ small window')
ax12[0].set_xlabel(r'$N$')
ax12[1].set_xlabel(r'$N$')
ax12[0].set_xlim(11.5, 21.5)
ax12[1].set_xlim(11.5, 21.5)
ax12[0].set_ylim(-0.75e-7, 0.25e-7)
ax12[1].set_ylim(-0.25e-7, 0.25e-7)
ax12[0].legend()
ax12[1].legend()
plt.savefig('../Plots/SR_rho_reduced.pdf', format='pdf', bbox_inches='tight')

###################################################################################################
plt.show()