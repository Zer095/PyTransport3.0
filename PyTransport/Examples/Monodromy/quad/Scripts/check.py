import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

from PyTransport.PyTransPy import PyTransSetup
from PyTransport.PyTransPy import PyTransScripts as PyS

import PyTransAxQ0 as PyT
from helper import *
###############################################################################################################
##################################### Set initial values ######################################################

nF = PyT.nF()
nP = PyT.nP()

# Free parameters/quantities
m_phi = 1.              # Controls the energy scale
phi_0 = 17.             # Controls inflation's duration
lambda_ratio = 10       # should be >> 1
rho_mass_ratio = 409/4  # should be >> 1
alpha = 30              # should be >> 1
beta = 10e-2            # should be << 1

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

####################################### Calculate PyT Background evolution ########################################
Nstart = 0.
Nend = 73.0
t = np.linspace(Nstart, Nend, 10000)
tols = np.array([10**-30,10**-30])
back = PyT.backEvolve(t, initial, params, tols, True)
print(f'Inflation ends at N = {back[-1,0]}')
####################################### Calculate Slow-Roll background evolution #################################

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

# print(f'Indx = {indx}, phi[indx] = {phi[indx]}, phi[indx + 1] = {phi[indx + 1]}')
        
zipped = zip(h, phi, rho, v_phi, v_rho)

SR_sol = np.array([list(time_values) for time_values in zipped])

SR_sol = np.array(SR_sol)
SR_epsilon = Epsilon(back[:indx,0], h[:indx])

# print(f'Epsilon = {SR_epsilon[-100:]}')

try:
    n_max = np.where(SR_epsilon > 1.)[0][0]
except IndexError:
    n_max = indx

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

####################################### Calculate parameters evolution #################################
# Slow roll
SR_alpha = SR_v_phi/(SR_h*f)
SR_b = a/(m_phi*SR_phi*f)
SR_lmbd = SR_v_phi/(lmbd*SR_h*np.sqrt(SR_alpha))

# PyTransport
PyT_alpha = PyT_v_phi/(PyT_h*f)
PyT_b = a/(m_phi*PyT_phi*f) 
PyT_lmbd = PyT_v_phi/(lmbd*PyT_h*np.sqrt(PyT_alpha))

####################################### Calculate discarded equation #################################
# Slow-roll
# phi equation
SR_phi_term1 = UnivariateSpline(time, SR_v_phi).derivative()(time)*SR_h
SR_phi_term2 = 3*SR_h*SR_v_phi
SR_phi_term3 = m_phi*SR_phi
SR_phi_term4 = (SR_v_phi*SR_v_rho)/lmbd
SR_phi_term5 = (m_phi*SR_phi*SR_rho)/lmbd
# rho equation
SR_rho_term1 = UnivariateSpline(time, SR_v_rho).derivative()(time)*SR_h
SR_rho_term2 = 3*SR_h*SR_v_rho
SR_rho_term3 = (SR_rho*SR_v_phi**2)/(2*lmbd**2)
SR_rho_term4 = (SR_v_phi**2)/(2*lmbd)
SR_rho_term5 = m_rho*(SR_rho - rho_0)

# PyTransport
# phi equation
PyT_phi_term1 = UnivariateSpline(time, PyT_v_phi).derivative()(time)*PyT_h
PyT_phi_term2 = 3*PyT_h*PyT_v_phi
PyT_phi_term3 = m_phi*PyT_phi
PyT_phi_term4 = (PyT_v_phi*PyT_v_rho)/lmbd
PyT_phi_term5 = (m_phi*PyT_phi*PyT_rho)/lmbd
# rho equation
PyT_rho_term1 = UnivariateSpline(time, PyT_v_rho).derivative()(time)*PyT_h
PyT_rho_term2 = 3*PyT_h*PyT_v_rho
PyT_rho_term3 = (PyT_rho*PyT_v_phi**2)/(2*lmbd**2)
PyT_rho_term4 = (PyT_v_phi**2)/(2*lmbd)
PyT_rho_term5 = m_rho*(PyT_rho - rho_0)

# Correction to the solution
n = np.where(time > 7)[0][0]
delta_v_rho = PyT_v_rho[n] - SR_v_rho[n]

n = np.where(time > 3)[0][0]
delta_phi = PyT_phi[n] - SR_phi[n]
delta_rho = PyT_rho[n] - SR_rho[n]
delta_v_phi = PyT_v_phi[n] - SR_v_phi[n]

c_time = time[n:]
c_phi = PyT_phi - delta_phi
c_rho = PyT_rho - delta_rho
c_v_phi = PyT_v_phi - delta_v_phi
c_v_rho = PyT_v_rho - delta_v_rho

n_star = np.where(time > 13)[0][0]
####################################### Plots #######################################
# φ SR-comparison
fig1, ax1 = plt.subplots(1,2, figsize=(10,6))
ax1[0].plot(time, PyT_phi, label=r'PyT $\phi$')
ax1[0].plot(time, SR_phi, label=r'SR $\phi$', linestyle='dashed')

# φ̇ comparison
ax1[1].plot(time, PyT_v_phi/PyT_h, label=r'PyT $\dot \phi$')
ax1[1].plot(time, SR_v_phi, label=r'SR $\dot \phi$', linestyle='dashed')

# Labels and title
fig1.suptitle(r'$\phi$ and $\dot \phi$ solutions')
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
ax2[1].plot(time, PyT_v_rho/PyT_h, label=r'PyT $\dot \rho$')
ax2[1].plot(time, SR_v_rho, label=r'SR $\dot \rho$', linestyle='dashed')

# Labels and title
fig2.suptitle(r'$\rho$ and $\dot \rho$ solutions')
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

fig5, ax5 = plt.subplots(1, 2, figsize=(10, 6))

# Solution manual
ax5[0].plot(time, SR_epsilon, label=r'SR $\epsilon$')
ax5[0].plot(time, SR_k_phi, label=r'SR $K(\phi)/H^2$', linestyle='dashed')
ax5[0].plot(time, SR_k_rho, label=r'SR $K(\rho)/H^2$', linestyle='dashed')
# Solution 
ax5[1].plot(time, PyT_epsilon, label=r'PyT $\epsilon$')
ax5[1].plot(time, PyT_k_phi/PyT_h**2, label=r'PyT $K(\phi)/H^2$', linestyle='dashed')
ax5[1].plot(time, PyT_k_rho/PyT_h**2, label=r'PyT $K(\rho)/H^2$', linestyle='dashed')

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
plt.show()