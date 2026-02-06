# This script is part of the analysis of the Axion Monodromy Inflationary model (2412.05762, 2505.19066)
# In this script we analize the slow-roll part of the solution
# Import packages

import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

from PyTransport.PyTransPy import PyTransSetup
from PyTransport.PyTransPy import PyTransScripts as PyS

import PyTransAxQ as PyT
from helper import *
#plt.rcParams['text.usetex'] = True

##################################### Set initial values ######################################################
nF = PyT.nF()
nP = PyT.nP()

# Free parameters/quantities
m_phi = 1.
phi_0 = 17.
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

print(f'N index = {n_max}, N = {back[n_max,0]}')
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

# Additional quantities
omega = (vel_phi/f)*h

# phi solution
vel_phi_pert = (3*beta/alpha)*vel_phi*np.cos( phi/f )
phi_pert = -(3*beta*f/alpha)*np.sin( phi/f )

# rho solution
B = (-a/lmbd)/np.sqrt( (m_rho - (omega)**2)**2 + 9*(h**2)*(omega)**2 ) # amplitude
delta = np.arcsin((m_rho - (omega)**2)/np.sqrt( (m_rho - (omega)**2)**2 + 9*(h**2)*(omega)**2) ) #phase
rho_pert = B*np.cos( phi/f + delta )
vel_rho_pert = B*(omega/h)*np.sin(phi/f + delta)
pyt = - (phi[0]/f + delta[0])
rho_pert = B*np.cos(phi/f + delta)
pyt =  - (phi[0]/f + delta[0]) + np.pi/2
###############################################################################################################
# Full solitions
FA_vel_phi = vel_phi + vel_phi_pert
FA_vel_phi += (PyT_v_phi[0]/PyT_h[0]) - FA_vel_phi[0] 
FA_phi = phi + phi_pert
FA_vel_rho = vel_rho + vel_rho_pert
FA_rho = rho + rho_pert

# Corrected solutions
N_star = back[n_max,0] - 60
n_star = np.where(back[:,0] > N_star)[0][0]
print(f'N_* = {N_star}, index_* = {n_star}')
delta_phi = back[n_star,1] - FA_phi[n_star]
delta_rho = back[n_star,2] - FA_rho[n_star]
delta_v_phi = back[n_star, 3]/PyT_h[n_star] - FA_vel_phi[n_star]
delta_v_rho = back[n_star, 4]/PyT_h[n_star] - FA_vel_rho[n_star]

c_time = back[n_star:n_max,0]
c_phi = FA_phi + delta_phi
c_rho = FA_rho + delta_rho
c_v_phi = FA_vel_phi + delta_v_phi
c_v_rho = FA_vel_rho + delta_v_rho

c_rho = transform( [PyT_rho[:n_max], PyT_v_rho[:n_max]/PyT_h[:n_max]], [FA_rho, FA_vel_rho], omega/h, B, n_star)

###############################################################################################################
# Power-spectrum
n_points = 1000
d = 10/n_points
sol = zip(time, FA_phi, FA_rho, FA_vel_phi, FA_vel_rho)
FA_sol = np.array([list(time_values) for time_values in sol])

Nout = []
Kout = []
Pz = []
# Compute the analytical power-spectrum and write it on file
# file = open('../Data/pz.txt', 'w')
# for i in range(n_points):
#     Nexit = N_star + i*d
#     k = PyS.kexitN(Nexit, FA_sol, params, PyT)
#     H_k = UnivariateSpline(time, h)(Nexit)
#     Eps_k = UnivariateSpline(time, SR_epsilon[:n_max])(Nexit)
#     # B_k = UnivariateSpline(time, B)(Nexit)
#     # print(k, H_k, Eps_k, alpha, lmbd, B_k)
#     pz = P_z(k, H_k, Eps_k, alpha, lmbd, beta)
#     fact = (k**3.)/(2.*np.pi**2.)
#     print(f'\n i = {i}, N = {Nexit}, k = {k}, Pz = {pz}')
#     file.write(f'{i}, {Nexit}, {k}, {pz}\n')
#     Nout.append(Nexit)
#     Kout.append(k)
#     Pz.append(fact*pz)
# file.close()

# Read on file
text = np.loadtxt('../Data/pz.txt',delimiter=',')
for line in text:
    Nout.append(line[1])
    Kout.append(line[2])
    Pz.append(line[3])

# Convert to array
FA_N = np.array(Nout)
FA_k = np.array(Kout)
FA_Pz = np.array(Pz)

# Rescale power-spectrum
eps_k = UnivariateSpline(time, SR_epsilon[:n_max], s=0)(FA_N)
FA_fact = (FA_k**3)/(4*(np.pi**2)*eps_k)
# FA_fact = (FA_k**3)/(2*(np.pi**2))
FA_Pz = FA_fact*FA_Pz


# Load Data
file1 = '../Data/PyT_pz1.csv' # Power-spectrum without the axion term
file2 = '../Data/PyT_pz.csv'  # Power-spectrum with the axion term

# Load first file
NA_N = []
NA_k = []
NA_pz = []

with open(file1, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for line in csvreader:
        NA_N.append(float(line[0]))
        NA_k.append(float(line[1]))
        NA_pz.append(float(line[2]))

NA_N = np.array(NA_N)
NA_k = np.array(NA_k)
NA_pz = np.array(NA_pz)

# # PyT Power-spectrum
# NB = 4.
# tols = np.array([10**-12,10**-12])
# zzOut, times = PyS.pSpectra(Kout, back, params, NB, tols,PyT)

# # Write on file
# with open('../Data/Pyt_pz.txt', 'a') as f:
#     for i in range(200):
#         f.write(f'{i}, {Nout[i]}, {Kout[i]}, {zzOut[i]}\n')

# Load PyT Power-spectrum
PyT_N = []
PyT_Pz = []
with open('../Data/PyT_pz.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for line in csvreader:
        PyT_N.append(float(line[0]))
        PyT_Pz.append(float(line[2]))

# Normalize
# SR_Pz = ((np.array(Kout)**3)/(2*np.pi**2))*np.array(Pz)
# PyT_Pz = (np.array(Kout)**3)/(2*np.pi**2)*np.array(zzOut)

###############################################################################################################
# PyT Evolution of one mode

NExit = N_star
NB = 4.0
k = PyS.kexitN(NExit, back, params, PyT)
Nstart, backExitMinus = PyS.ICs(NB, k, back, params, PyT) 
tsig = np.linspace(Nstart, back[-1,0], 1000)
tols = np.array([10**-17, 10**-17])

TwoPt = PyT.sigEvolve(tsig, k, backExitMinus, params, tols, True)
print(TwoPt[:, 1 + 1 + 2*nF:].shape)
fact = (k**3.)/(2.*np.pi**2.)
pz = fact*TwoPt[:, 1]

######################## Plots #########################################

phi_str = r"$\phi$"
rho_str = r"$\rho$"
pi_phi_str = r"$\pi_{\phi}$"
pi_rho_str = r"$\pi_{\rho}$"

# Full set of all 16 2-point correlation functions
labels2 = [
    # Field-Field Correlators (4)
    r"$\langle$" + phi_str + phi_str + r"$\rangle$",  # <phi phi>
    r"$\langle$" + phi_str + rho_str + r"$\rangle$",  # <phi rho>
    r"$\langle$" + rho_str + phi_str + r"$\rangle$",  # <rho phi>
    r"$\langle$" + rho_str + rho_str + r"$\rangle$",  # <rho rho>
    
    # Field-Momentum Correlators (4)
    r"$\langle$" + phi_str + pi_phi_str + r"$\rangle$",  # <phi pi_phi>
    r"$\langle$" + phi_str + pi_rho_str + r"$\rangle$",  # <phi pi_rho>
    r"$\langle$" + rho_str + pi_phi_str + r"$\rangle$",  # <rho pi_phi>
    r"$\langle$" + rho_str + pi_rho_str + r"$\rangle$",  # <rho pi_rho>

    # Momentum-Field Correlators (4)
    r"$\langle$" + pi_phi_str + phi_str + r"$\rangle$",  # <pi_phi phi>
    r"$\langle$" + pi_phi_str + rho_str + r"$\rangle$",  # <pi_phi rho>
    r"$\langle$" + pi_rho_str + phi_str + r"$\rangle$",  # <pi_rho phi>
    r"$\langle$" + pi_rho_str + rho_str + r"$\rangle$",  # <pi_rho rho>
    
    # Momentum-Momentum Correlators (4)
    r"$\langle$" + pi_phi_str + pi_phi_str + r"$\rangle$",  # <pi_phi pi_phi>
    r"$\langle$" + pi_phi_str + pi_rho_str + r"$\rangle$",  # <pi_phi pi_rho>
    r"$\langle$" + pi_rho_str + pi_phi_str + r"$\rangle$",  # <pi_rho pi_phi>
    r"$\langle$" + pi_rho_str + pi_rho_str + r"$\rangle$",  # <pi_rho pi_rho>
]



# φ SR-comparison
fig1, ax1 = plt.subplots(1,2, figsize=(10,6))
ax1[0].plot(time, PyT_phi, label=r'PyT $\phi$')
ax1[0].plot(time, FA_phi, label=r'FA Slow-roll $\phi$', linestyle='dashed')

# φ̇ comparison
ax1[1].plot(time, PyT_v_phi/PyT_h, label=r'PyT $\phi^\prime$')
ax1[1].plot(time, FA_vel_phi, label=r'FA Slow-roll $\phi^\prime$', linestyle='dashed')

# Inset for φ
axins0 = inset_axes(ax1[0], width="40%", height="40%", loc="lower left")
axins0.plot(time, PyT_phi)
axins0.plot(time, FA_phi, linestyle='dashed')
axins0.set_xlim(10, 11)
axins0.set_ylim(15.8, 15.6)  # Adjust as needed
axins0.set_ylim()
axins0.set_xticklabels([])
axins0.set_yticklabels([])
mark_inset(ax1[0], axins0, loc1=2, loc2=4, fc="none", ec="0.5")

# Inset for φ̇
axins1 = inset_axes(ax1[1], width="40%", height="40%", loc="lower left")
axins1.plot(time, PyT_v_phi/PyT_h)
axins1.plot(time, FA_vel_phi, linestyle='dashed')
axins1.set_xlim(10, 11)
axins1.set_ylim(-0.130, -0.125)  # Adjust as needed
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
ax2[1].plot(time, PyT_v_rho/PyT_h, label=r'PyT $\rho^\prime$')
ax2[1].plot(time, FA_vel_rho, label=r'FA Slow-roll $\rho^\prime$', linestyle='dashed')

# Inset for ρ
axins0 = inset_axes(ax2[0], width="40%", height="40%", loc="lower left")
axins0.plot(time, PyT_rho)
axins0.plot(time, FA_rho, linestyle='dashed')
axins0.set_xlim(5, 6)
axins0.set_ylim(-1.5e-7, 1.5e-7)  # Adjust as needed
axins0.set_xticklabels([])
axins0.set_yticklabels([])
mark_inset(ax2[0], axins0, loc1=2, loc2=4, fc="none", ec="0.5")

# Inset for ρ̇
axins1 = inset_axes(ax2[1], width="40%", height="40%", loc="lower right")
axins1.plot(time, PyT_v_rho/PyT_h)
axins1.plot(time, FA_vel_rho, linestyle='dashed')
axins1.set_xlim(10, 11)
axins1.set_ylim(-4e-6, 4e-6)  # Adjust as needed
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
fig11, ax11 = plt.subplots(1, 2, figsize=(10,6))

# phi
ax11[0].plot(c_time, back[n_star:n_max,1], label=r'PyT $\phi$')
ax11[0].plot(c_time, c_phi[n_star:], label=r'Corrected FA $\phi$', linestyle='dashed')
# \dot \phi
ax11[1].plot(c_time, back[n_star:n_max,3]/PyT_h[n_star:], label=r'PyT $\dot \phi$')
ax11[1].plot(c_time, c_v_phi[n_star:], label=r'Corrected FA $\dot \phi$', linestyle='dashed')

fig11.suptitle(r'FA $\phi_0$ and $\dot \phi_0$ small window')
ax11[0].set_xlabel(r'$N$')
ax11[1].set_xlabel(r'$N$')
ax11[0].set_xlim(N_star, N_star + 10)
ax11[1].set_xlim(N_star, N_star + 10)
ax11[0].set_ylim(14.0, 15.75)
ax11[1].set_ylim(-0.14, -0.12)
ax11[0].legend()
ax11[1].legend()
plt.savefig('../Plots/PERT_phi_reduced.pdf', format='pdf', bbox_inches='tight')
# ###################################################################################################
fig12, ax12 = plt.subplots(1, 2, figsize=(10,6))

# phi
ax12[0].plot(c_time, back[n_star:n_max,2], label=r'PyT $\rho$')
ax12[0].plot(c_time, c_rho[n_star:], label=r'Corrected FA $\rho$', linestyle='dashed')
# \dot \phi
ax12[1].plot(c_time, back[n_star:n_max,4]/PyT_h[n_star:], label=r'PyT $\dot \rho$')
ax12[1].plot(c_time, c_v_rho[n_star:], label=r'Corrected FA $\dot \rho$', linestyle='dashed')

fig12.suptitle(r'FA $\rho_0$ and $\dot \rho_0$ small window')
ax12[0].set_xlabel(r'$N$')
ax12[1].set_xlabel(r'$N$')
ax12[0].set_xlim(N_star, N_star + 10)
ax12[1].set_xlim(N_star, N_star + 10)
ax12[0].set_ylim(-1.5e-7, 1.5e-7)
ax12[1].set_ylim(-0.00003, 0.00003)
ax12[0].legend()
ax12[1].legend()
plt.savefig('../Plots/PERT_rho_reduced.pdf', format='pdf', bbox_inches='tight')
###################################################################################################
fig3 = plt.figure(figsize=(10,6))
plt.plot(FA_N, FA_Pz, label='Analytical')
plt.yscale('log')
plt.legend()
plt.savefig('../Plots/FA_pz.pdf', format='pdf', bbox_inches='tight')

###################################################################################################
fig4 = plt.figure(figsize=(10,6))
plt.plot(PyT_N, PyT_Pz, label='PyT With Axion')
plt.plot(FA_N, FA_Pz, label='Analytical')
plt.plot(NA_N, NA_pz, label='PyT Without Axion')
plt.yscale('log')
plt.legend()
plt.savefig('../Plots/PyT_pz.pdf', format='pdf', bbox_inches='tight')

###################################################################################################
fig5 = plt.figure(figsize=(10, 6))
for i in range(2*nF*2*nF):
    plt.plot(tsig, np.abs(TwoPt[:, 2 + 2*nF + i]), label=labels2[i])

plt.yscale('log')
plt.vlines(x=N_star, ymin=10**-30, ymax=10**-4, linestyles='dashed', color='gray')
#plt.legend()
plt.xlim(tsig[0], tsig[-1])
plt.savefig('../Plots/PyT_2pt.pdf', format='pdf', bbox_inches='tight')
###################################################################################################
fig6 = plt.figure(figsize=(10, 6))
plt.plot(tsig, np.abs(pz))
plt.yscale('log')
plt.legend()
plt.savefig('../Plots/PyT_pz_evo.pdf', format='pdf', bbox_inches='tight')
###################################################################################################
plt.show()