import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

from PyTransport.PyTransPy import PyTransSetup
from PyTransport.PyTransPy import PyTransScripts as PyS

import PyTransAMQ as PyT

##################################### Set initial values ######################################################

nF = PyT.nF()
nP = PyT.nP()

# Free parameters/quantities
m_phi = 1.
phi_0 = 15.
lambda_ratio = 10       # should be >> 1
rho_mass_ratio = 409/4  # should be >> 1
alpha = 30              # should be >> 1
beta = 10e-2            # should be << 1


# Parameters derived by this quantities
f = (-2)/(alpha*phi_0)
a = -(beta/alpha)*(2/m_phi)
lmbd = (lambda_ratio/np.sqrt(alpha))*(2/phi_0)
m_rho = rho_mass_ratio*0.5*m_phi*phi_0**2
rho_0 = -1/(3*m_phi*m_rho*lmbd)
# rho_0 = 0
# Set parameters potential
params = np.zeros(nP)
params[0] = lmbd
params[1] = m_phi
params[2] = a
params[3] = f
params[4] = m_rho
params[5] = rho_0

fields = np.array([phi_0, 0.])

V = PyT.V(fields,params) # calculate potential from some initial conditions
dV=PyT.dV(fields,params) # calculate derivatives of potential

vel = np.array([-dV[0]/np.sqrt(3*V), 0.])
initial = np.concatenate((fields, vel)) # sets an array containing field values and there derivative in cosmic time 
                                                      # (set using the slow roll equation)

####################################### Calculate the Background evolution ########################################
Nstart = 0.
Nend = 60.0
t = np.linspace(Nstart, Nend, 100000)
tols = np.array([10**-30,10**-30])
back = PyT.backEvolve(t, initial, params, tols, True)
print(f'Inflation ends at N = {back[-1,0]}')

####################################### Helper functions ########################################

# Potential
def Pot(x,y, m_phi, m_rho, rho_0):
    v_phi = 0.5*m_phi*x**2
    v_rho = 0.5*m_rho*(y - rho_0)**2
    return v_phi + v_rho

# Derivative of potential
def DPot(x,y, m_phi, m_rho, rho_0):
    dpot_phi = m_phi*x
    dpot_rho = m_rho*(y - rho_0)
    return dpot_phi, dpot_rho

# Kinetic term
def Kin(x,y, rho, lmbd):
    k_phi = np.exp(rho/lmbd)*(x**2)
    k_rho = y**2
    return 0.5*k_phi + 0.5*k_rho

# H
def H(x, x1, y, y1, m_phi, m_rho, rho_0):
    V = Pot(x, y, m_phi, m_rho, rho_0)
    K =  Kin(x1, y1, y, lmbd)

    return np.sqrt( V/( 3 - K ) )

def PyT_H(back, params):
    H = []
    for i in range(np.size(back[:,0])):
        Hv = PyT.H(back[i,1:], params)
        H.append(Hv)
    H = np.array(H)
    return H

def Epsilon(N,H):
    H_func = UnivariateSpline(N, H, s=0)
    H_prime = H_func.derivative()
    eps = -H_prime(N)/H
    return eps

def Eta(N,Epsilon):
    Eps_func = UnivariateSpline(N,Epsilon,s=0)
    Eps_prime = Eps_func.derivative()
    eta = Eps_prime(N)/Epsilon
    return eta
####################################### Solve analytical equations #################################################

def System(N, vars, lmbd, m_phi, m_rho, rho_0):

    # Unpack variables
    h, x, x1, y, y1 = vars

    # Kinetick energy
    K = Kin(x1, y1, y, lmbd)
    # Gradient of potential
    dv, dw = DPot(x, y, m_phi, m_rho, rho_0)

    # Evolution equation for H
    dh = -1*h*K
    # Ev. equation for phi
    dx = x1
    dx1 = -3*x1 - (dh/h)*x1 - dv/(h**2) 
    # Ev. equation for rho
    dy = y1
    dy1 = -3*y1 - (dh/h)*y1 - dw/(h**2) + (1./(2*lmbd))*(x1**2) 

    return [dh, dx, dx1, dy, dy1]

def System1(N, vars, lmbd, m_phi, m_rho, rho_0):

    # Unpack variables
    h, x, x1, y, y1 = vars
    # Kinetick energy
    K = Kin(x1, y1, y, lmbd)
    # Gradient of potential
    dv, dw = DPot(x, y, m_phi, m_rho, rho_0)

    # Evolution equation for H
    dh = -1*h*K
    # Ev. equation for phi
    dx = x1
    dx1 = -3*x1 - (dh/h)*x1 - dv/(h**2) - (1/lmbd)*x1*y1
    # Ev. equation for rho
    dy = y1
    dy1 = -3*y1 - (dh/h)*y1 - dw/(h**2) + (1./(2*lmbd))*(x1**2) 

    return [dh, dx, dx1, dy, dy1]

# Compute the intial condition
k = Kin(vel[0], 0, 0, lmbd)
dv, dw = DPot(phi_0, 0, m_phi, m_rho, rho_0)
v = Pot(phi_0, 0., m_phi, m_rho, rho_0)
v_phi = -dv/np.sqrt(3*v)

# find H(N)
h = H(phi_0, vel[0]/(np.sqrt((v + k))/3), 0., 0., m_phi, m_rho, rho_0)
# Correct V_phi(N)
v_phi = vel[0]/h
# Correct H(N)
h = H(phi_0, v_phi, 0., 0., m_phi, m_rho, rho_0)
# Correct V_phi
v_phi = vel[0]/h
# Initialize solver 
ics = [h, phi_0, v_phi, 0., 0.] # Initial conditions
t_span = (Nstart, back[-1,0])   # Start time, end time
t_eval = np.linspace(Nstart, back[-1,0], len(back[:,0]))    # time array
# Solve equations
sol = solve_ivp(System, t_span, ics, t_eval=back[:,0], args=(lmbd, m_phi, m_rho, rho_0) )
sol1 = solve_ivp(System1, t_span, ics, t_eval=back[:,0], args=(lmbd, m_phi, m_rho, rho_0) )
####################################### Compute background quantities with SR solutions #################################################

# 1) Compute 3H^2 and it's components
# PyTransport solution
PyT_3H2 = 3*PyT_H(back,params)**2
PyT_K_phi = 0.5*(np.exp(back[:,2]/lmbd))*(back[:,3]/PyT_H(back,params))**2
PyT_K_rho = 0.5*(back[:,4]/PyT_H(back,params))**2
PyT_V_phi = 0.5*m_phi*back[:,1]**2
PyT_V_rho = 0.5*m_rho*(back[:,2]-rho_0)**2
# SR solution
SR_3H2 = 3*sol.y[0]**2
SR_K_phi = 0.5*np.exp(sol.y[3]/lmbd)*sol.y[2]**2
SR_K_rho = 0.5*sol.y[4]**2
SR_V_phi = 0.5*m_phi*sol.y[1]**2
SR_V_rho = 0.5*m_rho*(sol.y[3]-rho_0)**2

# 2) Compute \epsilon
PyT_eps = Epsilon(back[:,0], PyT_H(back,params))
SR_eps = Epsilon(sol.t, sol.y[0])

# 3) Compute \eta
PyT_eta = Eta(back[:,0], PyT_eps)
SR_eta = Eta(sol.t, SR_eps)

####################################### Compute Axion-perturbations to the SR solution #################################################
# Unpack SR solution
N = sol.t
h_sr = sol.y[0]
phi_sr = sol.y[1]
phi1_sr = sol.y[2]
rho_sr = sol.y[3]
rho1_sr = sol.y[4]

# Phi perturbed solution:
phi1_pert = (3*beta/alpha)*phi1_sr*np.cos(phi_sr/f)
phi_pert = -(3*beta*f/alpha)*np.sin(phi_sr/f-phi1_sr[0]/f)

# Rho solution coefficients: 
omega_f = (phi1_sr*h_sr)/f
omega_s = 3*h_sr
omega_n = np.sqrt(m_rho)
a0 = a/lmbd
B = a0/np.sqrt( (omega_n**2 - omega_f**2)**2 + (omega_f**2)*(omega_s**2) )
delta = np.arctan( (omega_s*omega_f)/(omega_n**2-omega_f**2) )
# Rho perturbed solution:
rho1_pert = (-B)*(omega_f)*np.cos( (phi_sr/f) + delta - ((phi_sr[0]/f) + delta[0] - np.arccos(rho1_sr[0]))) 
rho_pert = B*np.sin( (phi_sr/f) + delta - ((phi_sr[0]/f) + delta[0] - np.arcsin(rho_sr[0])) ) 

# Define full analytical solutions:
phi1_full = phi1_sr + (phi1_pert/h_sr)
phi_full = phi_sr + phi_pert
rho1_full = rho1_sr + (rho1_pert/h_sr)
rho_full = rho_sr + rho_pert

####################################### Compute background quantities with FA solutions #################################################

# 1) Compute 3H^2 and it's components
# SR solution

FA_K_phi = 0.5*np.exp(rho_full/lmbd)*(phi1_full**2)
FA_K_rho = 0.5*(rho1_full**2)
FA_V_phi = 0.5*m_phi*(phi_full**2) + a*np.cos(phi_full/f)
FA_V_rho = 0.5*m_rho*(rho_full-rho_0)**2
FA_3H2 = FA_K_phi + FA_K_rho + FA_V_phi + FA_V_rho

FA_H = np.sqrt(FA_3H2/3)
# 2) Compute \epsilon
FA_eps = Epsilon(sol.t, FA_H)

# 3) Compute \eta
FA_eta = Eta(sol.t, FA_eps)

####################################### Check the validity of the parameters #################################################

# Slow-roll parameter
alpha_sr = phi1_sr/(f)
beta_sr = a/(m_phi*phi_sr*f)
lambda_sr = phi1_sr/(np.sqrt(alpha_sr)*lmbd)

h_PyT = PyT_H(back,params)
phi1_PyT = back[:,3]
phi_PyT = back[:,1]

alpha_PyT = phi1_PyT/(h_PyT*f)
beta_PyT = a/(m_phi*phi_PyT*f)
lambda_PyT = phi1_PyT/(np.sqrt(alpha_PyT)*h_PyT*lmbd)


############################################################ Plots ############################################################

fig1, ax1 = plt.subplots(1, 2, figsize=(10, 6))

# φ comparison
ax1[0].plot(back[:, 0], back[:, 1], label=r'PyT $\phi$')
ax1[0].plot(sol.t, sol.y[1], label=r'Slow-roll $\phi$', linestyle='dashed')

# φ̇ comparison
ax1[1].plot(back[:, 0], back[:, 3], label=r'PyT $\dot \phi$')
ax1[1].plot(sol.t, np.multiply(sol.y[2], sol.y[0]), label=r'Slow-roll $\dot \phi$', linestyle='dashed')

# Inset for φ
axins0 = inset_axes(ax1[0], width="40%", height="40%", loc="lower left")
axins0.plot(back[:, 0], back[:, 1])
axins0.plot(sol.t, sol.y[1], linestyle='dashed')
axins0.set_xlim(0, 5)
axins0.set_ylim(14.5, 15)  # Adjust as needed
axins0.set_xticklabels([])
axins0.set_yticklabels([])
mark_inset(ax1[0], axins0, loc1=2, loc2=4, fc="none", ec="0.5")

# Inset for φ̇
axins1 = inset_axes(ax1[1], width="40%", height="40%", loc="upper left")
axins1.plot(back[:, 0], back[:, 3])
axins1.plot(sol.t, np.multiply(sol.y[2], sol.y[0]), linestyle='dashed')
axins1.set_xlim(0, 5)
axins1.set_ylim(-0.88, -0.80)  # Adjust as needed
axins1.set_xticklabels([])
axins1.set_yticklabels([])
mark_inset(ax1[1], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

# Labels and title
fig1.suptitle(r'$\phi$ and $\dot \phi$ confront solutions')
ax1[0].set_xlabel(r'$N$')
ax1[1].set_xlabel(r'$N$')
ax1[0].legend(loc='upper right')
ax1[1].legend(loc='lower right')

plt.savefig('Plots/SR_phi_sol.pdf', format='pdf', bbox_inches='tight')
##################################################################################################
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 6))

# ρ comparison
ax2[0].plot(back[:, 0], back[:, 2], label=r'PyT $\rho$')
ax2[0].plot(sol.t, sol.y[3], label=r'Slow-roll $\rho$', linestyle='dashed')

# ρ̇ comparison
ax2[1].plot(back[:, 0], back[:, 4], label=r'PyT $\dot \rho$')
ax2[1].plot(sol.t, np.multiply(sol.y[4], sol.y[0]), label=r'Slow-roll $\dot \rho$', linestyle='dashed')

# Inset for ρ
axins0 = inset_axes(ax2[0], width="40%", height="40%", loc="upper right")
axins0.plot(back[:, 0], back[:, 2])
axins0.plot(sol.t, sol.y[3], linestyle='dashed')
axins0.set_xlim(0, 5)
axins0.set_ylim(-6e-6, 3e-5)  # Adjust as needed
axins0.set_xticklabels([])
axins0.set_yticklabels([])
mark_inset(ax2[0], axins0, loc1=2, loc2=4, fc="none", ec="0.5")

# Inset for ρ̇
axins1 = inset_axes(ax2[1], width="40%", height="40%", loc="upper center")
axins1.plot(back[:, 0], back[:, 4])
axins1.plot(sol.t, np.multiply(sol.y[4], sol.y[0]), linestyle='dashed')
axins1.set_xlim(0, 5)
axins1.set_ylim(-2e-6, 1e-5)  # Adjust as needed
axins1.set_xticklabels([])
axins1.set_yticklabels([])
mark_inset(ax2[1], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

# Labels and title
fig2.suptitle(r'$\rho$ and $\dot \rho$ confront solutions')
ax2[0].set_xlabel(r'$N$')
ax2[1].set_xlabel(r'$N$')
ax2[0].legend(loc='lower left')
ax2[1].legend(loc='lower right')

plt.savefig('Plots/SR_rho_sol.pdf', format='pdf', bbox_inches='tight')

##################################################################################################
# Compare(3H2 components )
fig3, ax3 = plt.subplots(1, 2, figsize=(10, 6))

# PyT
ax3[0].plot(back[:, 0], PyT_3H2,   label=r'PyT $3H^2$')
ax3[0].plot(back[:, 0], PyT_V_phi, label=r'PyT $V\left(\phi\right)$')
ax3[0].plot(back[:, 0], PyT_V_rho, label=r'PyT $V\left(\rho\right)$')
ax3[0].plot(back[:, 0], PyT_K_phi, label=r'PyT $K\left(\phi\right)$', linestyle='dashed')
ax3[0].plot(back[:, 0], PyT_K_rho, label=r'PyT $K\left(\rho\right)$', linestyle='dashed')

# SR
ax3[1].plot(sol.t, SR_3H2,   label=r'SR $3H^2$')
ax3[1].plot(sol.t, SR_V_phi, label=r'SR $V\left(\phi\right)$')
ax3[1].plot(sol.t, SR_V_rho, label=r'SR $V\left(\rho\right)$')
ax3[1].plot(sol.t, SR_K_phi, label=r'SR $K\left(\phi\right)$', linestyle='dashed')
ax3[1].plot(sol.t, SR_K_rho, label=r'SR $K\left(\rho\right)$', linestyle='dashed')

# Inset for PyT
axins0 = inset_axes(ax3[0], width="40%", height="40%", loc="upper right")
axins0.plot(back[:, 0], PyT_3H2)
axins0.plot(back[:, 0], PyT_V_phi)
axins0.plot(back[:, 0], PyT_V_rho)
axins0.plot(back[:, 0], PyT_K_phi, linestyle='dashed')
axins0.plot(back[:, 0], PyT_K_rho, linestyle='dashed')
axins0.set_xlim(56, 57)
axins0.set_ylim(-0.5, 2)  # Adjust as needed
axins0.set_xticklabels([])
axins0.set_yticklabels([])
mark_inset(ax3[0], axins0, loc1=2, loc2=4, fc="none", ec="0.5")

# Inset for SR
axins1 = inset_axes(ax3[1], width="40%", height="40%", loc="upper right")
axins1.plot(sol.t, SR_3H2)
axins1.plot(sol.t, SR_V_phi)
axins1.plot(sol.t, SR_V_rho)
axins1.plot(sol.t, SR_K_phi, linestyle='dashed')
axins1.plot(sol.t, SR_K_rho, linestyle='dashed')
axins1.set_xlim(56, 57)
axins1.set_ylim(-0.5, 2)  # Adjust as needed
axins1.set_xticklabels([])
axins1.set_yticklabels([])
mark_inset(ax3[1], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

# Labels and title
fig3.suptitle(r'$3H^2$ and components')
ax3[0].set_xlabel(r'$N$')
ax3[1].set_xlabel(r'$N$')
ax3[0].legend()
ax3[1].legend()

plt.savefig('Plots/SR_3h2.pdf', format='pdf', bbox_inches='tight')


##################################################################################################
# Compare epsilon and its components
fig4, ax4 = plt.subplots(1, 2, figsize=(10, 6))

# PyT
ax4[0].plot(back[:, 0], PyT_eps,   label=r'PyT $\epsilon$')
ax4[0].plot(back[:, 0], PyT_K_phi, label=r'PyT $K\left(\phi\right)$', linestyle='dashed')
ax4[0].plot(back[:, 0], PyT_K_rho, label=r'PyT $K\left(\rho\right)$')

# SR
ax4[1].plot(sol.t, SR_eps,   label=r'SR $\epsilon$')
ax4[1].plot(sol.t, SR_K_phi, label=r'SR $K\left(\phi\right)$', linestyle='dashed')
ax4[1].plot(sol.t, SR_K_rho, label=r'SR $K\left(\rho\right)$')

# Settings
fig4.suptitle(r'$\epsilon$ and components')
ax4[0].set_xlabel(r'$N$')
ax4[1].set_xlabel(r'$N$')
ax4[0].legend()
ax4[1].legend()

plt.savefig('Plots/SR_epsilon.pdf', format='pdf', bbox_inches='tight')

##################################################################################################
fig5, ax5 = plt.subplots(1,2,figsize=(10,6))
# PyT
ax5[0].plot(back[:,0], PyT_eta, label=r'PyT $\eta$')
# SR
ax5[1].plot(sol.t, SR_eta, label=r'SR $\eta$')
# Settings
fig5.suptitle(r'$\eta$')
ax5[0].set_xlabel(r'$N$')
ax5[1].set_xlabel(r'$N$')
ax5[0].legend()
ax5[1].legend()
plt.savefig('Plots/SR_eta.pdf', format='pdf',bbox_inches='tight')

##################################################################################################
fig6, ax6 = plt.subplots(1, 2, figsize=(10, 6))

# PyT
ax6[0].plot(back[:, 0], back[:, 1], label=r'PyT $\phi$')
ax6[0].plot(sol.t, sol.y[1] + phi_pert, label=r'Full $\phi$', linestyle='dashed')

# SR + pert
ax6[1].plot(back[:, 0], back[:, 3], label=r'PyT $\dot \phi$')
ax6[1].plot(sol.t, np.multiply(sol.y[2], sol.y[0]) + phi1_pert * h_sr, label=r'Full $\dot \phi$', linestyle='dashed')

# Inset for phi
axins0 = inset_axes(ax6[0], width="40%", height="40%", loc="lower left")
axins0.plot(back[:, 0], back[:, 1])
axins0.plot(sol.t, sol.y[1] + phi_pert, linestyle='dashed')
axins0.set_xlim(0, 5)
axins0.set_ylim(14, 16)  # adjust as needed
axins0.set_xticklabels([])
axins0.set_yticklabels([])
mark_inset(ax6[0], axins0, loc1=2, loc2=4, fc="none", ec="0.5")

# Inset for dot phi
axins1 = inset_axes(ax6[1], width="40%", height="40%", loc="upper center")
axins1.plot(back[:, 0], back[:, 3])
axins1.plot(sol.t, np.multiply(sol.y[2], sol.y[0]) + phi1_pert * h_sr, linestyle='dashed')
axins1.set_xlim(20, 21)
axins1.set_ylim(-0.825, -0.805)  # adjust as needed
axins1.set_xticklabels([])
axins1.set_yticklabels([])
mark_inset(ax6[1], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

# Settings
fig6.suptitle(r'$\phi$ and $\dot \phi$ Full analytical solution vs PyTransport')
ax6[0].set_xlabel(r'$N$')
ax6[1].set_xlabel(r'$N$')
ax6[0].legend()
ax6[1].legend()

plt.savefig('Plots/phi_full_sol.pdf', format='pdf', bbox_inches='tight')
##################################################################################################
fig7, ax7 = plt.subplots(1, 2, figsize=(10, 6))

# PyT
ax7[0].plot(back[:, 0], back[:, 2], label=r'PyT $\rho$')
ax7[0].plot(sol.t, sol.y[3] + rho_pert, label=r'Full $\rho$', linestyle='dashed')

# SR + pert
ax7[1].plot(back[:, 0], back[:, 4], label=r'PyT $\dot \rho$')
ax7[1].plot(sol.t, np.multiply(sol.y[4], sol.y[0]) + rho1_pert, label=r'Full $\dot \rho$', linestyle='dashed')

# Inset for rho
axins0 = inset_axes(ax7[0], width="40%", height="40%", loc="lower left")
axins0.plot(back[:, 0], back[:, 2])
axins0.plot(sol.t, sol.y[3] + rho_pert, linestyle='dashed')
axins0.set_xlim(0, 5)
axins0.set_ylim(-1e-5, 3e-5)  # adjust based on your data
axins0.set_xticklabels([])
axins0.set_yticklabels([])
mark_inset(ax7[0], axins0, loc1=2, loc2=4, fc="none", ec="0.5")

# Inset for dot rho
axins1 = inset_axes(ax7[1], width="40%", height="40%", loc="upper center")
axins1.plot(back[:, 0], back[:, 4])
axins1.plot(sol.t, np.multiply(sol.y[4], sol.y[0]) + rho1_pert, linestyle='dashed')
axins1.set_xlim(20, 21)
axins1.set_ylim(-0.0003, 0.0003)  # adjust based on your data
axins1.set_xticklabels([])
axins1.set_yticklabels([])
mark_inset(ax7[1], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

# Settings
fig7.suptitle(r'$\rho$ and $\dot \rho$ Full analytical solution vs PyTransport')
ax7[0].set_xlabel(r'$N$')
ax7[1].set_xlabel(r'$N$')
ax7[0].legend()
ax7[1].legend()

plt.savefig('Plots/rho_full_sol.pdf', format='pdf', bbox_inches='tight')


##################################################################################################
fig8 = plt.figure(figsize=(10,6))
# PyT
plt.plot(back[:,0], PyT_3H2, label=r'PyT $3H^2$')
# FA
plt.plot(sol.t, FA_3H2, label=r'FA $3H^2$', linestyle='dashed')
plt.title(r'$3H^2$ Comparison: PyTransport vs Full Analytical')
plt.xlabel(r'$N$')
plt.legend()
plt.savefig('Plots/3H2_full_comparison.pdf', format='pdf', bbox_inches='tight')

##################################################################################################
fig9 = plt.figure(figsize=(10,6))
plt.plot(back[:,0], PyT_eps, label=r'PyT $\epsilon$')
# FA
plt.plot(sol.t, FA_eps, label=r'FA $\epsilon$', linestyle='dashed')
plt.title(r'$\epsilon$ Comparison: PyTransport vs Full Analytical')
plt.xlabel(r'$N$')
plt.legend()
plt.savefig('Plots/eps_full_comparison.pdf', format='pdf', bbox_inches='tight')

##################################################################################################
fig10, ax10 = plt.subplots(1,2,figsize=(10,6))
# SR
ax10[0].plot(sol.t, sol.y[1], label=r'SR $\phi$')
ax10[0].plot(sol1.t, sol1.y[1], label=r'Coupling $\phi$', linestyle='dashed')
# SR + coupling
ax10[1].plot(sol.t, sol.y[2], label=r'SR $\dot \phi$')
ax10[1].plot(sol.t, sol1.y[2], label=r'Coupling $\dot \phi$', linestyle='dashed')
# Settings
fig10.suptitle(r'SR vs SR + coupling $\phi$ and $\dot \phi$')
ax10[0].set_xlabel(r'$N$')
ax10[1].set_xlabel(r'$N$')
ax10[0].legend()
ax10[1].legend()
plt.savefig('Plots/phi_coupling_comparison.pdf', format='pdf',bbox_inches='tight')

##################################################################################################
fig11, ax11 = plt.subplots(1,2,figsize=(10,6))
# SR
ax11[0].plot(sol.t, sol.y[3], label=r'SR $\rho$')
ax11[0].plot(sol1.t, sol1.y[3], label=r'Coupling $\rho$', linestyle='dashed')
# SR + coupling
ax11[1].plot(sol.t, sol.y[4], label=r'SR $\dot \rho$')
ax11[1].plot(sol.t, sol1.y[4], label=r'Coupling $\dot \rho$', linestyle='dashed')
# Settings
fig11.suptitle(r'SR vs SR + coupling $\rho$ and $\dot \rho$')
ax11[0].set_xlabel(r'$N$')
ax11[1].set_xlabel(r'$N$')
ax11[0].legend()
ax11[1].legend()
plt.savefig('Plots/rho_coupling_comparison.pdf', format='pdf',bbox_inches='tight')

##################################################################################################

fig12, ax12 = plt.subplots(1, 2, figsize=(10, 6))

# SR
ax12[0].plot(sol.t, sol.y[3], label=r'SR $\rho$')
ax12[0].plot(sol1.t, sol1.y[3], label=r'Coupling $\rho$', linestyle='dashed')

# SR + coupling
ax12[1].plot(sol.t, sol.y[4], label=r'SR $\dot \rho$')
ax12[1].plot(sol1.t, sol1.y[4], label=r'Coupling $\dot \rho$', linestyle='dashed')

# === Inset Zoom on ax12[0] ===
axins0 = inset_axes(ax12[0], width="45%", height="45%", loc="upper right")
axins0.plot(sol.t, sol.y[3], label=r'SR $\rho$')
axins0.plot(sol1.t, sol1.y[3], label=r'Coupling $\rho$', linestyle='dashed')

x1, x2 = 0, 5  # x-limits
y1, y2 = -6e-6, 2.6e-5  # y-limits
axins0.set_xlim(x1, x2)
axins0.set_ylim(y1, y2)
axins0.set_xticklabels([])
axins0.set_yticklabels([])
mark_inset(ax12[0], axins0, loc1=2, loc2=4, fc="none", ec="0.5")

# === Inset Zoom on ax12[1] ===
axins1 = inset_axes(ax12[1], width="45%", height="45%", loc="upper right")
axins1.plot(sol.t, sol.y[4], label=r'SR $\dot \rho$')
axins1.plot(sol1.t, sol1.y[4], label=r'Coupling $\dot \rho$', linestyle='dashed')

# Set limits for the second zoom — adjust to your case
x3, x4 = 0, 5  # x-limits
y3, y4 = -1e-5, 1e-5  # y-limits
axins1.set_xlim(x3, x4)
axins1.set_ylim(y3, y4)
axins1.set_xticklabels([])
axins1.set_yticklabels([])
mark_inset(ax12[1], axins1, loc1=2, loc2=4, fc="none", ec="0.5")

# Labels and legend
fig12.suptitle(r'SR vs SR + coupling $\rho$ and $\dot \rho$')
ax12[0].set_xlabel(r'$N$')
ax12[1].set_xlabel(r'$N$')
ax12[0].legend()
ax12[1].legend()

plt.savefig('Plots/rho_coupling_comparison_zoom.pdf', format='pdf', bbox_inches='tight')


##################################################################################################
fig13 = plt.figure(figsize=(10,6))
plt.plot(back[:,0], PyT_eta, label=r'PyT $\eta$')
# FA
plt.plot(sol.t, FA_eta, label=r'FA $\eta$', linestyle='dashed')
plt.title(r'$\eta$ Comparison: PyTransport vs Full Analytical')
plt.xlabel(r'$N$')
plt.legend()
plt.savefig('Plots/eta_full_comparison.pdf', format='pdf', bbox_inches='tight')

##################################################################################################
fig14, ax14 = plt.subplots(1,3,figsize=(10,6))
# alpha
ax14[0].plot(sol.t, alpha_PyT, label=r'PyT $\alpha$')
ax14[0].plot(sol.t, alpha_sr, label=r'SR $\alpha$', linestyle='dashed')
# beta
ax14[1].plot(sol.t, beta_PyT, label=r'PyT $\beta$')
ax14[1].plot(sol.t, beta_sr, label=r'SR $\beta$', linestyle='dashed')
# Lambda
ax14[2].plot(sol.t, np.abs(lambda_PyT), label=r'PyT $\Lambda$ condition')
ax14[2].plot(sol.t, np.abs(lambda_sr), label=r'SR $\Lambda$ condition', linestyle='dashed')
# Settings
fig14.suptitle(r'Evolution of the parameters')
ax14[0].set_xlabel(r'$N$')
ax14[1].set_xlabel(r'$N$')
ax14[2].set_xlabel(r'$N$')
ax14[0].legend()
ax14[1].legend()
ax14[2].legend()
plt.savefig('Plots/parameters_comparison.pdf', format='pdf',bbox_inches='tight')

#####################################################################################################################
plt.show()