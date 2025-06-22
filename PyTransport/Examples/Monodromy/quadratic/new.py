import numpy as np
import matplotlib.pyplot as plt
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

# Set parameters potential
params = np.zeros(nP)
params[0] = lmbd
params[1] = m_phi
params[2] = a
params[3] = f
params[4] = m_rho
params[5] = rho_0

print(params)
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
####################################################################################################################

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

# Solve SLOW-ROLL solution
# def System(N, vars, lmbd, m_phi, m_rho, rho_0):
#     # Unpack variables
#     h, x, x1, y, y1 = vars
#     # Kinetick energy
#     K = Kin(x1, y1, y, lmbd)
#     # Gradient of potential
#     dv, dw = DPot(x, y, m_phi, m_rho, rho_0)

#     # Evolution equation for H
#     dh = -1*h*K
#     # Ev. equation for phi
#     dx = x1
#     dx1 = -3*x1 - (dh/h)*x1 - (x1*y1)/lmbd - np.exp(-1*y/lmbd)*dv/(h**2)
#     # Ev. equation for rho
#     dy = y1
#     dy1 = -3*y1 - (dh/h)*y1 + (np.exp(y/lmbd)/(2*lmbd))*(x1**2) - dw/(h**2)
    
#     return [dh, dx, dx1, dy, dy1]

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
    dx1 = -3*x1 - (dh/h)*x1 - dv/(h**2) - (x1*y1)/lmbd 
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
ics = [h, phi_0, v_phi, 0., 0.]
t_span = (Nstart, back[-1,0])
t_eval = np.linspace(Nstart, back[-1,0], len(back[:,0]))
sol = solve_ivp(System, t_span, ics, t_eval=back[:,0], args=(lmbd, m_phi, m_rho, rho_0) )

# Compute the perturbed solution
def Phi_dot_pert(h, phi_0, phi_1, A, f):
    phi_dot_pert = -A/((phi_1*h))*np.cos(phi_0/f)
    return phi_dot_pert 

def Phi_pert(h, phi_0, phi_1, A, f):
    phi_pert = -A/((phi_1*h)**2)*np.sin(phi_0/f)
    return phi_pert

def Rho_pert(h, phi_0, phi_1, A, f, m_rho, lmbd):
    ws = 3*h
    wn = np.sqrt(m_rho)
    wf = (phi_1*h)/f
    a_0 = A/lmbd
    delta = np.arctan((ws*wf)/(wn**2 - wf**2))
    beta = a_0/np.sqrt((wn**2 - wf**2)**2 + (ws**2)*(wf**2))
    rho_pert = beta*np.cos(phi_0/f + delta)
    return rho_pert

phi_dot_pert = Phi_dot_pert(sol.y[0],sol.y[1], sol.y[2], a, f)
phi_pert = Phi_pert(sol.y[0], sol.y[1], sol.y[2], a, f)
rho_pert = Rho_pert(sol.y[0],sol.y[1],sol.y[2],a,f,m_rho,lmbd)

def H_PyT(back, params):
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
#########################################################################################################################
# Two point correlation function
# NExit = back[-1,0] - 50.0
# NB = 5.0
# twoPt_tols = np.array([10**-8, 10**-8])
# k = PyS.kexitN(NExit, back, params, PyT)
# Nstart, backExit = PyS.ICsBE(NB, k, back, params, PyT)
# tsig = np.linspace(Nstart, back[-1, 0], 1000)
# twoPt = PyT.sigEvolve(tsig, k, backExit, params, twoPt_tols, True)
# sigma = twoPt[:,1+1+2*nF:]
#########################################################################################################################
# Plots
# fig1 = plt.figure(figsize=(10,8))
# plt.title("Evolution of Slow-roll H")
# plt.plot(sol.t, sol.y[0], label = r'$H(N)$')
# plt.xlabel(r'$N$')
# plt.ylabel(r'$H$')
# plt.grid()
# plt.legend()
# plt.savefig('Plots/SR_H_evo.pdf', format='pdf',bbox_inches='tight')

fig2, ax2 = plt.subplots(1,2, figsize=(10,6))
fig2.suptitle(r'$\phi$ and $\dot \phi$ confront solutions')
ax2[0].plot(back[:,0], back[:,1], label=r'PyT $\phi$')
ax2[0].plot(sol.t, sol.y[1], label=r'Slow-roll $\phi$', linestyle='dashed')
ax2[1].plot(back[:,0], back[:,3], label=r'PyT $\dot \phi$')
ax2[1].plot(sol.t, np.multiply(sol.y[2],sol.y[0]), label=r'Slow-roll $\dot \phi$', linestyle='dashed')
ax2[0].set_xlabel(r'$N$')
ax2[1].set_xlabel(r'$N$')
ax2[0].legend()
ax2[1].legend()
plt.savefig('Plots/SR_phi_sol.pdf', format='pdf',bbox_inches='tight')

fig3, ax3 = plt.subplots(1,2, figsize=(10,6))
fig3.suptitle(r'$\rho$ and $\dot \rho$ confront solutions')
ax3[0].plot(back[:,0], back[:,2], label=r'PyT $\rho$')
ax3[0].plot(sol.t, sol.y[3], label=r'Slow-roll $\rho$', linestyle='dashed')
ax3[1].plot(back[:,0], back[:,4], label=r'PyT $\dot \rho$')
ax3[1].plot(sol.t, np.multiply(sol.y[4],sol.y[0]), label=r'Slow-roll $\dot \rho$', linestyle='dashed')
ax3[0].set_xlabel(r'$N$')
ax3[1].set_xlabel(r'$N$')
ax3[0].legend()
ax3[1].legend()
plt.savefig('Plots/SR_rho_sol.pdf', format='pdf',bbox_inches='tight')

fig4, ax4 = plt.subplots(1,2, figsize=(10,6))
fig4.suptitle('Full solution + Slow-roll solution')
ax4[0].plot(back[:,0], back[:,1], label=r'PyT $\phi$')
ax4[0].plot(sol.t, sol.y[1]+phi_pert, label=r'Perturbed $\phi$', linestyle='dashed')
ax4[1].plot(back[:,0], back[:,2], label=r'PyT $\rho$')
ax4[1].plot(sol.t, sol.y[3]+rho_pert, label=r'Perturbed $\rho$', linestyle='dashed')
ax4[0].set_xlabel(r'$N$')
ax4[1].set_xlabel(r'$N$')
ax4[0].legend()
ax4[1].legend()
plt.savefig('Plots/SR_H_evo.pdf', format='pdf',bbox_inches='tight')

# fig5 = plt.figure(figsize=(10,6))
# plt.title('Evolution of Slow-roll epsilon')
# plt.plot(sol.t, Epsilon(sol.t, sol.y[0]), label=r'$\epsilon$')
# plt.xlabel(r'$N$')
# plt.grid()
# plt.yscale('log')
# plt.legend()

# fig6 = plt.figure(figsize=(10,6))
# plt.plot(back[:,0], H_PyT(back,params), label='PyT H')
# plt.xlabel(r'$N$')
# plt.ylabel(r'$H$')
# plt.grid()
# plt.legend()

# fig7 = plt.figure(figsize=(10,6))
# plt.plot(sol.t, Epsilon(back[:,0], H_PyT(back, params)), label=r'PyT $\epsilon$')
# plt.xlabel(r'$N$')
# plt.grid()
# plt.yscale('log')
# plt.legend()

fig8, ax5 = plt.subplots(1,2, figsize=(10,8))
# First plot PyTransport
ax5[0].plot(back[:,0], 3*(H_PyT(back,params))**2, label=r'PyT $3H^2$')
ax5[0].plot(back[:,0], 0.5*np.exp(back[:,2]/lmbd)*(back[:,3])**2, label=r'PyT $\phi$ kinetic energy')
ax5[0].plot(back[:,0], 0.5*(back[:,4])**2, label=r'PyT $\rho$ kinetic energy')
ax5[0].plot(back[:,0], 0.5*m_phi*back[:,1]**2, label=r'PyT $\phi$ potential energy',linestyle='dashed')
ax5[0].plot(back[:,0], 0.5*m_rho*(back[:,2] - rho_0)**2, label=r'PyT $\rho$ potential energy', linestyle='dashed')
ax5[0].set_title('PyTransport')
ax5[0].set_xlabel(r'$N$')
ax5[0].legend()
# Second plot SR solutions
ax5[1].plot(sol.t, 3*(sol.y[0])**2, label=r'SR $3H^2$')
ax5[1].plot(sol.t, 0.5*np.exp(sol.y[3]/lmbd)*(sol.y[2])**2, label=r'SR $\phi$ kinetic energy')
ax5[1].plot(sol.t, 0.5*(sol.y[4])**2, label=r'SR $\rho$ kinetic energy')
ax5[1].plot(sol.t, 0.5*m_phi*sol.y[1]**2, label=r'SR $\phi$ potential energy',linestyle='dashed')
ax5[1].plot(sol.t, 0.5*m_rho*(sol.y[3] - rho_0)**2, label=r'SR $\rho$ potential energy', linestyle='dashed')
ax5[1].set_title('Slow-roll solution')
ax5[1].set_xlabel(r'$N$')
ax5[1].legend()
plt.savefig('Plots/3H_evo.pdf', format='pdf',bbox_inches='tight')


# fig8 = plt.figure(figsize=(10,8))
# plt.title(r'Evolution of $3H^2$')
# plt.plot(back[:,0], 3*(H_PyT(back,params))**2, label=r'PyT $3H^2$')
# plt.plot(back[:,0], 0.5*np.exp(back[:,2]/lmbd)*(back[:,3])**2, label=r'$\phi$ kinetic energy')
# plt.plot(back[:,0], 0.5*(back[:,4])**2, label=r'$\rho$ kinetic energy')
# plt.plot(back[:,0], 0.5*m_phi*back[:,1]**2, label=r'$\phi$ potential energy',linestyle='dashed')
# plt.plot(back[:,0], 0.5*m_rho*(back[:,2] - rho_0)**2, label=r'$\rho$ potential energy', linestyle='dashed')
# plt.xlabel(r'$N$')
# plt.grid()
# plt.legend()

fig9, ax6 = plt.subplots(1,2, figsize=(10,8))
# First plot PyTransport
ax6[0].plot(back[:,0], Epsilon(back[:,0], H_PyT(back,params)), label=r'PyT $\epsilon$')
ax6[0].plot(back[:,0], (0.5/H_PyT(back,params)**2)*np.exp(back[:,2]/lmbd)*(back[:,3])**2, label=r'PyT $\phi$ kinetic energy', linestyle='dashed')
ax6[0].plot(back[:,0], (0.5/H_PyT(back,params)**2)*(back[:,4])**2, label=r'PyT $\rho$ kinetic energy')
ax6[0].set_title('PyTransport')
ax6[0].set_yscale('log')
ax6[0].set_xlabel(r'$N$')
ax6[0].legend()
# Second plot SR solutions
ax6[1].plot(back[:,0], Epsilon(sol.t, sol.y[0]), label=r'SR $\epsilon$')
ax6[1].plot(back[:,0], (0.5)*np.exp(sol.y[3]/lmbd)*(sol.y[2])**2, label=r'SR $\phi$ kinetic energy', linestyle='dashed')
ax6[1].plot(back[:,0], (0.5)*(sol.y[4])**2, label=r'SR $\rho$ kinetic energy')
ax6[1].set_title('Slow-roll solution')
ax6[1].set_yscale('log')
ax6[1].set_xlabel(r'$N$')
ax6[1].legend()
plt.savefig('Plots/epsilon_evo.pdf', format='pdf',bbox_inches='tight')

PyT_eta = Eta(back[:,0],Epsilon(back[:,0],H_PyT(back,params)))
SR_eta = Eta(sol.t, Epsilon(sol.t,sol.y[0]))
fig10, ax7 = plt.subplots(1,2, figsize=(10,8))
ax7[0].plot(back[:,0], PyT_eta, label=r'PyT $\eta$')
ax7[1].plot(sol.t, SR_eta, label=r'SR $\eta$')
# ax7[0].set_yscale('log')
# ax7[1].set_yscale('log')
ax7[0].set_xlabel(r'$N$')
ax7[1].set_xlabel(r'$N$')
ax7[0].legend()
ax7[1].legend()
plt.savefig('Plots/eta_evo.pdf', format='pdf',bbox_inches='tight')

# fig9 = plt.figure(figsize=(10,6))
# plt.title(r'PyTransport $\epsilon$ ')
# plt.plot(back[:,0], Epsilon(back[:,0], H_PyT(back,params)), label=r'$\epsilon$')
# plt.plot(back[:,0], (0.5/H_PyT(back,params)**2)*np.exp(back[:,2]/lmbd)*(back[:,3])**2, label=r'$\phi$ kinetic energy', linestyle='dashed')
# plt.plot(back[:,0], (0.5/H_PyT(back,params)**2)*(back[:,4])**2, label=r'$\rho$ kinetic energy')
# plt.xlabel(r'$N$')
# plt.yscale('log')
# plt.grid()
# plt.legend()

# fig10 = plt.figure(figsize=(10,6))
# plt.title(r'Kinetic coupling')
# plt.plot(back[:,0], np.exp(back[:,2]/lmbd), label=r'coupling')
# plt.xlabel(r'$N$')
# plt.yscale('log')
# plt.grid()
# plt.legend()

# fig11 = plt.figure(figsize=(10,6))
# plt.plot(back[:,0], back[:,3], label=r'PyT $\dot \phi$')
# plt.plot(sol.t, (sol.y[2]*sol.y[0])+phi_dot_pert, label=r'Perturbed $\phi$', linestyle='dashed')
# plt.xlabel(r'$N$')
# plt.grid()
# plt.legend()



# fig5 = plt.figure(figsize=(10,8))
# for i in range(2*nF):
#     for j in range(2*nF):
#         plt.plot(twoPt[:,0], np.abs(sigma[:,2*nF*i + j]))
# plt.yscale('log')
# plt.grid()

# fig6 = plt.figure(figsize=(10,8))
# plt.plot(twoPt[:,0], (k**3/(2*np.pi**2))*np.abs(twoPt[:,1]))
# plt.yscale('log')
# plt.grid()
#########################################################################################################################
plt.show()