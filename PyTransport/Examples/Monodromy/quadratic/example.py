import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.integrate import solve_ivp

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

print(f'Ratio B/A = {beta/alpha}')

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

print(f'Parameters = {params}')

fields = np.array([phi_0, 0.]) 

V = PyT.V(fields,params) # calculate potential from some initial conditions
dV=PyT.dV(fields,params) # calculate derivatives of potential

print(f'V = {V}')
print(f'dV = {dV}')
vel = np.array([-dV[0]/np.sqrt(3* V), 0])
initial = np.concatenate((fields, vel)) # sets an array containing field values and there derivative in cosmic time 
                                                      # (set using the slow roll equation)
h = PyT.H(initial, params)
print(f'H = {h}')
####################################### Calculate the Background evolution ########################################
Nstart = 0.
Nend = 105.0
t = np.linspace(Nstart, Nend, 10000)
tols = np.array([10**-25,10**-25])
back = PyT.backEvolve(t, initial, params, tols, True)

print(f'Nend = {back[-1,0]}')

# Compute the first three slow-roll parameters
epsilons = PyS.epsilonI(3, back, params, PyT)
print(f'Len Epsilon = {len(epsilons)}')

H = []
for i in range(np.size(back[:,0])):
    Hv = PyT.H(back[i,1:], params)
    H.append(Hv)

H = np.array(H)

eps_alt = 0.5*(np.exp(back[:,2]/lmbd)*back[:,3]**2 + back[:,4]**2)/H**2

# Numerical integration of the slow-roll solution in time
def SR_system(t, vars, lmbd, m_phi, m_rho, rho_0):
    x1, x2, y1, y2 = vars
    h = np.sqrt( (m_phi*x1**2 + m_rho*(y1 - rho_0)**2)/6. ) + np.sqrt((np.exp(y1/lmbd)*x2**2 + y2**2)/6.) 
    dx1 = x2
    dx2 =  - 3*(h**2)*x2 - (x2*y2)/lmbd  - np.exp(-1*y1/lmbd)*m_phi*x1 
    dy1 = y2
    dy2 = - 3*(h**2)*y2  + np.exp(y1/lmbd)*(x2**2)/(2*lmbd) - m_rho*(y1 - rho_0)  
    return [dx1, dx2, dy1, dy2]

v_phi = -m_phi*phi_0/np.sqrt(3/2*m_phi*phi_0**2)
ics = [phi_0, vel[0], 0, 0]
t_span = (Nstart, back[-1,0])
t_eval = np.linspace(Nstart, back[-1,0], len(back[:,0]))
sol = solve_ivp(SR_system, t_span, ics, t_eval=back[:,0], args=(lmbd, m_phi, m_rho, rho_0))


# Numerical integration of the slow-roll solution in e-folds
# def SR_systemN(N, vars, lmbd, m_phi, m_rho, rho_0):
#     x1, x2, y1, y2 = vars
#     h = np.sqrt( (m_phi*x1**2 + m_rho*(y1 - rho_0)**2)/6. ) + np.sqrt( (np.exp(y1/lmbd)*(x2**2) + y2**2)/6.) 
#     dx1 = x2/h
#     dx2 = -3*x2 - (x2*y2)/(h*lmbd) - np.exp( (-1*y1)/lmbd)*(m_phi/h)*x1
#     dy1 = y2/h
#     dy2 = -3*y2 + np.exp(y1/lmbd)*(x2**2)/(2*h*lmbd) - (m_rho/h)*(y1 - rho_0) 
#     return [dx1, dx2, dy1, dy2]

# v_phi = -(m_phi*phi_0)/np.sqrt((3/2)*m_phi*phi_0**2)
# ics = [phi_0, v_phi, 0, 0]
# t_span = (Nstart, back[-1,0])
# t_eval = np.linspace(Nstart, back[-1,0], len(back[:,0]))
# solN = solve_ivp(SR_systemN, t_span, ics, t_eval=back[:,0], args=(lmbd, m_phi, m_rho, rho_0))

# # Numerical integration of the slow-roll solution in e-folds
def SR_systemN(N, vars, lmbd, m_phi, m_rho, rho_0):
    x1, x2, y1, y2 = vars
    h = np.sqrt( (m_phi*x1**2 + m_rho*(y1 - rho_0)**2)/6. ) + np.sqrt( (np.exp(y1/lmbd)*(x2**2) + y2**2)/6.) 
    dx1 = x2/h
    dx2 = -3*x2 - (x2*y2)/(h*lmbd) - np.exp( (-1*y1)/lmbd)*(m_phi/h)*x1
    dy1 = y2/h
    dy2 = -3*y2 + np.exp(y1/lmbd)*(x2**2)/(2*h*lmbd) - (m_rho/h)*(y1 - rho_0) 
    return [dx1, dx2, dy1, dy2]

# v_phi = -(m_phi*phi_0)/np.sqrt((3/2)*m_phi*phi_0**2)
# ics = [phi_0, v_phi, 0, 0]
# t_span = (Nstart, back[-1,0])
# t_eval = np.linspace(Nstart, back[-1,0], len(back[:,0]))
# solN = solve_ivp(SR_systemN, t_span, ics, t_eval=back[:,0], args=(lmbd, m_phi, m_rho, rho_0))

# Numerical integration of the slow-roll solution in e-folds
def SR_systemN(N, vars, lmbd, m_phi, m_rho, rho_0, a, f):
    x1, x2, y1, y2 = vars
    pot = 0.5*m_phi*(x1**2) + 0.5*m_rho*(y1-rho_0)**2
    kin = 0.5*np.exp(y1/lmbd)*(x2**2) + 0.5*y2**2
    h = np.sqrt(pot/3. + kin/3.)
    
    dpot_phi = m_phi*x1 
    dpot_rho = m_rho*(y1 - rho_0)

    dx1 = x2/h
    dx2 = -3*x2 - (x2*y2)/(h*lmbd) - np.exp( (-1*y1)/lmbd)*dpot_phi/h
    dy1 = y2/h
    dy2 = -3*y2 + np.exp(y1/lmbd)*(x2**2)/(2*h*lmbd) - dpot_rho/h
    return [dx1, dx2, dy1, dy2]

v_phi = -(m_phi*phi_0)/np.sqrt((3/2)*m_phi*phi_0**2)
ics = [phi_0, v_phi, 0, 0]
t_span = (Nstart, back[-1,0])
t_eval = np.linspace(Nstart, back[-1,0], len(back[:,0]))
solN = solve_ivp(SR_systemN, t_span, ics, t_eval=back[:,0], args=(lmbd, m_phi, m_rho, rho_0, a, f))


def Pert_sol(sol, lmbd, m_phi, m_rho, rho_0, a, f):
    N = sol.t
    x1 = sol.y[0]
    x2 = sol.y[1]
    y1 = sol.y[2]
    y2 = sol.y[3]

    h = np.sqrt( (m_phi*x1**2 + m_rho*(y1 - rho_0)**2)/6. ) + np.sqrt( (np.exp(y1/lmbd)*x2**2 + y2**2)/6.)
    ws = 3*h
    wn = np.sqrt(m_rho - 0.5*(x2**2/lmbd**2))
    wf = x2/f
    a_0 = -a/lmbd

    delta = np.arctan( (ws*wf)/(wn**2-wf**2) )
    B = a_0/np.sqrt((wn**2-wf**2)**2 + ws**2*wf**2)

    rho = B*np.cos(x1/f - delta)
    phi = -(a*f/(x2**2))*np.cos(x1/f)

    return [N, phi, rho]

pert = Pert_sol(solN, lmbd, m_phi, m_rho, rho_0, a, f)
#########################################################################################################

# Plots

fig1, ax1 = plt.subplots(1,2, figsize=(10,6))
fig1.suptitle('Full solution PyTransport')
ax1[0].plot(back[:,0], back[:,1], label=r'$\phi$')
ax1[0].plot(back[:,0], back[:,3], label=r'$\dot \phi$')
ax1[1].plot(back[:,0], back[:,2], label=r'$\rho$')
ax1[1].plot(back[:,0], back[:,4], label=r'$\dot \rho$')
ax1[0].legend()
ax1[1].legend()

fig2 = plt.figure(2,figsize=(8,6))
plt.title(r'$\rho(\phi)$ PyTransport')
plt.plot(back[:,1], back[:,2])
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\rho$')

# fig3, ax2 = plt.subplots(1,2, figsize=(10,6))
# fig3.suptitle('Numerical solution of differential equations #1')
# ax2[0].plot(sol.t, sol.y[0], label=r'$\phi$')
# ax2[0].plot(sol.t, sol.y[1], label=r'$\dot \phi$')
# ax2[1].plot(sol.t, sol.y[2], label=r'$\rho$')
# ax2[1].plot(sol.t, sol.y[3], label=r'$\dot \rho$')
# ax2[0].legend()
# ax2[1].legend()

# fig4, ax3 = plt.subplots(1,len(epsilons), figsize=(10,6))
# fig4.suptitle('Evolution of Slow-roll parameters')
# for i in range(len(epsilons)):
#     ax3[i].plot(back[:,0], epsilons[i])

# Plots
fig5, ax4 = plt.subplots(1,2, figsize=(10,6))
fig5.suptitle('Numerical solution of differential equations #2')
ax4[0].plot(solN.t, solN.y[0], label=r'$\phi$')
ax4[0].plot(solN.t, solN.y[1], label=r'$\dot \phi$')
ax4[1].plot(solN.t, solN.y[2], label=r'$\rho$')
ax4[1].plot(solN.t, solN.y[3], label=r'$\dot \rho$')
ax4[0].legend()
ax4[1].legend()

fig6, ax5 = plt.subplots(1,2, figsize=(10,6))
fig6.suptitle(r'$\phi$ and $\dot \phi$ confront solutions #2')
ax5[0].plot(back[:,0], back[:,1], label=r'PyT $\phi$')
ax5[0].plot(solN.t, solN.y[0], label=r'Analytical $\phi$')
ax5[1].plot(back[:,0], back[:,3], label=r'PyT $\dot \phi$')
ax5[1].plot(solN.t, solN.y[1], label=r'Analytical $\dot \phi$')
ax5[0].legend()
ax5[1].legend()

fig7, ax6 = plt.subplots(1,2, figsize=(10,6))
fig7.suptitle(r'$\rho$ and $\dot \rho$ confront solutions #2')
ax6[0].plot(back[:,0], back[:,2], label=r'PyT $\rho$')
ax6[0].plot(solN.t, solN.y[2], label=r'Analytical $\rho$')
ax6[1].plot(back[:,0], back[:,4], label=r'PyT $\dot \rho$')
ax6[1].plot(solN.t, solN.y[3], label=r'Analytical $\dot \rho$')
ax6[0].legend()
ax6[1].legend()

# fig8, ax7 = plt.subplots(1,2, figsize=(10,6))
# fig8.suptitle(r'$\phi$ and $\dot \phi$ confront solutions #1')
# ax7[0].plot(back[:,0], back[:,1], label=r'PyT $\phi$')
# ax7[0].plot(sol.t, sol.y[0], label=r'Analytical $\phi$')
# ax7[1].plot(back[:,0], back[:,3], label=r'PyT $\dot \phi$')
# ax7[1].plot(sol.t, sol.y[1], label=r'Analytical $\dot \phi$')
# ax7[0].legend()
# ax7[1].legend()

# fig9, ax8 = plt.subplots(1,2, figsize=(10,6))
# fig9.suptitle(r'$\rho$ and $\dot \rho$ confront solutions #1')
# ax8[0].plot(back[:,0], back[:,2], label=r'PyT $\rho$')
# ax8[0].plot(sol.t, sol.y[2], label=r'Analytical $\rho$')
# ax8[1].plot(back[:,0], back[:,4], label=r'PyT $\dot \rho$')
# ax8[1].plot(sol.t, sol.y[3], label=r'Analytical $\dot \rho$')
# ax8[0].legend()
# ax8[1].legend()


fig10, ax9 = plt.subplots(1,2, figsize=(10,6))
fig10.suptitle('Perturbed solutions')
ax9[0].plot(pert[0], pert[1], label=r'Perturbed $\phi$')
ax9[1].plot(pert[0], pert[2], label=r'Perturbed $\rho$')
ax9[0].legend()
ax9[1].legend()

fig11, ax10 = plt.subplots(1,2, figsize=(10,6))
fig11.suptitle('Full Analytical perturbed solutions')
ax10[0].plot(pert[0], solN.y[0] + pert[1], label=r'Full Perturbed $\phi$')
ax10[0].plot(back[:,0], back[:,1], label=r'PyT $\phi$')
ax10[1].plot(pert[0], solN.y[2] + pert[2], label=r'Full Perturbed $\rho$')
ax10[1].plot(back[:,0], back[:,2], label=r'PyT $\rho$')
ax10[0].legend()
ax10[1].legend()
#########################################################################################################
plt.show()
