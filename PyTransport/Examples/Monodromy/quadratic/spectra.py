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
    dx1 = -3*x1 - (dh/h)*x1 - dv*(1 - y/lmbd)/(h**2) - (1/lmbd)*x1*y1
    # Ev. equation for rho
    dy = y1
    dy1 = -3*y1 - (dh/h)*y1 - dw/(h**2) + (1./(2*lmbd))*(1 + y/lmbd)*(x1**2)  

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


print('Here 1')

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

print('Here 2')

####################################### Compute Axion-perturbations to the SR solution #################################################
# Unpack SR solution
N = sol.t
h_sr = sol.y[0]
phi_sr = sol.y[1]
phi1_sr = sol.y[2]
rho_sr = sol.y[3]
rho1_sr = sol.y[4]

# Phi perturbed solution:
phi1_pert = -(a/(phi1_sr*h_sr))*np.cos(phi_sr/f)
phi_pert = -(a*f/(phi1_sr/h_sr)**2)*np.sin(phi_sr/f-phi1_sr[0]/f)

# Rho solution coefficients: 
omega_f = (phi1_sr*h_sr)/f
omega_s = 3*h_sr
omega_n = np.sqrt(m_rho)
a0 = -a/lmbd
B = a0/np.sqrt( (omega_n**2 - omega_f**2)**2 + (omega_f**2)*(omega_s**2) )
delta = np.arctan( (omega_s*omega_f)/(omega_n**2-omega_f**2) )
delta = np.arcsin((omega_n**2 - omega_f**2)/(np.sqrt( (omega_n**2 - omega_f**2)**2 + (omega_f**2)*(omega_s**2) )))
# Rho perturbed solution:
rho1_pert = (-B)*(omega_f)*np.cos( (phi_sr/f) + delta - ((phi_sr[0]/f) + delta[0] - np.pi/2 )) 
rho_pert = (B)*np.sin( (phi_sr/f) + delta - ((phi_sr[0]/f) + delta[0] ) ) 

# Rho perturbed + transient solution
A_0 = np.sqrt( (-B[0]*np.cos(delta[0]))**2 + ( (B[0]*((omega_s[0]/2)*np.cos(delta[0]) + omega_f[0]*np.sin(delta[0])) )/omega_s[0] )**2 )
pi = np.arctan( (B[0]*((omega_s[0]/2)*np.cos(delta[0])+omega_f[0]*np.sin(delta[0])))/((omega_s[0])*(-B[0]*np.cos(delta[0]))) )
rho_pert_test = A_0*np.exp(-(omega_s[0]/2)*N)*np.cos(omega_s[0]*N + pi) + (B)*np.sin( (phi_sr/f) + delta)

u_prime = - (omega_s[0]/2) * A_0 * np.exp(-(omega_s[0]/2)*N)
v_prime = - omega_s[0] * np.sin(omega_s[0]*N + pi)

velocity_homogeneous_term = (u_prime * np.cos(omega_s[0]*N + pi) +
                                A_0 * np.exp(-(omega_s[0]/2)*N) * v_prime)

velocity_particular_term = B * np.cos((phi_sr / f) + delta) * (phi1_sr / f)
rho_test = 1/(2*lmbd*m_rho)*(phi1_sr*h_sr)**2 + rho_0
rho1_pert_test = velocity_homogeneous_term + velocity_particular_term

# Define full analytical solutions:
phi1_full = phi1_sr + (phi1_pert/h_sr)
phi_full = phi_sr + phi_pert
rho1_full = rho1_sr + (rho1_pert/h_sr)
rho_full = rho_sr + rho_pert
  
# Compute the frequency:
num = len(back[:,0])


def dominant_frequencies(N, T, signal):
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(N, T )[:N//2]
    print(f'Len xf = {len(xf)}')

    magnitude = 2.0/N * np.abs(yf[0:N//2])

    # Plot the spectrum to identify the frequencies
    plt.figure(figsize=(10, 6))
    plt.plot(xf, magnitude)
    plt.title('Frequency Spectrum of the Oscillatory Function')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    sorted_indices = np.argsort(magnitude)[::-1]
    dominant_frequencies = []
    for idx in sorted_indices:
        if xf[idx] > 0.1:
            dominant_frequencies.append(xf[idx])
            if len(dominant_frequencies) == 2:
                break

    return dominant_frequencies


print('Here 3')

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


print('Here 4')

####################################### Power-spectrum ############################################################
# Build vectors for analytical solutions
N = sol.t
backAN = np.column_stack((N,phi_full,rho_full,phi1_full,rho1_full))

Rho_full = rho_test + rho_pert_test
Rho1_full = rho1_pert_test
backFA = np.column_stack((N,phi_full,Rho_full,phi1_full,Rho1_full))

print('Here 5')


# Set parameters
NB = 4.0
NExit = back[-1,0] - 50.0

PyT_k = PyS.kexitN(NExit, back, params, PyT)
AN_k = PyS.kexitN(NExit, backAN, params, PyT)
FA_k = PyS.kexitN(NExit, backFA, params, PyT)
print(f'PyT k = {PyT_k}, AN k = {AN_k}, FA k = {FA_k}')

PyT_Ns, PyT_backExit = PyS.ICsBE(NB, PyT_k, back, params, PyT)
AN_Ns, AN_backExit = PyS.ICsBE(NB, AN_k, backAN, params, PyT)
FA_Ns, FA_backExit = PyS.ICsBE(NB, FA_k, backFA, params, PyT)
print(f'PyT BackExit = {PyT_backExit},\n Analytical BackExit = {AN_backExit},\n Full Analytic BackExit = {FA_backExit}')

PyT_tsig = np.linspace(PyT_Ns, back[-1,0], 10000)
AN_tsig = np.linspace(AN_Ns, backAN[-1,0], 10000)
FA_tsig = np.linspace(FA_Ns, backAN[-1,0], 10000)

tols = np.array([10e-10,10e-10])

PyT_twoPt = PyT.sigEvolve(PyT_tsig, PyT_k, PyT_backExit, params, tols, True)
AN_twoPt = PyT.sigEvolve(AN_tsig, AN_k, AN_backExit, params, tols, True)
FA_twoPt = PyT.sigEvolve(FA_tsig, FA_k, FA_backExit, params, tols, True)

print('Here 6')

# Compute the spectra 

PyT_pzOut= np.array([])

PyT_kOut = np.array([])
PhiOut = np.array([])
NOut = np.array([])

for i in range(1001):
    NExit = 5 + i*0.05
    NOut = np.append(NOut, NExit)
    k = PyS.kexitN(NExit, back, params, PyT)
    PyT_kOut = np.append(PyT_kOut, k)

PyT_zzOut, times = PyS.pSpectra(PyT_kOut, back, params, NB, tols, PyT)

AN_kOut = np.array([])

for i in range(1001):
    NExit = 5 + i*0.05
    NOut = np.append(NOut, NExit)
    k = PyS.kexitN(NExit, backAN, params, PyT)
    AN_kOut = np.append(AN_kOut, k)

AN_zzOut, times = PyS.pSpectra(AN_kOut, backAN, params, NB, tols, PyT)

FA_kOut = np.array([])
for i in range(1001):
    NExit = 5 + i*0.05
    NOut = np.append(NOut, NExit)
    k = PyS.kexitN(NExit, backFA, params, PyT)
    FA_kOut = np.append(FA_kOut, k)

FA_zzOut, times = PyS.pSpectra(FA_kOut, backFA, params, NB, tols, PyT)

PyT_pzOut = (PyT_kOut**3/(2*np.pi**2))*PyT_zzOut
AN_pzOut = (AN_kOut**3/(2*np.pi**2))*AN_zzOut
FA_pzOut = (FA_kOut**3/(2*np.pi**2))*FA_zzOut

############################################################ Plots ############################################################

fig1, ax1 = plt.subplots(1, 3, figsize=(10,6))
ax1[0].plot(PyT_kOut, np.abs(PyT_pzOut), label='PyT')
ax1[1].plot(AN_kOut, np.abs(AN_pzOut), label='AN')
ax1[2].plot(FA_kOut, np.abs(FA_pzOut), label='FA')

# Settings
fig1.suptitle(r'$\mathcal{P}_\zeta$ confront')
ax1[0].set_yscale('log')
ax1[1].set_yscale('log')
ax1[2].set_yscale('log')
ax1[0].set_xlabel(r'$k$')
ax1[1].set_xlabel(r'$k$')
ax1[2].set_xlabel(r'$k$')
ax1[0].legend()
ax1[1].legend()
ax1[2].legend()

#####################################################################################################################
plt.show()