import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

from PyTransport.PyTransPy import PyTransSetup
from PyTransport.PyTransPy import PyTransScripts as PyS

import PyTransAxQ as PyT
###############################################################################################################
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
def H(x, x1, y, y1, m_phi, m_rho, rho_0, lmbd):
    V = Pot(x, y, m_phi, m_rho, rho_0)
    K =  Kin(x1, y1, y, lmbd)

    try:
        return np.sqrt( V/( 3 - K ) )
    except RuntimeWarning:
        print(f'V = {V}, K = {K}')
        return None

def PyT_H(back, params):
    H = []
    for i in range(np.size(back[:,0])):
        Hv = PyT.H(back[i,1:], params)
        H.append(Hv)
    H = np.array(H)
    return H

def Epsilon(N,H):
    H_func = UnivariateSpline(N, H, s=0)
    #H_log_func = UnivariateSpline(N, np.log(H), s=0)
    H_prime = H_func.derivative()
    eps = -H_prime(N)/(H)
    #eps = -1*H_log_func.derivative()(N)
    return eps

def Eta(N,Epsilon):
    Eps_func = UnivariateSpline(N,Epsilon,s=0)
    Eps_prime = Eps_func.derivative()
    eta = Eps_prime(N)/Epsilon
    return eta

def TH2(sol, m_phi, m_rho, rho_0):
    # Unpack solutions
    phi = sol[:,1]
    vel_phi = sol[:,2]
    rho = sol[:,3]
    vel_rho = sol[:,4]

    # Compute 3H^2 with both methods
    # Analytic solutions
    pot_phi = 0.5*m_phi*phi**2 
    pot_rho =  0.5*m_rho*(rho - rho_0)**2
    k_phi = 0.5*vel_phi**2
    k_rho = 0.5*vel_rho**2

    return pot_phi + pot_rho + k_phi + k_rho


def Energy_SR(sol, params, SR=True, n_max=-1):
    # Unpack solutions
    if SR:
        h = sol[:n_max,0]
        phi = sol[:n_max,1]
        rho = sol[:n_max,2]
        vel_phi = sol[:n_max,3]
        vel_rho = sol[:n_max,4]
    else:
        h = PyT_H(sol, params)
        phi = sol[:n_max,1]
        rho = sol[:n_max,2]
        vel_phi = sol[:n_max,3]
        vel_rho = sol[:n_max,4]
    # Unpack params
    m_phi = params[1]
    m_rho = params[4]
    rho_0 = params[5]
    lmbd = params[0]
    # Compute potential energy
    pot_phi = 0.5*m_phi*phi**2 
    pot_rho =  0.5*m_rho*(rho - rho_0)**2
    # Compute kinetic energy
    if SR:
        k_phi = 0.5*(vel_phi**2)
    else:
        k_phi = 0.5*np.exp(rho/lmbd)*(vel_phi**2)
    k_rho = 0.5*(vel_rho**2)
    return pot_phi, pot_rho, k_phi, k_rho

def deltaP(eps, lmbd, b):
    E = 0.02
    delta = -2.*(eps/lmbd**2)*E*b
    return delta

def P_z(k, H, eps, alpha, lmbd, b):
    delta = deltaP(eps, lmbd, b)
    theta = 0
    p_k = (H**2/(2*k**3))*(1. + delta*np.cos(alpha*np.log(k) + theta))
    return p_k

def get_sol(N, params, phi_0):
    # Unpack parameters
    lmbd = params[0]
    m_phi = params[1]
    m_rho = params[4]
    rho_0 = params[5]

    h_0 = np.sqrt( (m_phi*phi_0**2)/6. )

    # Phi
    h = np.sqrt(h_0**2 - (2/3)*m_phi*N)
    phi = np.sqrt(phi_0**2 - 4*N)
    v_phi = - 2/np.sqrt(phi_0**2 - 4*N)
    rho = ( (m_phi)/(3*m_rho*lmbd) + rho_0 )*np.ones(len(N))
    v_rho = np.zeros(len(N))

    return h, phi, rho, v_phi, v_rho

def transform(sol1, sol2, omega, b, n):
    # Unpack
    if len(sol1) == 2:
        y1 = sol1[0]
        v1 = sol1[1]
        y2 = sol2[0]
        v2 = sol2[1]
    else:
        print('Error')
    # Compute coefficient
    k = 1./(omega**2*b**2)
    C1 = k*(omega**2*y1[n]*y2[n] + v1[n]*v2[n])
    C2 = k*(y1[n]*v2[n] - v1[n]*y2[n])

    # Solution 
    return 1.3*(C1*y2 + C2*v2) + (y1[0] - y2[0])/2


