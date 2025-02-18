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
from PyTransport.PyTransPy import PyTransSetup
PyTransSetup.pathSet()  # his add sets the other paths that PyTransport uses

import PyTransMatt as PyT;  # import module
import PyTransport.PyTransPy.PyTransScripts as PyS;
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


# calculate eta(N)
deps = eps_func.derivative()
eta = deps([back[:,0]])/eps_func(back[:,0])
eta_func = UnivariateSpline(back[:,0], eta, s=0)

###########################################################################################################################################

######################################## Read file  ################################################################

# Ks = []                 # K modes for each NExit
# MPP_Pz = []             # MPP Pz
# PyT_Pz = []             # PyT Pz
# MPP_fNL = []            # MPP f_NL
# PyT_fNL = []            # PyT f_NL

# file_path = os.path.join( os.path.dirname(__file__),  'cluster/Total.csv')
# file = open(file_path, 'r')

# csvFile = csv.reader(file)
# next(csvFile, None)
# for line in csvFile:
#     Ks.append(float(line[0]))
#     PyT_Pz.append(np.abs(float(line[1])))
#     MPP_Pz.append(np.abs(float(line[2])))
#     PyT_fNL.append(np.abs(float(line[3])))
#     MPP_fNL.append(np.abs(float(line[4])))

# Define the path to the CSV file
csv_file = 'Data/SqTotal.csv'

# Initialize lists for each column
fact_list = []
Ks = []
pzt_list = []
pzm_list = []
fnlt_list = []
fnlm_list = []

# Read the CSV file
with open(csv_file, 'r') as f:
    csvreader = csv.reader(f)
    # Skip the header row
    headers = next(csvreader)
    # Iterate through the rows and add values to respective lists
    for row in csvreader:
        fact_list.append(float(row[0]))
        Ks.append(float(row[1]))
        pzt_list.append(float(row[2]))
        pzm_list.append(float(row[3]))
        fnlt_list.append(float(row[4]))
        fnlm_list.append(float(row[5]))

pzt_norm = np.divide( np.array( pzt_list ), pzt_list[0] )
pzm_norm = np.divide( np.array( pzm_list ), pzm_list[0] )
fnlt_norm = np.divide( np.array( fnlt_list ), fnlt_list[0] )
fnlm_norm = np.divide( np.array( fnlm_list ), fnlm_list[0] )


###########################################################################################################################################

# print(len(Ks))
# print(Ks[0],Ks[-1])

# N0, a = PyS.ICsBE(0, Ks[0], back, pvalue, PyT)
# N1, b = PyS.ICsBE(0, Ks[-1], back, pvalue, PyT)

# print(N0, N1)

###########################################################################################################################################
from scipy.optimize import curve_fit

# Definition of the function to fit the power spectrum in order to obtain ns and alphas
def func(x,b,c):
    '''
    This function represents a quadratic fit for ln(P_\zeta(k))
    x = ln(k/k_star)
    a = d ns/d ln k
    b = ns-1
    c = ln(As)
    '''
    return b*x+c


sample_k = np.geomspace(Ks[0], 25435.782136144888, num=1000)

rng = np.random.default_rng(seed=42)

kvals = Ks
params = pvalue

nmb = 20
close = 0.05
data = np.array([]) 
size_test = 50

for kk in range(len(sample_k)):
    k = sample_k[kk]
    k_around = np.linspace(k - close*k, k + close*k, num=nmb)
    print(k)
    print(k_around)
    Pz, times = PyS.pSpectra(k_around, back, params, NB, tols, PyT)  
    normalisedPz_around = k_around**3. / (2. * np.pi**2.) * Pz
    
    ydata = np.log(normalisedPz_around)
    xdata = np.log(k_around/kvals[0])
    
    popt, pcov = curve_fit(func, xdata, ydata)
    ns = popt[0] + 1.
    
    data = np.append(data, [k, ns], axis=-1)
    
    plt.scatter(k_around/kvals[0], normalisedPz_around, color='black')
    plt.axvline(k/kvals[0])
    k_test = np.exp(np.log(np.min(k_around)) + rng.random(size=size_test) * (np.log(np.max(k_around))-np.log(np.min(k_around))))
    k_test = np.sort(k_test)

    
data = np.reshape(data,(int(len(data)/2),2))

np.savetxt('data.txt', data)