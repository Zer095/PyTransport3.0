import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.interpolate import UnivariateSpline


from PyTransport.PyTransPy import PyTransSetup
from PyTransport.PyTransPy import PyTransScripts as PyS
import PyTransMatt as PyT
#####################################################################################################################################
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
#####################################################################################################################################
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
#print(eps_func(back[:,0]))

# calculate eta(N)
deps = eps_func.derivative()
eta = deps([back[:,0]])/eps_func(back[:,0])
eta_func = UnivariateSpline(back[:,0], eta, s=0)
#print(eta_func(back[:,0]))

#####################################################################################################################################

kpeak = 5.018093148060832505e+02
kL= 0.0004*kpeak 

k = []
ns = []

# Load data for the clean Maldacena relation
f = open('Data/data.txt')
lines = [line for line in f]
for i in range(len(lines)):
    k.append(float(lines[i].split(' ')[0]))
    ns.append(float(lines[i].split(' ')[1]))

k_data = np.divide(np.array(k), k[0])
ns_data = np.divide(np.array(ns), ns[0])

fnl_data = (5/12)*(np.array(ns) - 1)

# Define the path to the CSV file
csv_file = 'Data/SQTotal.csv'

# Initialize lists for each column
fact_list = []
k_list = []
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
        k_list.append(float(row[1]))
        pzt_list.append(float(row[2]))
        pzm_list.append(float(row[3]))
        fnlt_list.append(float(row[4]))
        fnlm_list.append(float(row[5]))

pzt_norm = np.divide( np.array( pzt_list ), pzt_list[0] )
pzm_norm = np.divide( np.array( pzm_list ), pzm_list[0] )
fnlt_norm = np.divide( np.array( fnlt_list ), fnlt_list[0] )
fnlm_norm = np.divide( np.array( fnlm_list ), fnlm_list[0] )



#####################################################################################################################################
fig1 = plt.figure(1, figsize=(10,8))
plt.plot(k, np.abs(ns))
plt.grid()
plt.yscale('log')
plt.xscale('log')

fig2 = plt.figure(2, figsize=(10,8))
plt.plot(k_list, np.abs(fnlt_norm), label='PyT')
plt.plot(k_list, np.abs(fnlm_norm), label='MPP', linestyle='dashed')
plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.legend()

fig3 = plt.figure(3, figsize=(10,8))
plt.plot(k_list, np.abs(fnlm_list), label='MPP')
plt.plot(k, np.abs(fnl_data), label='Maldacena', linestyle='dashed')
plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.legend()

fig4 = plt.figure(4, figsize=(10,8))
plt.plot(k_list, np.abs(fnlt_list), label='PyT')
plt.plot(k, np.abs(fnl_data), label='Maldacena', linestyle='dashed')
plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.legend()
#####################################################################################################################################
plt.show()