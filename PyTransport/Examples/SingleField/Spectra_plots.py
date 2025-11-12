import numpy as np
import matplotlib.pyplot as plt
import csv 

# File name
file_name = "Data/SFspectra.csv"

# List to store data
NOut = []
kOut = []
PyT_Pz = []
MPP_Pz = []
PyT_fnl = []
MPP_fnl = []

# Read file
with open(file_name, 'r') as file:
    file_csv = csv.reader(file)
    for line in file_csv:        
        try:
            line0 = str(line[0])[:5]
        except IndexError:
            line0 = str(line[0])[:4]

        NOut.append(float(line0))
        kOut.append(float(line[1]))
        PyT_Pz.append(float(line[2]))
        MPP_Pz.append(float(line[3]))
        PyT_fnl.append(float(line[4]))
        MPP_fnl.append(float(line[5]))

# Normalize Power-spectrum and f_NL
k_norm = np.divide(np.array(kOut), kOut[0])
PyT_Pz_norm = np.divide(np.array(PyT_Pz), PyT_Pz[0])
MPP_Pz_norm = np.divide(np.array(MPP_Pz), MPP_Pz[0])
PyT_fnl_norm = np.divide(np.array(PyT_fnl), PyT_fnl[0])
MPP_fnl_norm = np.divide(np.array(MPP_fnl), MPP_fnl[0])
 
# Plot parameters
sz_tit = 24
sz_leg = 20
sz_lab = 18

k_cmb = 324073.5674229475

# Plot Pz vs k
fig1 = plt.figure(1, figsize=(10,8))
plt.plot(k_norm, np.abs(PyT_Pz_norm), label='PyT')
plt.plot(k_norm, np.abs(MPP_Pz_norm), label='MPP', linestyle='dashed')
plt.vlines(x=k_cmb/kOut[0], ymax=max(MPP_Pz_norm), ymin=min(MPP_Pz_norm), color='black',linestyle='dashed', label=r'$k_{CMB}$')
#plt.title(r'$P_{\zeta}$ spectrum, PyT vs MPP', fontsize = sz_tit)
plt.grid()
plt.legend(fontsize=sz_leg)
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$k/k_0$', fontsize = sz_lab)
plt.ylabel(r'$P_{\zeta}(k)$/$P_{\zeta}(k_0)$', fontsize = sz_lab)
plt.savefig('plots/SF_spectra_pz.pdf', format='pdf',bbox_inches='tight')

# Plot fnl vs k
fig2 = plt.figure(2, figsize=(10,8))
plt.plot(k_norm, np.abs(PyT_fnl_norm), label='PyT')
plt.plot(k_norm, np.abs(MPP_fnl_norm), label='MPP', linestyle='dashed')
plt.vlines(x=k_cmb/kOut[0], ymax=max(MPP_fnl_norm), ymin=min(MPP_fnl_norm), color='black',linestyle='dashed', label=r'$k_{CMB}$')
#plt.title(r'$f_{NL}$ spectrum, PyT vs MPP', fontsize = sz_tit)
plt.grid()
plt.legend(fontsize=sz_leg)
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$k/k_0$', fontsize = sz_lab)
plt.ylabel(r'$f_{NL}(k)$/$f_{NL}(k_0)$', fontsize = sz_lab)
plt.savefig('plots/SF_spectra_fnl.pdf', format='pdf',bbox_inches='tight')

plt.show() 