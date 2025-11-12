import numpy as np
import matplotlib.pyplot as plt
import csv 

# File name
file_name = "Data/QAspectra.csv"

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


# Load cmb data
cmbdata = np.loadtxt('Data/cmbdata.txt')
k_cmb = cmbdata[1]
PyT_Pz_cmb = cmbdata[2]
MPP_Pz_cmb = cmbdata[3]
PyT_fnl_cmb = cmbdata[4]
MPP_fnl_cmb = cmbdata[5]

# Normalize Power-spectrum and f_NL
k_norm = np.divide(np.array(kOut), k_cmb)
PyT_Pz_norm = np.divide(np.array(PyT_Pz), PyT_Pz_cmb)
MPP_Pz_norm = np.divide(np.array(MPP_Pz), MPP_Pz_cmb)
PyT_fnl_norm = np.divide(np.array(PyT_fnl), PyT_fnl_cmb)
MPP_fnl_norm = np.divide(np.array(MPP_fnl), MPP_fnl_cmb)

k_cmb = 2346.7302447056713

# Plot parameters
sz_tit = 24
sz_leg = 20
sz_lab = 18
ticksz = 18

clr = ["#B30000", "#1A53FF", "#5AD45A", "#ED7F2C"]

# Plot Pz vs k
fig1 = plt.figure(1, figsize=(10,8))
plt.plot(k_norm, np.abs(PyT_Pz_norm), label='PyT')
plt.plot(k_norm, np.abs(MPP_Pz_norm), label='MPP', linestyle='dashed')
plt.vlines(x=k_cmb/k_cmb, ymax=max(MPP_Pz_norm), ymin=min(MPP_Pz_norm), color='black',linestyle='dashed', label=r'$k_{CMB}$')
#plt.title(r'$P_{\zeta}$ spectrum, PyT vs MPP', fontsize = sz_tit)
plt.grid()
plt.legend(fontsize=sz_leg)
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$k/k_0$', fontsize = sz_lab)
plt.ylabel(r'$P_{\zeta}(k)$/$P_{\zeta}(k_0)$', fontsize = sz_lab)
plt.savefig('plots/QA_spectra_pz.pdf', format='pdf',bbox_inches='tight')

# Plot fnl vs k
fig2 = plt.figure(2, figsize=(10,8))
plt.plot(k_norm, np.abs(PyT_fnl_norm), label='PyT')
plt.plot(k_norm, np.abs(MPP_fnl_norm), label='MPP', linestyle='dashed')
plt.vlines(x=k_cmb/k_cmb, ymax=max(MPP_fnl_norm), ymin=min(MPP_fnl_norm), color='black',linestyle='dashed', label=r'$k_{CMB}$')
#plt.title(r'$f_{NL}$ spectrum, PyT vs MPP', fontsize = sz_tit)
plt.grid()
plt.legend(fontsize=sz_leg)
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$k/k_0$', fontsize = sz_lab)
plt.ylabel(r'$f_{NL}(k)$/$f_{NL}(k_0)$', fontsize = sz_lab)
plt.savefig('plots/QA_spectra_fnl.pdf', format='pdf',bbox_inches='tight')

fig3 = plt.figure(3, figsize=(10,8))
plt.plot(k_norm, np.abs(PyT_Pz_norm), label=r'PyT $P_{\zeta}$', color=clr[0])
plt.plot(k_norm, np.abs(MPP_Pz_norm), label=r'MPP $P_{\zeta}$', linestyle=(0, (10, 8)), color=clr[2])
plt.plot(k_norm, np.abs(PyT_fnl_norm), label=r'PyT $f_{NL}$', color=clr[1])
plt.plot(k_norm, np.abs(MPP_fnl_norm), label=r'MPP $f_{NL}$', linestyle=(0, (10, 8)), color=clr[3])
plt.grid()
plt.legend(fontsize=sz_leg, framealpha=1.0)
plt.yscale('log')
plt.xscale('log')
plt.xticks(fontsize=ticksz)
plt.yticks(fontsize=ticksz)
plt.xlabel(r'$k/k_{CMB}$', fontsize = sz_lab)
plt.savefig('plots/QA_spectra_tot.pdf', format='pdf',bbox_inches='tight')

plt.show()