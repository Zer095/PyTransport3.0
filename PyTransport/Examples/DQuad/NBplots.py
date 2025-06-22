import numpy as np
import matplotlib.pyplot as plt
import csv

################################# Reading file ######################################

fileMPP = 'Data/NB_MPP.csv'     # File with quantities computed with MPP
filePyT = 'Data/NB_PyT.csv'     # File with quantities computed with PyT

tol = np.linspace(5, 18, 27)    # Array with tolerances
nbs = np.linspace(1, 14, 27)    # Array with Nbs

tsm = []        # Time-sigma MPP
tst = []        # Time-sigma Transport 
pzm = []        # Final Pz MPP
pzt = []        # Final Pz Transport
tam = []        # Time-alpha MPP
tat = []        # Time-alpha Transport
fnlm = []       # Final Fnl MPP
fnlt = []       # Final Fnl Transport
goldtol2M = []  # Golden Tolerance Pz Mpp
goldtol2T = []  # Golden Tolerance Pz Trans
goldtol3M = []  # Golden Tolerance Fnl MPP
goldtol3T = []  # Golden Tolerance Fnl Transport

table = np.ones(shape=(27,27,12)) # table(i,j,k): i = NBs, j = Tol, k = [tsm, tst, .. ]
scaling = (1.702050*10.0**(-6.0))**2 # scaling factor for P_z

# Load data into table
with open(fileMPP, mode='r') as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        i = np.where(nbs == float(line[0]))[0][0]       # NB index
        j = np.where(tol == abs(float(line[1])))[0][0]  # Tol index

        table[i,j,0] = float(line[2])                   # Time 2pt
        table[i,j,1] = float(line[3])                   # Pz
        table[i,j,4] = float(line[4])                   # Time 3pt
        table[i,j,5] = float(line[5])                   # fNL
 
with open(filePyT, mode='r') as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        i = np.where(nbs == float(line[0]))[0][0]       # NB index
        j = np.where(tol == abs(float(line[1])))[0][0]  # Tol index
        table[i,j,2] = float(line[2])                   # Time 2pt
        table[i,j,3] = float(line[3])                   # Pz
        table[i,j,6] = float(line[4])                   # Time 3pt
        table[i,j,7] = float(line[5])                   # fNL

# Compute the relative error
epsilon = list(np.multiply(tol,-1.0))
for i in range(27):
    # Loop over NBs
    pzm = []
    pzt = []
    fnlm = []
    fnlt = []
    for j in range(27):
        #Loop over TOLs
        pzm.append(scaling*table[i,j,1])
        pzt.append(scaling*table[i,j,3])
        fnlm.append(table[i,j,5])
        fnlt.append(table[i,j,7])

    for j in range(1,len(pzm)):
        # Loop over Tols
        table[i,j,8] = np.abs(
            (pzm[j] - pzm[j-1])/pzm[j]
        )

        table[i,j,9] = np.abs(
            (pzt[j]-pzt[j-1])/pzt[j]
        )

        table[i,j,10] = np.abs(
            (fnlm[j]-fnlm[j-1])/fnlm[j]
        )

        table[i,j,11] = np.abs(
            (fnlt[j]-fnlt[j-1])/fnlt[j]
        )

# Threshold for computing golden tolerance
threshold = 10**-8
stable_points = 3
# Compute golden tolerance
for i in range(27):
    try:
        # MPP 2pt GT
        idx = np.where(table[i, :, 8] < threshold)[0]
        if len(idx) >= stable_points and np.all(table[i, idx[:stable_points], 8] < threshold):
            goldtol2M.append(epsilon[idx[0]])  # Take the first stable index
        else:
            raise IndexError
    except IndexError:
        print(f"MPP fails 2pt stability, NB = {nbs[i]}")
        goldtol2M.append(np.nan)

    try:
        # PyT 2pt GT
        idx = np.where(table[i, :, 9] < threshold)[0]
        if len(idx) >= stable_points and np.all(table[i, idx[:stable_points], 9] < threshold):
            goldtol2T.append(epsilon[idx[0]])
        else:
            raise IndexError
    except IndexError:
        print(f"PyTransport fails 2pt stability, NB = {nbs[i]}")
        goldtol2T.append(np.nan)

    try:
        # MPP 3pt GT
        idx = np.where(table[i, :, 10] < threshold)[0]
        if len(idx) >= stable_points and np.all(table[i, idx[:stable_points], 10] < threshold):
            goldtol3M.append(epsilon[idx[0]])
        else:
            raise IndexError
    except IndexError:
        print(f"MPP fails 3pt stability, NB = {nbs[i]}")
        goldtol3M.append(np.nan)

    try:
        # PyT 3pt GT
        idx = np.where(table[i, :, 11] < threshold)[0]
        if len(idx) >= stable_points and np.all(table[i, idx[:stable_points], 11] < threshold):
            goldtol3T.append(epsilon[idx[0]])
        else:
            raise IndexError
    except IndexError:
        print(f"PyTransport fails 3pt stability, NB = {nbs[i]}")
        goldtol3T.append(np.nan)

#######################################################################################################################################

# Plots
epsilon = list(np.multiply(tol,-1.0)) # list with log_10 of Tolerance
# Parameters

# Fontsizes
titsz = 24
legsz = 20
labsz = 20
ticsz = 18
# Colors
clr = ["#B30000", "#1A53FF", "#5AD45A", "#ED7F2C"]

# Plot GoldenTol2 vs NB
fig1 = plt.figure(1, figsize=(10,8))
plt.plot(nbs, goldtol2M, label='MPP', color=clr[0], linestyle='dashed')
plt.plot(nbs, goldtol2T, label='PyT', color=clr[1])
plt.title(r'Optimal tolerance for $P_\zeta(k_{CMB})$', fontsize=titsz, pad=10)
plt.xlabel(r'$\text{NB}$', fontsize=labsz, rotation=0)
plt.ylabel(r'$\tilde \varepsilon$', fontsize=labsz, rotation=0, labelpad=10)
plt.xticks(fontsize=ticsz)
plt.xlim(left=1, right=14)
plt.yticks(fontsize=ticsz)
plt.legend(fontsize=legsz, framealpha=1.0)
plt.grid()
plt.savefig('plots/DQ_GT_2pt.pdf', format='pdf',bbox_inches='tight')

# Plot GoldenTol3 vs NB
fig2 = plt.figure(2, figsize=(10,8))
plt.plot(nbs,goldtol3M, label='MPP', color=clr[0], linestyle='dashed')
plt.plot(nbs,goldtol3T, label='PyT', color=clr[1])
plt.title(r'Optimal tolerance for $f_{NL}(k_{CMB})$', fontsize=titsz, pad=10)
plt.xlabel(r'$\text{NB}$', fontsize=labsz)
plt.ylabel(r'$\tilde \varepsilon$', fontsize=labsz, rotation=0, labelpad=10)
plt.xticks(fontsize=ticsz)
plt.xlim(left=1, right=11.5)
plt.yticks(fontsize=ticsz)
plt.legend(fontsize=legsz, framealpha=1.0)
plt.grid()
plt.savefig('plots/DQ_GT_3pt.pdf', format='pdf',bbox_inches='tight')

# Plot Delta(Pz) vs Tol vs NB
# fig3 = plt.figure(3, figsize=(10,8))
# #plt.title('(MPP vs PyT) '+r'$\Delta(P_\zeta)$ vs Tol '+'at different NBs')
# gs = fig3.add_gridspec(9, 3, hspace=0, wspace=0.05)
# ax = plt.gca()
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# axs = gs.subplots(sharex='col', sharey='row')
# for i in range(9):
#     for j in range(3):
#         axs[i,j].plot(epsilon, table[3*i+j,:,8], label='MPP', color=clr[0]) # Plotting MPP Rel(pz) vs epsilon
#         axs[i,j].plot(epsilon, table[3*i+j,:,9], label='PyT', color=clr[1]) # Plotting PyT Rel(pz) vs epsilon
#         axs[i,j].hlines(y=threshold, xmin=-18, xmax=-5, linestyle='dashed', color='black')
#         axs[i,j].text(-8, 0.01, f'NB = {nbs[3*i+j]}', fontsize=8)
#         axs[i,j].set_yscale('log')
#         axs[i,j].tick_params(axis='both', direction='in', labelsize=5)
#         axs[i,j].grid()
#         axs[i,j].legend(fontsize=8)
# fig3.tight_layout(pad=2,h_pad=5,w_pad=5)
# plt.savefig('plots/DQ_DeltaPz.pdf', format='pdf',bbox_inches='tight')

# # Plot Delta(fnl) vs tol vs NB
# fig4 = plt.figure(4, figsize=(10,8))
# #plt.title('(MPP vs PyT) '+r'$\Delta(f_\text{NL})$ vs Tol '+'at different NBs')
# gs = fig4.add_gridspec(9, 3, hspace=0, wspace=0.05)
# ax = plt.gca()
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# axs = gs.subplots(sharex='col', sharey='row')
# for i in range(9):
#     for j in range(3):
#         axs[i,j].plot(epsilon, table[3*i+j,:,10], label='MPP', color=clr[0]) # Plotting MPP Rel(pz) vs epsilon
#         axs[i,j].plot(epsilon, table[3*i+j,:,11], label='PyT', color=clr[1]) # Plotting PyT Rel(pz) vs epsilon
#         axs[i,j].hlines(y=threshold, xmin=-18, xmax=-5, linestyle='dashed', color='black')
#         axs[i,j].text(-8, 0.01, f'NB = {nbs[3*i+j]}', fontsize=8)
#         axs[i,j].set_yscale('log')
#         axs[i,j].tick_params(axis='both', direction='in', labelsize=5)
#         axs[i,j].grid()
#         axs[i,j].legend(fontsize=8)
# fig4.tight_layout(pad=2,h_pad=5,w_pad=5)
# plt.savefig('plots/DQ_DeltafNL.pdf', format='pdf',bbox_inches='tight')

# Plot running time 2pt vs NB
fig5 = plt.figure(5,figsize=(10,8))
x = nbs
y1 = []
y2 = []
for i in range(len(nbs)):
    if np.isnan(goldtol2M[i]):
        y1.append(0)
    else:
        k1 = np.where(epsilon == goldtol2M[i])[0][0]
        y1.append(table[i,k1,0])
    if np.isnan(goldtol2T[i]):
        y2.append(0)
    else:
        k2 = np.where(epsilon == goldtol2T[i])[0][0]
        y2.append(table[i,k2,2])
y1 = np.array(y1)
y2 = np.array(y2)
stop = np.where(y2 == 0)[0][0]
x_ticks = nbs.astype(int)
y_ticks = np.multiply(np.linspace(8.8,9.3,6), 10**2)
plt.title(r'Running time at optimal tolerance for $P_\zeta(k_{CMB})$', fontsize=titsz, pad=10)
plt.semilogy(nbs, y1, label='MPP', color=clr[0], linestyle='dashed')
plt.semilogy(nbs[:stop], y2[:stop], label = 'PyT', color=clr[1])
plt.xlabel(r'$\text{NB}$', fontsize=labsz)
plt.ylabel(r'$t(s)$', fontsize=labsz, rotation=0, labelpad=10)
plt.xticks(fontsize=ticsz)
plt.xlim(left=1, right=14)
plt.yticks(fontsize=ticsz)
plt.legend(fontsize=legsz, framealpha=1.0)
plt.grid()
plt.savefig('plots/DQ_Time_Pz.pdf', format='pdf',bbox_inches='tight')

# Plot running time 3pt vs NB
fig6 = plt.figure(6,figsize=(10,8))
x = nbs
y1 = []
y2 = []
for i in range(len(nbs)):
    if np.isnan(goldtol3M[i]):
        y1.append(0)
    else:
        k1 = np.where(epsilon == goldtol3M[i])[0][0]
        y1.append(table[i,k1,4])
    if np.isnan(goldtol3T[i]):
        y2.append(0)
    else:
        k2 = np.where(epsilon == goldtol3T[i])[0][0]
        y2.append(table[i,k2,6])
y1 = np.array(y1)
y2 = np.array(y2)
print(np.where(y1 == 0))
try:    
    stop1 = np.where(y1 == 0)[0][0]
except IndexError:
    stop1 = -1
stop2 = np.where(y2 == 0)[0][0]
x_ticks = nbs.astype(int)
#y_ticks = np.multiply(np.linspace(8.8,9.3,6), 10**2)
#plt.title('Three-point running time at golden tolerance vs NB', fontsize=titsz)
plt.title(r'Running time at optimal tolerance for $f_{NL}(k_{CMB})$', fontsize=titsz, pad=10)
plt.semilogy(nbs[:stop1], y1[:stop1], label='MPP', color=clr[0], linestyle='dashed')
plt.semilogy(nbs[:stop2], y2[:stop2], label = 'PyT', color=clr[1])
plt.xlabel(r'$\text{NB}$', fontsize=labsz)
plt.ylabel(r'$t(s)$', fontsize=labsz, rotation=0, labelpad=10)
plt.xticks(fontsize=ticsz)
plt.xlim(left=1, right=11.5)
plt.yticks(fontsize=ticsz)
plt.legend(fontsize=legsz, framealpha=1.0)
plt.grid()
plt.savefig('plots/DQ_Time_fNL.pdf', format='pdf',bbox_inches='tight')

# Reduced Delta(pz) vs tol vs NB
fig7 = plt.figure(7, figsize=(10,8))
plt.title(r'$\Delta \mathcal{P}_\zeta(k_{CMB})$', fontsize=titsz, pad=20)
gs = fig7.add_gridspec(3, 2, hspace=0, wspace=0.05)
ind = [2, 6, 10, 14, 18, 22]
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
axs = gs.subplots(sharex='col', sharey='row') 
for i in range(3):
    for j in range(2):
        k = ind[2*i + j]
        axs[i,j].plot(epsilon, table[k,:,8], label='MPP', color=clr[0], linestyle='dashed') # Plotting MPP Rel(pz) vs epsilon
        axs[i,j].plot(epsilon, table[k,:,9], label='PyT', color=clr[1]) # Plotting PyT Rel(pz) vs epsilon
        axs[i,j].hlines(y=threshold, xmin=-18, xmax=-5, linestyle=(0, (10, 5)), color='black')
        axs[i, j].text(0.8, 0.2, f'NB = {nbs[k]}', 
               fontsize=10, 
               bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'),
               transform=axs[i, j].transAxes)  # Use axes coordinates
        axs[i,j].set_yscale('log')
        axs[i,j].set_xlabel(r'$\varepsilon \equiv \log_{10}(Tol)$', fontsize=12)
        axs[i,j].set_xlim(-18, -5)
        axs[i,j].tick_params(axis='both', direction='in', labelsize=8)
        axs[i,j].grid()
        axs[i,j].legend(fontsize=10, framealpha=1.0)
#fig7.tight_layout(pad=3,h_pad=5,w_pad=8)
plt.savefig('plots/DQ_DeltaPz_alt.pdf', format='pdf',bbox_inches='tight')

# Reduced Delta(fnl) vs tol vs NB
fig8 = plt.figure(8, figsize=(10,8))
plt.title(r'$\Delta f_\text{NL}(k_{CMB})$', fontsize=titsz, pad=20)
gs = fig8.add_gridspec(3, 2, hspace=0, wspace=0.05)
ind = [2, 6, 10, 14, 18, 22]
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
axs = gs.subplots(sharex='col', sharey='row')
for i in range(3):
    for j in range(2):
        k = ind[2*i + j]
        axs[i,j].plot(epsilon, table[k,:,10], label='MPP', color=clr[0], linestyle='dashed') # Plotting MPP Rel(pz) vs epsilon
        axs[i,j].plot(epsilon, table[k,:,11], label='PyT', color=clr[1]) # Plotting PyT Rel(pz) vs epsilon
        axs[i,j].hlines(y=threshold, xmin=-18, xmax=-5, linestyle=(0, (10, 5)), color='black')
        axs[i, j].text(0.4, 0.8, f'NB = {nbs[k]}', 
               fontsize=10, 
               bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'),
               transform=axs[i, j].transAxes)
        axs[i,j].set_yscale('log')
        axs[i,j].set_xlim(-18,-5)
        axs[i,j].set_xlabel(r'$\varepsilon \equiv \log_{10}(Tol)$', fontsize=12)
        axs[i,j].tick_params(axis='both', direction='in', labelsize=8)
        axs[i,j].grid()
        axs[i,j].legend(fontsize=10, framealpha=1.0)
#fig8.tight_layout(pad=3,h_pad=5,w_pad=8)
plt.savefig('plots/DQ_DeltafNL_alt.pdf', format='pdf',bbox_inches='tight')

NB = 6.0
NB_index = np.where(nbs == NB)[0][0]

fig9 = plt.figure(9, figsize=(10,8))
#plt.title('(MPP vs PyT) '+r'$\Delta(P_\zeta)$ vs Tol '+f'at NB = {NB}')
plt.plot(epsilon, table[NB_index,:,8], label='MPP', color=clr[0], linestyle='dashed')
plt.plot(epsilon, table[NB_index,:,9], label='PyT', color=clr[1])
plt.hlines(y=threshold, xmin=-18, xmax=-5, linestyle=(0, (10, 5)), color='black')
plt.vlines(x=goldtol2M[NB_index] + 0.5, ymin = 10**-14, ymax = 10**-0, linestyle='dashed', color=clr[0])
plt.vlines(x=goldtol2T[NB_index] + 0.5, ymin = 10**-14, ymax = 10**-0, linestyle='solid', color=clr[1])
plt.xlabel(r'$\epsilon$', fontsize=labsz)
plt.xticks(fontsize=ticsz)
plt.xlim(left=-18, right=-5)
plt.yticks(fontsize=ticsz)
plt.yscale('log')
plt.legend(fontsize=legsz, framealpha=1.0)
plt.grid()
plt.savefig(f'plots/DQ_DeltaPz_{NB}.pdf', format='pdf',bbox_inches='tight')

fig10 = plt.figure(10, figsize=(10,8))
#plt.title('(MPP vs PyT) 2pt Running time vs tol '+f'at NB = {NB}')
plt.plot(epsilon, table[NB_index,:,0], label='MPP', color=clr[0], linestyle='dashed')
plt.plot(epsilon, table[NB_index,:,2], label='PyT', color=clr[1])
plt.vlines(x=goldtol2M[NB_index], ymin=min(table[NB_index,:,0]), ymax=max(table[NB_index,:,2]), color=clr[0], linestyle='dashed')
plt.vlines(x=goldtol2T[NB_index], ymin=min(table[NB_index,:,0]), ymax=max(table[NB_index,:,2]), color=clr[1])
plt.xlabel(r'$\epsilon$', fontsize=labsz)
plt.ylabel(r'$t(s)$', fontsize=labsz, rotation=0)
plt.xticks(fontsize=ticsz)
plt.xlim(left=-18, right=-5)
plt.yticks(fontsize=ticsz)
plt.yscale('log')
plt.legend(fontsize=legsz, framealpha=1.0)
plt.grid()
plt.savefig(f'plots/DQ_2ptTime_{NB}.pdf', format='pdf',bbox_inches='tight') 

fig11 = plt.figure(11, figsize=(10,8))
#plt.title('(MPP vs PyT) '+r'$\Delta(f_\text{NL})$ vs Tol '+f'at NB = {NB}')
plt.plot(epsilon, table[NB_index,:,10], label='MPP', color=clr[0], linestyle='dashed')
plt.plot(epsilon, table[NB_index,:,11], label='PyT', color=clr[1])
plt.hlines(y=threshold, xmin=-18, xmax=-5, linestyle=(0, (10, 5)), color='black')
plt.vlines(x=goldtol3M[NB_index] + 0.25, ymin = 10**-14, ymax = 10**-0, linestyle='dashed', color=clr[0])
plt.vlines(x=goldtol3T[NB_index] + 0.25, ymin = 10**-14, ymax = 10**-0, linestyle='solid', color=clr[1])
plt.xlabel(r'$\epsilon$', fontsize=labsz)
plt.xticks(fontsize=ticsz)
plt.xlim(left=-18, right=-5)
plt.yticks(fontsize=ticsz)
plt.yscale('log')
plt.legend(fontsize=legsz, framealpha=1.0)
plt.grid()
plt.savefig(f'plots/DQ_DeltafNL_{NB}.pdf', format='pdf',bbox_inches='tight')

fig12 = plt.figure(12, figsize=(10,8))
#plt.title('(MPP vs PyT) 3pt Running time vs tol '+f'at NB = {NB}')
plt.plot(epsilon, table[NB_index,:,4], label='MPP', color=clr[0], linestyle='dashed')
plt.plot(epsilon, table[NB_index,:,6], label='PyT', color=clr[1])
plt.vlines(x=goldtol3M[NB_index], ymin=min(table[NB_index,:,4]), ymax=max(table[NB_index,:,6]), color=clr[0], linestyle='dashed')
plt.vlines(x=goldtol3T[NB_index], ymin=min(table[NB_index,:,4]), ymax=max(table[NB_index,:,6]), color=clr[1])
plt.xlabel(r'$\epsilon$', fontsize=labsz)
plt.ylabel(r'$t(s)$', fontsize=labsz, rotation=0)
plt.xticks(fontsize=ticsz)
plt.xlim(left=-18, right=-5)
plt.yticks(fontsize=ticsz)
plt.yscale('log')
plt.legend(fontsize=legsz, framealpha=1.0)
plt.grid()
plt.savefig(f'plots/DQ_3ptTime_{NB}.pdf', format='pdf',bbox_inches='tight')
#######################################################################################################################################
plt.show()