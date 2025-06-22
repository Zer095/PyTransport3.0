import numpy as np
import matplotlib.pyplot as plt
import csv


# Load Double Quadratic Data
DQ_MPP = 'DQuad/Data/NB_MPP.csv' 
DQ_PyT = 'DQuad/Data/NB_PyT.csv'

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
with open(DQ_MPP, mode='r') as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        i = np.where(nbs == float(line[0]))[0][0]       # NB index
        j = np.where(tol == abs(float(line[1])))[0][0]  # Tol index

        table[i,j,0] = float(line[2])                   # Time 2pt
        table[i,j,1] = float(line[3])                   # Pz
        table[i,j,4] = float(line[4])                   # Time 3pt
        table[i,j,5] = float(line[5])                   # fNL
 
with open(DQ_PyT, mode='r') as file:
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

# Save on separate lists only the files needed
DQ_MPP_gt2 = goldtol2M
DQ_PyT_gt2 = goldtol2T
DQ_MPP_gt3 = goldtol3M
DQ_PyT_gt3 = goldtol3T

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
try:    
    stop1 = np.where(y1 == 0)[0][0]
except IndexError:
    stop1 = -1
stop2 = np.where(y2 == 0)[0][0]

DQ_MPP_time2 = y1[:stop1]
DQ_PyT_time2 = y2[:stop2]

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
try:    
    stop1 = np.where(y1 == 0)[0][0]
except IndexError:
    stop1 = -1
stop2 = np.where(y2 == 0)[0][0]

DQ_MPP_time3 = y1[:stop1]
DQ_PyT_time3 = y2[:stop2]

#############################################################################################################################

# Load Quartic Axion Data

QA_MPP = 'QAxion/Data/NB_MPP.csv'
QA_PyT = 'QAxion/Data/NB_PyT.csv'

tol = np.linspace(5,18,27)
nbs = np.linspace(1, 14, 27)
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

table = np.ones(shape=(27,27,12)) # table(i,j,k): i = NBs, j = Tol, k= tsm, tst, ..
scaling = (1.702050*10.0**(-6.0))**2

# Load data into table
with open(QA_MPP, mode='r') as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        i = np.where(nbs == float(line[0]))[0][0]
        j = np.where(tol == abs(float(line[1])))[0][0]

        table[i,j,0] = float(line[2])
        table[i,j,1] = float(line[3])
        table[i,j,4] = float(line[4])
        table[i,j,5] = float(line[5])
 
with open(QA_PyT, mode='r') as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        i = np.where(nbs == float(line[0]))[0][0]
        j = np.where(tol == abs(float(line[1])))[0][0]
        table[i,j,2] = float(line[2])
        table[i,j,3] = float(line[3])
        table[i,j,6] = float(line[4])
        table[i,j,7] = float(line[5])


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


# Save on separate lists only the data needed
QA_MPP_gt2 = goldtol2M
QA_PyT_gt2 = goldtol2T
QA_MPP_gt3 = goldtol3M
QA_PyT_gt3 = goldtol3T

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
try:    
    stop1 = np.where(y1 == 0)[0][0]
except IndexError:
    stop1 = -1
stop2 = np.where(y2 == 0)[0][0]

QA_MPP_time2 = y1[:stop1]
QA_PyT_time2 = y2[:stop2]

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
try:    
    stop1 = np.where(y1 == 0)[0][0]
except IndexError:
    stop1 = -1
stop2 = np.where(y2 == 0)[0][0]

QA_MPP_time3 = y1[:stop1]
QA_PyT_time3 = y2[:stop2]

#############################################################################################################################

SF_MPP = 'SingleField/Data/NB_MPP.csv'
SF_PyT = 'SingleField/Data/NB_PyT.csv'

tol = np.linspace(5,18,27)
nbs = np.linspace(1, 14, 27)
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

table = np.ones(shape=(27,27,12)) # table(i,j,k): i = NBs, j = Tol, k= tsm, tst, ..
scaling = (1.702050*10.0**(-6.0))**2

# Load data into table
with open(SF_MPP, mode='r') as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        i = np.where(nbs == float(line[0]))[0][0]
        j = np.where(tol == abs(float(line[1])))[0][0]

        table[i,j,0] = float(line[2])
        table[i,j,1] = float(line[3])
        table[i,j,4] = float(line[4])
        table[i,j,5] = float(line[5])
 
with open(SF_PyT, mode='r') as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        i = np.where(nbs == float(line[0]))[0][0]
        j = np.where(tol == abs(float(line[1])))[0][0]
        table[i,j,2] = float(line[2])
        table[i,j,3] = float(line[3])
        table[i,j,6] = float(line[4])
        table[i,j,7] = float(line[5])

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

threshold = 10**-8

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

# Save on separate lists only the data needed
SF_MPP_gt2 = goldtol2M
SF_PyT_gt2 = goldtol2T
SF_MPP_gt3 = goldtol3M
SF_PyT_gt3 = goldtol3T

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
try:    
    stop1 = np.where(y1 == 0)[0][0]
except IndexError:
    stop1 = -1
stop2 = np.where(y2 == 0)[0][0]

SF_MPP_time2 = y1[:stop1]
SF_PyT_time2 = y2[:stop2]

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
try:    
    stop1 = np.where(y1 == 0)[0][0]
except IndexError:
    stop1 = -1
stop2 = np.where(y2 == 0)[0][0]

SF_MPP_time3 = y1[:stop1]
SF_PyT_time3 = y2[:stop2]
#############################################################################################################################

file_name = 'Braglia/Data/NBTotal.csv'

tol = np.linspace(5, 18, 27)
nbs = np.linspace(1, 14, 27)
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
table = np.ones(shape=(27,27,12))

# Load data into table
with open(file_name, mode='r') as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        i = np.where(nbs == float(line[0]))[0][0]       # NB
        j = np.where(tol == abs(float(line[1])))[0][0]  # exp

        table[i,j,0] = float(line[2])                   # tsm
        table[i,j,1] = float(line[3])                   # pzm
        table[i,j,2] = float(line[4])                   # tst
        table[i,j,3] = float(line[5])                   # pzt
        table[i,j,4] = float(line[6])                   # tam
        table[i,j,5] = float(line[7])                   # fnlm
        table[i,j,6] = float(line[8])                   # tat
        table[i,j,7] = float(line[9])                   # fnlt

scaling = 1.0

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

for i in range(27):
    print(f"NB = {nbs[i]}")
    print('MPP')
    print(table[i,:,10])
    print('Trans')
    print(table[i,:,11])
    print('-------------------------------------------------------------')

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

# Save on separate lists only the data needed
LNC_MPP_gt2 = goldtol2M[:23]
LNC_PyT_gt2 = goldtol2T[:17]
LNC_MPP_gt3 = goldtol3M[:23]
LNC_PyT_gt3 = goldtol3T[:18]


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
try:    
    stop1 = np.where(y1 == 0)[0][0]
except IndexError:
    stop1 = -1
stop2 = np.where(y2 == 0)[0][0]

LNC_MPP_time2 = y1[:23]
LNC_PyT_time2 = y2[:17]

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
try:    
    stop1 = np.where(y1 == 0)[0][0]
except IndexError:
    stop1 = -1
stop2 = np.where(y2 == 0)[0][0]

LNC_MPP_time3 = y1[:23]
LNC_PyT_time3 = y2[:16]


print(f'Len MPP gt2 = {len(LNC_MPP_gt2)}, Len PyT gt2 = {len(LNC_PyT_gt2)}')
print(f'Len MPP T2 = {len(LNC_MPP_time2)}, Len PyT T2 = {len(LNC_PyT_time2)}')
print(f'Len MPP g3 = {len(LNC_MPP_gt3)}, Len PyT G3 = {len(LNC_PyT_gt3)}')
print(f'Len MPP T3 = {len(LNC_MPP_time3)}, Len PyT T3 = {len(LNC_PyT_time3)}')
#############################################################################################################################
#Plot parameter
# Fontsizes
titsz = 16
legsz = 20
labsz = 12
ticsz = 18
# Colors
clr = ["#B30000", "#1A53FF", "#5AD45A", "#ED7F2C"] 
# Names
model_names = ['double inflation', 'axion-quartic', 'single field with features', 'two-field model with\n curved field space']
# Dic with variables
variables2 = {
    'DQ': [DQ_MPP_gt2,DQ_PyT_gt2,DQ_MPP_time2,DQ_PyT_time2],
    'QA': [QA_MPP_gt2,QA_PyT_gt2,QA_MPP_time2,QA_PyT_time2],
    'SF': [SF_MPP_gt2,SF_PyT_gt2,SF_MPP_time2,SF_PyT_time2],
    'LNC': [LNC_MPP_gt2,LNC_PyT_gt2,LNC_MPP_time2,LNC_PyT_time2],
    }

variables3 = {
    'DQ': [DQ_MPP_gt3,DQ_PyT_gt3,DQ_MPP_time3,DQ_PyT_time3],
    'QA': [QA_MPP_gt3,QA_PyT_gt3,QA_MPP_time3,QA_PyT_time3],
    'SF': [SF_MPP_gt3,SF_PyT_gt3,SF_MPP_time3,SF_PyT_time3],
    'LNC': [LNC_MPP_gt3,LNC_PyT_gt3,LNC_MPP_time3,LNC_PyT_time3],
    }

models = ['DQ', 'QA', 'SF', 'LNC']

#P_\zeta plot
fig1, ax1 = plt.subplots(nrows=4, ncols=2, figsize=(8,10), sharex='col')
for i, model in enumerate(models):
    data = variables2[model]
    # Titles
    ax1[0, 0].set_title(r'Optimal tolerance $\tilde \varepsilon$'+' for' + r' $\mathcal{P}_\zeta(k_{CMB})$', fontsize=titsz)
    ax1[0, 1].set_title(r'Running time in seconds at $\tilde \varepsilon$', fontsize=titsz)

    #Legend
    ax1[0, 0].legend()

    # Left panel
    ax1[i, 0].plot(nbs[:len(data[0])], data[0], label='MPP', color=clr[0], linestyle='dashed')
    ax1[i, 0].plot(nbs[:len(data[1])], data[1], label='PyT', color=clr[1])
    ax1[i, 0].grid()
    ax1[i, 0].set_ylabel(model_names[i], fontsize = labsz)

    # Right panel
    ax1[i, 1].plot(nbs[:len(data[2])], data[2], label='MPP', color=clr[0], linestyle='dashed')
    ax1[i, 1].plot(nbs[:len(data[3])], data[3], label='PyT', color=clr[1])
    ax1[i, 1].grid()
    ax1[i, 1].set_yscale('log')

    # Labels
    ax1[3, 0].set_xlabel('NB', fontsize=labsz)
    ax1[3, 1].set_xlabel('NB', fontsize=labsz)
    ax1[3, 0].set_xlim(1,14)
    ax1[3, 1].set_xlim(1, 14)
plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0.2)
plt.savefig('Tot_2pt.pdf', format='pdf',bbox_inches='tight')



# f_NL plot
fig2, ax2 = plt.subplots(nrows=4, ncols=2, figsize=(8,10), sharex='col')
for i, model in enumerate(models):
    data = variables3[model]
    #Titles
    ax2[0, 0].set_title(r'Optimal tolerance $\tilde \varepsilon$'+' for' + r' $f_{NL}(k_{CMB})$', fontsize=titsz)
    ax2[0, 1].set_title(r'Running time in seconds at $\tilde \varepsilon$', fontsize=titsz)

    # legend
    ax2[0, 0].legend()
    # Left panel
    ax2[i, 0].plot(nbs[:len(data[0])], data[0], label='MPP', color=clr[0], linestyle='dashed')
    ax2[i, 0].plot(nbs[:len(data[1])], data[1], label='PyT', color=clr[1])
    ax2[i, 0].grid()
    ax2[i, 0].set_ylabel(model_names[i], fontsize = labsz)

    # Right panel
    ax2[i, 1].plot(nbs[:len(data[2])], data[2], label='MPP', color=clr[0], linestyle='dashed')
    ax2[i, 1].plot(nbs[:len(data[3])], data[3], label='PyT', color=clr[1])
    ax2[i, 1].grid()
    ax2[i, 1].set_yscale('log')

    # Labels
    ax2[3, 0].set_xlabel('NB', fontsize=labsz)
    ax2[3, 1].set_xlabel('NB', fontsize=labsz)
    ax2[3, 0].set_xlim(1,14)
    ax2[3, 1].set_xlim(1, 14)

plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0.2)
plt.savefig('Tot_3pt.pdf', format='pdf',bbox_inches='tight')

plt.show()


