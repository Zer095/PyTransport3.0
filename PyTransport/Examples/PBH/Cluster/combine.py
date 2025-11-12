import csv

f = open("Total.csv", 'w')
Ns = []
Ks = []
MPP_Pz = []
PyT_Pz = []
MPP_fNL = []
PyT_fNL = []
for i in range(401):
    name = 'zzz'+str(i)+'.txt'
    r = open(name,'r')
    lines = [line for line in r]
    Ns.append(float(lines[0]))
    Ks.append(float(lines[1]))
    MPP_Pz.append(float(lines[2]))
    PyT_Pz.append(float(lines[3]))
    MPP_fNL.append(float(lines[4]))
    PyT_fNL.append(float(lines[5]))

    r.close()

csvwriter = csv.writer(f)
header = ['Nout', 'K', 'MPP Pz', 'PyT Pz', 'MPP fNL', 'PyT fNL']
csvwriter.writerow(header)
for i in range(401):
    string = [Ns[i], Ks[i], MPP_Pz[i], PyT_Pz[i], MPP_fNL[i], PyT_fNL[i]]
    csvwriter.writerow(string)

f.close()