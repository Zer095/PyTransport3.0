import csv

f = open("Total.csv", 'w')
NOut = []
PyT_k = []
PyT_pz = []
AN_k = []
AN_pz = []
FA_k = []
FA_pz = []
for i in range(1001):
    name = 'zzz'+str(i)+'.txt'
    r = open(name,'r')
    lines = [line for line in r]
    NOut.append(float(lines[0]))
    PyT_k .append(float(lines[1]))
    PyT_pz.append(float(lines[2]))
    AN_k.append(float(lines[3]))
    AN_pz.append(float(lines[4]))
    FA_k.append(float(lines[5]))
    FA_pz.append(float(lines[6]))

    r.close()

csvwriter = csv.writer(f)
for i in range(1001):
    string = [NOut[i], PyT_k[i], PyT_pz[i], AN_k[i], AN_pz[i], FA_k[i], FA_pz[i] ]
    csvwriter.writerow(string)

f.close()