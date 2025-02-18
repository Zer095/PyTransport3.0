import numpy as np
import csv
import os

file_dir = 'Cluster'
#file_dir = os.path.dirname(__file__)

tolerances = np.linspace(5,18,27)
nbs = np.linspace(1, 14, 27)
values = []
for nb in nbs:
    for tol in tolerances:
        values.append([nb,tol])


indices = []
for entry in os.listdir(file_dir):
    if entry.startswith('zzz'):
        string = entry.split('.')[0].replace('z','')
        try:
            num = int(string)
            indices.append(num)
        except ValueError:
            print(f'EXCEPTION: entry = {entry}, string = {string}, type = {type(entry)}, index = {os.listdir(file_dir).index(entry)}, is_digit = {string.isdigit()}')

indices = sorted(indices)

for i in range(len(values)):
    if i not in indices:
        print(f'Missing run {i}, NB = {values[i][0]}, tol = {values[i][1]}')


f = open('Data/NBTotal.csv', 'w')
csvwriter = csv.writer(f)
for i in indices:
    name = os.path.join(file_dir, 'zzz'+str(i)+'.out')
    r = open(name, 'r')
    lines = [line for line in r]
    Nb = float(lines[0])
    Ex = float(lines[1])
    tsm = float(lines[2])
    pzm = float(lines[3])
    tst = float(lines[4])
    pzt = float(lines[5])
    tam = float(lines[6])
    fnlm = float(lines[7])
    tat = float(lines[8])
    fnlt = float(lines[9])
    string = [Nb, Ex, tsm, pzm, tst, pzt, tam, fnlm, tat, fnlt]
    if Nb <= 12:
        csvwriter.writerow(string)

f.close()