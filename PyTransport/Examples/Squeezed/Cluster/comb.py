import numpy as np
import csv
import os

file_dir = 'Cluster'
#file_dir = os.path.dirname(__file__)

values = range(5000)

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
        print(f'Missing run {i}')


f = open('Data/NBTotal.csv', 'w')
csvwriter = csv.writer(f)
for i in indices:
    name = os.path.join(file_dir, 'zzz'+str(i)+'.out')
    r = open(name, 'r')
    lines = [line for line in r]
    factor = float(lines[0])
    ks = float(lines[1])
    pzt = float(lines[2])
    pzm = float(lines[3])
    fnlt = float(lines[4])
    fnlm = float(lines[5])
    string = [factor, ks, pzt, pzm, fnlt, fnlm]

f.close()