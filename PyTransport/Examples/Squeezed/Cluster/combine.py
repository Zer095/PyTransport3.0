import csv
from os import listdir, getcwd
from os.path import isfile, join

mypath = getcwd()

f = open("Total.csv", 'w')
csvwriter = csv.writer(f)
Ks = []
MPP_fNL = []
PyT_fNL = []

files = [int(f[3:-4]) for f in listdir(mypath) if isfile(join(mypath, f)) and f[0:3] == 'zzz']

num = max(files)

for i in range(num):
    name = 'zzz'+str(i)+'.out'
    r = open(name,'r')
    lines = [line for line in r]
    string = [float(lines[0]), float(lines[1]), float(lines[2]), float(lines[3]), float(lines[4]), float(lines[5])]
    csvwriter.writerow(string)
    r.close()

f.close()