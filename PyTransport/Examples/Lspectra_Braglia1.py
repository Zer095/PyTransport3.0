{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
import matplotlib.pyplot as plt\
import csv\
\
\
from PyTransport.PyTransPy import PyTransSetup\
from PyTransport.PyTransPy import PyTransScripts as PyS\
\
import PyTransLNC as PyT\
\
##################################### Set initial values ######################################################\
\
nF = PyT.nF()\
nP = PyT.nP()\
\
\
fields = np.array([7., 7.31])\
pvalue = np.zeros(nP)\
\
pvalue[0] = 1.0;\
pvalue[1] = np.sqrt(6.); pvalue[2] = 1/500.0; pvalue[3] = 7.8\
\
V = PyT.V(fields, pvalue)\
dV = PyT.dV(fields, pvalue)\
initial = np.concatenate((fields, np.array([0,0])))\
\
################################# Run Background ####################################################\
\
Nstart = 0.\
Nend = 85.6\
t = np.linspace(Nstart, Nend, 10**4)\
tols = np.array([10e-10,10e-10])\
back = PyT.backEvolve(t, initial, pvalue, tols, True)\
\
\
############################################## Set PyT parameters ##################################################################\
NB = 6.0\
tols = np.array([10**-10,10**-10])\
\
\
PyT_pzOut= np.array([])\
\
kOut = np.array([])\
PhiOut = np.array([])\
NOut = np.array([])\
\
for i in range(601):\
    NExit = 10 + i*0.073\
    NOut = np.append(NOut, NExit)\
    k = PyS.kexitN(NExit, back, pvalue, PyT)\
    kOut = np.append(kOut, k)\
\
zzOut, times = PyS.pSpectra(kOut, back, pvalue, NB, tols, PyT)\
\
#PyT_fnlOut = 5./6*zzzOut/(3.0*zzOut**2)\
PyT_pzOut = (kOut**3/(2*np.pi**2))*zzOut\
\
#zzOut, zzzOut, times = PyS.MPPSpectra(kOut, back, pvalue, NB, tols, PyT)\
\
#MPP_fnlOut = 5./6*zzzOut/(3.0*zzOut**2)\
#MPP_pzOut = (kOut**3/(2*np.pi**2))*zzOut\
\
with open('Data/LNCspectra.csv', 'w') as file:\
    csvwriter = csv.writer(file)\
    for i in range(401):\
        row = [NOut[i], kOut[i], PyT_pzOut[i]]\
        csvwriter.writerow(row) \
}