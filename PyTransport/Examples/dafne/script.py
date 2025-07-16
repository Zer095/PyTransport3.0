import sympy as sp
import numpy as np
from gravipy import *
import subprocess
#################################################################
from PyTransport.PyTransPy import PyTransSetup, PyTransScripts

PyTransSetup.pathSet()
#############################################################

def str_pot(V):
    # Function that writes the potential in a suitable string for the ModelSetup.py
    params = [arg for arg in sp.postorder_traversal(V) if (arg.is_number and np.abs(arg) != 1)]
    pvalues = []
    for p in params:
        if np.abs(p) not in pvalues:
            pvalues.append(p)
    print(f'Pvalues = {pvalues}')
    params = pvalues
    symbols = [arg for arg in sp.postorder_traversal(V) if arg.is_symbol]
    operations = [arg for arg in sp.postorder_traversal(V) if arg.is_Function]
    symbols = list(set(symbols))

    V_string = str(V)
    # Replace the functions
    for i in range(len(operations)):
        op = 'sym.'+str(operations[i])
        V_string = V_string.replace(str(operations[i]), op)  

    # Replace the parameters
    for i in range(len(params)):
        try:
            formatted_number_str = f"{params[i]:g}"
        except TypeError:
            formatted_number_str = str(params[i])
        
        V_string = V_string.replace(formatted_number_str, f'p[{i}]')

        try:
            formatted_number_str = f"{float(params[i])}"
        except TypeError:
            formatted_number_str = str(params[i])

        V_string = V_string.replace(formatted_number_str, f'p[{i}]')

    # Replace symbols
    for i in range(len(symbols)):
        symbol = str(symbols[i])
        V_string = V_string.replace(symbol, f'f[{i}]')

    nF = len(symbols)
    nP = len(params)
  
    return V_string, nF, nP

def write_setup(name, V_string, nF, nP, G_string=None):
    # Function that writes the ModelSetup.py
    read_file = open('setup_template.py')
    lines = read_file.readlines()

    with open(name+'setup.py', 'w') as write_file:
        for line in lines:
            if line.endswith('#nF\n'):
                write_file.write(f'nF={nF}\n')
            elif line.endswith('#nP\n'):
                write_file.write(f'nP={nP}\n')
            elif line.endswith('#potential\n'):
                write_file.write(f'V = {V_string}\n')
            elif line.endswith('#metric\n') and G_string != None:
                write_file.write(f'G = {G_string}\n')
            elif line.endswith('#metric\n') and G_string == None:
                print('no metric')
            elif line.endswith('#setup\n') and G_string == None:
                write_file.write('PyTransSetup.potential(V,nF,nP,True)\n')
            elif line.endswith('#setup\n') and G_string != None:
                write_file.write('PyTransSetup.potential(V,nF,nP,False,G)\n')
            elif line.endswith('#compile'):
                write_file.write(f'PyTransSetup.compileName("{name}") ')    
            else:
                write_file.write(line)

def computations(PyT, V, initial, sr=True):
    # Function to compute the background evolution of the model and n_s
    import PyTransScripts as PyS
    ############################################## Set initial values ##################################################################
    # Check the number of parameters and the number of fields
    print(f'V = {V}')
    params = [arg for arg in sp.postorder_traversal(V) if (arg.is_number and np.abs(arg) != 1.)]
    pvalues = []
    for p in params:
        if np.abs(p) not in pvalues:
            print(f'Type = {type(p)}, Native p = {p}, np p = {np.float32(p)}, python p = {float(p)}')
            pvalues.append(np.float64(p))
    params = pvalues
    symbols = list(set([arg for arg in sp.postorder_traversal(V) if arg.is_symbol]))
    nF = PyT.nF()
    if len(symbols) != nF:
        print('Error with number of Fields')
        return None
    nP = PyT.nP()
    if len(params) != nP:
        print('Error with number of parameters')
        return None
    
    # Set the parameters' and initial conditions' arrays
    params = np.array(params)
    # if sr = True, the velocity will be computed with Slow-roll conditions
    if sr:
        fields = initial[:nF]
        v = PyT.V(fields, params)
        dv = PyT.dV(fields, params)
        initial = np.concatenate([fields, -dv/np.sqrt(3.*v)])
    else:
        initial = initial
        
    
    ############################################## Background run ##################################################################
    Nstart = 0.
    Nend = 600.
    t = np.linspace(Nstart, Nend, 1000)
    tols = np.array([10**-10,10**-10])
    back = PyT.backEvolve(t, initial, params, tols, True)

    # Make sure the inflation last at least 60 efolds
    print(f'Inflation ends at N = {back[-1, 0]}')
    if back[-1,0] < 60:
        print('Inflation too short')
        return 0

    ############################################## Set 2pt and and compute n_s ##################################################################
    tols = np.array([10**-10,10**-10])
    NB = 6.0
    Nexit = back[-1, 0] - 50.0
    k = PyS.kexitN(Nexit, back, params, PyT)


    Nstart, backExit = PyS.ICsBE(NB, k, back, params, PyT)
    tsig = np.linspace(Nstart, back[-1, 0], 1000)


    PyT_twoPt = PyT.sigEvolve(tsig, k, backExit, params, tols, True)
    pz1 = PyT_twoPt[-1,1]
    Nstart, backExit = PyS.ICsBE(NB, k*(1.001), back, params, PyT)
    tsig = np.linspace(Nstart, back[-1, 0], 1000)
    PyT_twoPt = PyT.sigEvolve(tsig, k*(1.001), backExit, params, tols, True)
    pz2 = PyT_twoPt[-1,1]
    n_s = (np.log(pz2)-np.log(pz1))/(np.log( k*(1.001))-np.log(k)) + 4.0
    return n_s

def main():
    phi = sp.Symbol('x0')
    #V_0 = 2.4078784e-11
    V_0 = 1.071e-10
    #V = V_0*(phi**0.94424653 - (sp.exp(phi*(0.48191538 - phi)) - 0.9756909))
    V = V_0 * (1 - sp.exp(-np.sqrt(2/3)*phi))**2
    # Get the string for the potential, the number of fields and number of parameters
    V_str, nF, nP = str_pot(V)
    # Write the ModelSetup.py
    write_setup('Dafne', V_str, nF, nP)
    # Install the Model
    subprocess.run(['python Dafnesetup.py'], shell=True)
    # Import model
    import PyTransDafne as PyT
    # Define initial conditions
    initial = np.array([6., 0.])
    # Compute n_s
    n_s = computations(PyT, V, initial)
    print(f'n_s = {n_s}')


    
if __name__ == '__main__':
    main()
