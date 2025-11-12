#This file is part of PyTransport.

#PyTransport is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#PyTransport is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with PyTransport.  If not, see <http://www.gnu.org/licenses/>.


# This file contains python scripts used to setup the complided PyTrans module

import sympy as sym
import numpy as np
from pathlib import Path
import subprocess
import sys
import os
import shutil
import time
from gravipy.tensorial import *



def directory(NC):
    dir = Path(os.path.dirname(__file__)).parent.absolute()
    filename = os.path.join(dir, 'PyTransCpp/PyTrans.cpp')
    f = open(filename,"r")
    lines = f.readlines()
    f.close()
    f = open(filename,"w")
    if NC==False:
        for line in lines:
            if not  line.endswith("//evolve\n") and not line.endswith("//moments\n") and not line.endswith("//model\n") and not line.endswith("//stepper\n"):
                f.write(line)
            if line.endswith("//evolve\n"):
                fileT = os.path.join(dir, 'PyTransCpp/cppsrc/evolve.h')
                f.write('#include' + '"'+ fileT +'"' + '//evolve' +'\n')
            if line.endswith("//moments\n"):
                fileT = os.path.join(dir, 'PyTransCpp/cppsrc/moments.h')
                f.write('#include' + '"'+ fileT +'"' + '//moments' +'\n')
            if line.endswith("//model\n"):
                fileT = os.path.join(dir, 'PyTransCpp/cppsrc/model.h')
                f.write('#include' + '"'+ fileT +'"' + '//model' +'\n')
            if line.endswith("//stepper\n"):
                fileT = os.path.join(dir, 'PyTransCpp/cppsrc/stepper/rkf45.hpp')
                f.write('#include' + '"'+ fileT +'"' + '//stepper' +'\n')
    else:
        for line in lines:
            if not  line.endswith("//evolve\n") and not line.endswith("//moments\n") and not line.endswith("//model\n") and not line.endswith("//stepper\n"):
                f.write(line)
            if line.endswith("//evolve\n"):
                fileT = os.path.join(dir, 'PyTransCpp/cppsrc/NC/evolve.h')
                f.write('#include' + '"'+ fileT +'"' + '//evolve' +'\n')
            if line.endswith("//moments\n"):
                fileT = os.path.join(dir, 'PyTransCpp/cppsrc/NC/moments.h')
                f.write('#include' + '"'+ fileT +'"' + '//moments' +'\n')
            if line.endswith("//model\n"):
                fileT = os.path.join(dir, 'PyTransCpp/cppsrc/NC/model.h')
                f.write('#include' + '"'+ fileT +'"' + '//model' +'\n')
            if line.endswith("//stepper\n"):
                fileT = os.path.join(dir, 'PyTransCpp/cppsrc/stepper/rkf45.hpp')
                f.write('#include' + '"'+ fileT +'"' + '//stepper' +'\n')
    f.close()

def pathSet():
    dir = os.path.dirname(__file__)
    parent_dir = Path(dir).resolve().parents[4]

    path1 = os.path.join(parent_dir, 'PyTransport/PyTransPy')

    path2 = os.path.join(parent_dir, 'PyTransport/PyTransCpp')

    
    sys.path.append(dir)
    sys.path.append(path1)
    sys.path.append(path2)

    # Add all subfolders to PATH
    sub = os.walk(path1)
    for x in sub:
        fold = os.path.join(path1, x[0])
        sys.path.append(fold)
    

def compileName(name, NC=False):
    directory(NC)
    dir = Path(os.path.dirname(__file__)).parent.absolute()
    location = str(dir)
    filename1 = os.path.join(dir, 'setup.py')
    f = open(filename1, "r")
    lines = f.readlines()
    f.close()
    f = open(filename1,"w")
    name_module = 'PyTrans'+name
    result = subprocess.run(['pip','uninstall','-y', name_module], check=True, capture_output=True, text=True) 
    print(result.stdout)
    print(result.stderr)
    extension1 = os.path.join(dir, 'PyTransCpp/PyTrans.cpp')
    extension2 = os.path.join(dir, 'PyTransCpp/cppsrc/stepper/rkf45.cpp')
    sys.path.append(os.path.join(dir, 'PyTransCpp/cppsrc'))
    include_dir = [os.path.join(location, 'PyTransCpp'),os.path.join(location, 'PyTransCpp/cppsrc/stepper'), np.get_include()]

    ln = iter(lines) 
    # Write on setup.py
    for line in ln:
        if line.startswith('# Define the extension module'):
            f.write(line)
            f.write(name_module+' = Extension(\n')
            f.write(f'    "{name_module}",\n')
            f.write(f'    sources=[r"{extension1}",r"{extension2}"],\n')
            f.write(f'    include_dirs = {include_dir},\n')
            f.write('    #extra_compile_args=compile_args,\n')
            next(ln, None)
            next(ln, None)
            next(ln, None)
            next(ln, None)
            next(ln, None)
        elif line.startswith('# Setup configuration'):
            f.write(line)
            f.write('setup(\n')
            f.write(f'    name="{name_module}",\n')
            f.write(f'    ext_modules=[{name_module}],\n')
            f.write('    package_data={\n')
            f.write(f'        "{name_module}": [numpy.get_include()],\n')
            next(ln, None)
            next(ln, None)
            next(ln, None)
            next(ln, None)
            next(ln, None)


        else:
            f.write(line)

    f.close()

    f = open(extension1,"r")
    lines = f.readlines()
    f.close()
    f = open(extension1,"w")
    for line in lines:
        if not  line.endswith("//FuncDef\n") and not line.endswith("//initFunc\n") and not line.endswith("//modDef\n"):
            f.write(line)
        if line.endswith("//FuncDef\n"):
            f.write('static PyMethodDef PyTrans'+name+'_methods[] = {{"H", (PyCFunction)MT_H,    METH_VARARGS, PyTrans_docs},{"Ep", (PyCFunction)MT_Ep,    METH_VARARGS, PyTrans_docs},{"Eta", (PyCFunction)MT_Eta,    METH_VARARGS, PyTrans_docs},{"nF", (PyCFunction)MT_fieldNumber,        METH_VARARGS, PyTrans_docs},{"nP", (PyCFunction)MT_paramNumber,        METH_VARARGS, PyTrans_docs},{"V", (PyCFunction)MT_V,            METH_VARARGS, PyTrans_docs},{"dV", (PyCFunction)MT_dV,                METH_VARARGS, PyTrans_docs},  {"ddV", (PyCFunction)MT_ddV,                METH_VARARGS, PyTrans_docs},  {"backEvolve", (PyCFunction)MT_backEvolve,        METH_VARARGS, PyTrans_docs},  {"sigEvolve", (PyCFunction)MT_sigEvolve,        METH_VARARGS, PyTrans_docs},  {"gamEvolve", (PyCFunction)MT_gamEvolve,        METH_VARARGS, PyTrans_docs},    {"alphaEvolve", (PyCFunction)MT_alphaEvolve,        METH_VARARGS, PyTrans_docs}, {"MPP2", (PyCFunction)MT_MPP2,        METH_VARARGS, PyTrans_docs}, {"MPPSigma", (PyCFunction)MT_MPPSigma,        METH_VARARGS, PyTrans_docs}, {"MPP3", (PyCFunction)MT_MPP3,        METH_VARARGS, PyTrans_docs},{"MPPAlpha", (PyCFunction)MT_MPPAlpha,        METH_VARARGS, PyTrans_docs},{NULL, NULL, 0, NULL}};//FuncDef\n')

        if line.endswith("//modDef\n"):
            f.write('static struct PyModuleDef PyTransModule = {PyModuleDef_HEAD_INIT, "PyTrans'+name+'", PyTrans_docs, -1, PyTrans'+name+'_methods}; //modDef\n')

        if line.endswith("//initFunc\n"):
            f.write('PyMODINIT_FUNC PyInit_PyTrans'+name+'(void)    {    PyObject *m = PyModule_Create(&PyTransModule); import_array(); return m;} //initFunc\n')
    f.close()
 
    loc = '../../'
    loc = Path(os.path.dirname(__file__)).resolve().parents[0]

    # os.system("export CFLAGS='-I /Users/apx050/Desktop/Projects/PyTransport/env/lib/python3.11/site-packages/numpy/core/include/'")
    # os.system("export CXX=gcc")
    subprocess.run(['python -m build'], cwd=loc, shell=True)
    subprocess.run(['pip install -v .'], cwd=loc, shell=True)
    sys.path.append(location+"lib/python/")
    sys.path.append(location+"../PyTransScripts")
    # Remove .egg-info folder
    try:
        shutil.rmtree(os.path.join(loc, name_module+'.egg-info'))  
    except FileNotFoundError:
        print(f"File {os.path.join(loc, name_module+'.egg-info')} not found")
    # Remove build folder
    try:       
        shutil.rmtree(os.path.join(loc, 'build'))
    except FileNotFoundError:
        print(f"File {os.path.join(loc, 'build')} not found")
    # Remove dist folder
    try:
        shutil.rmtree(os.path.join(loc, 'dist'))
    except FileNotFoundError:
        print(f'File {os.path.join(loc, "dist")} not found')


def deleteModule(name):
    location = os.path.join(dir, 'PyTrans/')
    [os.remove(os.path.join(location,f)) for f in os.listdir(location) if f.startswith("PyTrans"+name)]


def write_cse_decls(decls, g, nF, nP):
    # emit declarations for common subexpressions
    for rule in decls:
        symb = sym.printing.cxxcode(rule[0])
        expr = sym.printing.cxxcode(rule[1])
        new_expr = rewrite_indices(expr, nF, nP)
        g.write('  double ' + symb + ' = ' + new_expr + ';\n')


def rewrite_indices(expr, nF, nP):
    new_expr = expr

    for l in range(max(nP, nF)):
        l = max(nP, nF) - 1 - l
        new_expr = new_expr.replace("_" + str(l), "[" + str(l) + "]")

    return new_expr

def tol(rtol, atol):
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, 'PyTrans', 'PyTrans.cpp')
    f = open(filename,"r")  

def potential(V,nF,nP,simple=False,G=0,silent=True):
    f=sym.symarray('f',nF)
    p=sym.symarray('p',nP)
 
    vd=sym.symarray('vd',nF)
    vdd=sym.symarray('vdd',nF*nF)
    vddd=sym.symarray('vddd',nF*nF*nF)

    if not silent:
        timer = time.process_time()
        print('[{time}] computing symbolic potential derivatives'.format(time=time.ctime()))

    if G!=0:

        if not silent:
            timer2 = time.process_time()
            print('  [{time}] computing curvature quantities'.format(time=time.ctime()))

        g, Ga, Ri, Rm =fieldmetric(G,nF,nP,simple=simple,silent=silent)

        if not silent:
            print('  [{time}] complete in {x} sec'.format(time=time.ctime(), x=time.process_time() - timer2))

        FMP=0
        for i in range(nF):
            if simple==True:
                vd[i] = sym.simplify(V.diff(f[i]))
            else:
                vd[i] = V.diff(f[i])
        for i in range(nF):
            for j in range(nF):
                for l in range(nF):
                    FMP=FMP+Ga(-(l+1),i+1,j+1) * vd[l]
                if simple==True:	
                    vdd[i+j*nF] = sym.simplify(V.diff(f[i]).diff(f[j])-FMP)
                else:
                    vdd[i+j*nF] = V.diff(f[i]).diff(f[j])-FMP
                FMP=0
        for i in range(nF):
            for j in range(nF):
                for k in range(nF):
                    for l in range(nF):
                       FMP=FMP+Ga(-(l+1),i+1,k+1)*vdd[l+j*nF] + Ga(-(l+1),j+1,k+1)*vdd[i+l*nF] +sym.expand(Ga(-(1+l),1+i,1+j)).diff(f[k])*vd[l]+sym.expand(Ga(-(1+l),1+i,j+1))*vd[l].diff(f[k])#	+sym.expand(Ga(-(l+1),i+1,j+1)).diff(f[k]) * vd[l] +Ga(-(l+1),i+1,j+1)* (sym.expand(vd[l]).diff(f[k])) 				
                    if simple==True:
                        vddd[i+j*nF+k*nF*nF] =sym.simplify(V.diff(f[i]).diff(f[j]).diff(f[k]) -FMP)
                    else:
                        vddd[i+j*nF+k*nF*nF] =V.diff(f[i]).diff(f[j]).diff(f[k]) -FMP
                    FMP=0
    else:
        for i in range(nF):
            if simple==True:
                vd[i] = sym.simplify(V.diff(f[i]))
            else:
                vd[i] = V.diff(f[i])
            for j in range(nF):
                if simple==True:
                      vdd[i+j*nF] = sym.simplify(V.diff(f[i]).diff(f[j]) )
                else:
                      vdd[i+j*nF] = V.diff(f[i]).diff(f[j])
                for k in range(nF):
                    if simple==True:
                        vddd[i+j*nF+k*nF*nF] = sym.simplify(V.diff(f[i]).diff(f[j]).diff(f[k]))
                    else:
                        vddd[i+j*nF+k*nF*nF] = V.diff(f[i]).diff(f[j]).diff(f[k])

    if not silent:
        print('[{time}] complete in {x} sec'.format(time=time.ctime(), x=time.process_time() - timer))

    import os
    dir = Path(os.path.dirname(__file__)).parent.absolute()
    filename1 = os.path.join(dir, 'PyTransCpp/cppsrc', 'potentialProto.h')
    filename2 = os.path.join(dir, 'PyTransCpp/cppsrc', 'potential.h')
    f = open(filename1, 'r')
    g = open(filename2, 'w')

    if not silent:
        timer = time.process_time()
        print('[{time}] writing to potential.h'.format(time=time.ctime()))

    for line in f: 

        g.write(line) 

        if line == "// #Rewrite\n":
            g.write('// Potential file rewriten at' + ' ' + time.strftime("%c") +'\n')

        if line == "// #FP\n":
            g.write('nF='+str(nF)+';\n'+'nP='+str(nP)+';\n')

        if line == "// Pot\n":

            # extract common subexpressions from V
            if not silent:
                timer_cse = time.process_time()
                print('  [{time}] performing CSE for V'.format(time=time.ctime()))
            decls, new_expr = sym.cse(V, order='none')
            if not silent:
                print('  [{time}] complete in {x} sec'.format(time=time.ctime(), x=time.process_time() - timer_cse))

            # emit declarations for CSE variables
            write_cse_decls(decls, g, nF, nP)

            # emit main expression
            emit_expr = sym.printing.cxxcode(new_expr[0])
            rw_expr = rewrite_indices(emit_expr, nF, nP)
            g.write('  sum='+str(rw_expr)+';\n')

        if line == "// dPot\n":

            if not silent:
                timer_cse = time.process_time()
                print('  [{time}] performing CSE for dV'.format(time=time.ctime()))
            decls, new_exprs = sym.cse(vd, order='none')
            if not silent:
                print('  [{time}] complete in {x} sec'.format(time=time.ctime(), x=time.process_time() - timer_cse))

            # emit declarations for CSE variables
            write_cse_decls(decls, g, nF, nP)

            for i in range(nF):

                emit_expr = sym.printing.cxxcode(new_exprs[i])
                rw_expr = rewrite_indices(emit_expr, nF, nP)
                g.write('\n sum[' + str(i) + ']=' + str(rw_expr) + ';\n')

        if line == "// ddPot\n":

            if not silent:
                timer_cse = time.process_time()
                print('  [{time}] performing CSE for ddV'.format(time=time.ctime()))
            decls, new_exprs = sym.cse(vdd, order='none')
            if not silent:
                print('  [{time}] complete in {x} sec'.format(time=time.ctime(), x=time.process_time() - timer_cse))

            # emit declarations for CSE variables
            write_cse_decls(decls, g, nF, nP)

            for i in range(nF):
                for j in range(nF):
                    emit_expr = sym.printing.cxxcode(new_exprs[i + nF * j])
                    rw_expr = rewrite_indices(emit_expr, nF, nP)
                    g.write('\n sum[' + str(i + nF * j) + ']=' + str(rw_expr) + ';\n')

        if line == "// dddPot\n":

            if not silent:
                timer_cse = time.process_time()
                print('  [{time}] performing CSE for dddV'.format(time=time.ctime()))
            decls, new_exprs = sym.cse(vddd, order='none')
            if not silent:
                print('  [{time}] complete in {x} sec'.format(time=time.ctime(), x=time.process_time() - timer_cse))

            # emit declarations for CSE variables
            write_cse_decls(decls, g, nF, nP)

            for i in range(nF):
                for j in range(nF):
                    for k in range(nF):

                        emit_expr = sym.printing.cxxcode(new_exprs[i + nF * j + nF * nF * k])
                        rw_expr = rewrite_indices(emit_expr, nF, nP)
                        g.write('\n sum[' + str(i + nF * j + nF * nF * k) + ']=' + str(rw_expr) + ';\n')

    if not silent:
        print('[{time}] complete in {x} sec'.format(time=time.ctime(), x=time.process_time() - timer))

    g.close()
    f.close()

def fieldmetric(G,nF,nP,simple=False,silent=True):
    f=sym.symarray('f',nF)
    p=sym.symarray('p',nP)

    COR = Coordinates(r'\chi', f)
    g = MetricTensor('g',COR , G)
    Ga = Christoffel('Ga', g)
    Ri = Ricci('Ri', g)
    Rm = Riemann('Rm',g)

    import os
    from importlib.resources import files

    pathSet()

    #dir = Path(os.path.dirname(__file__)).parent.absolute()
    dir = Path(os.path.dirname(__file__)).resolve().parents[0]

    filename1 = os.path.join(dir, 'PyTransCpp/cppsrc', 'fieldmetricProto.h')
    filename2 = os.path.join(dir, 'PyTransCpp/cppsrc', 'fieldmetric.h')
    # filename1 = files('PyTransCpp.cppsrc').joinpath( 'fieldmetricProto.h' )
    # filename2 = files( 'PyTransCpp.cppsrc').joinpath( 'fieldmetric.h' )
    e = open(filename1, 'r')
    h = open(filename2, 'w')

    G_array = sym.symarray('G', 2*nF * 2*nF)
    Gamma_array = sym.symarray('Gamma', 2*nF * 2*nF * 2*nF)
    R_array = sym.symarray('Riemann', nF*nF*nF*nF)
    gradR_array = sym.symarray('gradRiemann', nF*nF*nF*nF*nF)

    # populate Riemann matrix
    for i in range(2 * nF):
        for j in range(2 * nF):
            if i < nF:
                ii = -i - 1
            else:
                ii = i - (nF - 1)
            if j < nF:
                jj = -j - 1
            else:
                jj = j - (nF - 1)

            if simple is True:
                G_array[(2 * nF) * i + j] = sym.simplify(g(ii, jj))
            else:
                G_array[(2 * nF) * i + j] = g(ii, jj)

    # populate connexion matrix
    for i in range(2 * nF):
        for j in range(2 * nF):
            for k in range(2 * nF):
                if i < nF:
                    ii = -i - 1
                else:
                    ii = i - (nF - 1)
                if j < nF:
                    jj = -j - 1
                else:
                    jj = j - (nF - 1)
                if k < nF:
                    kk = -k - 1
                else:
                    kk = k - (nF - 1)

                if kk < 0 or jj < 0 or ii > 0:
                    Gamma_array[(2*nF)*(2*nF)*i+(2*nF)*j+k] = sym.simplify(0)
                else:
                    if simple is True:
                        Gamma_array[(2*nF)*(2*nF)*i+(2*nF)*j+k] = sym.simplify(Ga(ii, jj, kk))
                    else:
                        Gamma_array[(2*nF)*(2*nF)*i+(2*nF)*j+k] = Ga(ii, jj, kk)

    # populate Riemann matrix
    for i in range(nF):
        for j in range(nF):
            for k in range(nF):
                for l in range(nF):
                    ii=i+1
                    jj=j+1
                    kk=k+1
                    ll=l+1

                    if simple is True:
                        R_array[(nF)*(nF)*(nF)*i+(nF)*(nF)*j+(nF)*k+l] = sym.simplify(Rm(ii,jj,kk,ll))
                    else:
                        R_array[(nF)*(nF)*(nF)*i+(nF)*(nF)*j+(nF)*k+l] = Rm(ii,jj,kk,ll)

    # populate covariant-derivative of Riemann matrix
    for i in range(nF):
        for j in range(nF):
            for k in range(nF):
                for l in range(nF):
                    for m in range(nF):
                        ii = i + 1
                        jj = j + 1
                        kk = k + 1
                        ll = l + 1
                        mm = m + 1

                        if simple is True:
                            gradR_array[(nF)*(nF)*(nF)*(nF)*i+(nF)*(nF)*(nF)*j+(nF)*(nF)*k+(nF)*l+m] = sym.simplify(Rm.covariantD(ii, jj, kk, ll, mm))
                        else:
                            gradR_array[(nF)*(nF)*(nF)*(nF)*i+(nF)*(nF)*(nF)*j+(nF)*(nF)*k+(nF)*l+m] = Rm.covariantD(ii, jj, kk, ll, mm)

    for line in e:
        h.write(line)
        if line == "// #FP\n":
            #h.write('nF='+str(nF)+';\n')
            h.write('nF='+str(nF)+';\n'+'nP='+str(nP)+';\n')

        if line == "// metric\n":

            if not silent:
                timer_cse = time.process_time()
                print('    [{time}] performing CSE for field metric'.format(time=time.ctime()))
            decls, new_expr = sym.cse(G_array, order='none')
            if not silent:
                print('    [{time}] complete in {x} sec'.format(time=time.ctime(), x=time.process_time() - timer_cse))

            # emit declarations for CSE variables
            write_cse_decls(decls, h, nF, nP)

            for i in range(2*nF):
                for j in range(2*nF):
                    # emit main expression
                    emit_expr = sym.printing.cxxcode(new_expr[(2*nF)*i+j])
                    rw_expr = rewrite_indices(emit_expr, nF, nP)
                    h.write('\n FM['+str((2*nF)*i+j)+']=' + str(rw_expr) + ';\n')

        if line == "// Christoffel\n":

            if not silent:
                timer_cse = time.process_time()
                print('    [{time}] performing CSE for Christoffel symbols'.format(time=time.ctime()))
            decls, new_expr = sym.cse(Gamma_array, order='none')
            if not silent:
                print('    [{time}] complete in {x} sec'.format(time=time.ctime(), x=time.process_time() - timer_cse))

            # emit declarations for CSE variables
            write_cse_decls(decls, h, nF, nP)

            for i in range(2 * nF):
                for j in range(2 * nF):
                    for k in range(2 * nF):
                        # emit main expression
                        emit_expr = sym.printing.cxxcode(new_expr[(2*nF)*(2*nF)*i+(2*nF)*j+k])
                        rw_expr = rewrite_indices(emit_expr, nF, nP)
                        h.write('\n CS['+str((2*nF)*(2*nF)*i+(2*nF)*j+k)+']=' + str(rw_expr) + ';\n')
    
        if line == "// Riemann\n":

            if not silent:
                timer_cse = time.process_time()
                print('    [{time}] performing CSE for Riemann tensor'.format(time=time.ctime()))
            decls, new_expr = sym.cse(R_array, order='none')
            if not silent:
                print('    [{time}] complete in {x} sec'.format(time=time.ctime(), x=time.process_time() - timer_cse))

            # emit declarations for CSE variables
            write_cse_decls(decls, h, nF, nP)

            for i in range(nF):
                for j in range(nF):
                    for k in range(nF):
                        for l in range(nF):
                            # emit main expression
                            emit_expr = sym.printing.cxxcode(new_expr[(nF)*(nF)*(nF)*i+(nF)*(nF)*j+(nF)*k+l])
                            rw_expr = rewrite_indices(emit_expr, nF, nP)
                            h.write('\n RM['+str((nF)*(nF)*(nF)*i+(nF)*(nF)*j+(nF)*k+l)+']=' + str(rw_expr) + ';\n')
       
        if line == "// Riemanncd\n":

            if not silent:
                timer_cse = time.process_time()
                print('    [{time}] performing CSE for Riemann tensor'.format(time=time.ctime()))
            decls, new_expr = sym.cse(gradR_array, order='none')
            if not silent:
                print('    [{time}] complete in {x} sec'.format(time=time.ctime(), x=time.process_time() - timer_cse))

            # emit declarations for CSE variables
            write_cse_decls(decls, h, nF, nP)

            for i in range(nF):
                for j in range(nF):
                    for k in range(nF):
                        for l in range(nF):
                            for m in range(nF):
                                # emit main expression
                                emit_expr = sym.printing.cxxcode(new_expr[(nF)*(nF)*(nF)*(nF)*i+(nF)*(nF)*(nF)*j+(nF)*(nF)*k+(nF)*l+m])
                                rw_expr = rewrite_indices(emit_expr, nF, nP)
                                h.write('\n RMcd['+str((nF)*(nF)*(nF)*(nF)*i+(nF)*(nF)*(nF)*j+(nF)*(nF)*k+(nF)*l+m)+']=' + str(rw_expr) + ';\n')

    h.close()
    e.close()

    return g, Ga, Ri, Rm