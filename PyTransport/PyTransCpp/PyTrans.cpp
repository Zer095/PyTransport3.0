//#This file is part of PyTransport.

//#PyTransport is free software: you can redistribute it and/or modify
//#it under the terms of the GNU General Public License as published by
//#the Free Software Foundation, either version 3 of the License, or
//#(at your option) any later version.

//#PyTransport is distributed in the hope that it will be useful,
//#but WITHOUT ANY WARRANTY; without even the implied warranty of
//#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#GNU General Public License for more details.
 
//#You should have received a copy of the GNU General Public License
//#along with PyTransport.  If not, see <http://www.gnu.org/licenses/>.

// C++ file which defines the functions make available to Python through the MTeasy module.
#include <Python.h> 
#include <iostream>
#include <cstdio>
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

//don't adjust the labels at the end of the 4 lines below (they are used to fix directory structure)
#include"/Users/apx050/Desktop/Projects/PyTransport/PyTransport/PyTransCpp/cppsrc/NC/evolve.h"//evolve
#include"/Users/apx050/Desktop/Projects/PyTransport/PyTransport/PyTransCpp/cppsrc/NC/moments.h"//moments
#include"/Users/apx050/Desktop/Projects/PyTransport/PyTransport/PyTransCpp/cppsrc/NC/model.h"//model
#include"/Users/apx050/Desktop/Projects/PyTransport/PyTransport/PyTransCpp/cppsrc/stepper/rkf45.hpp"//stepper
//************************************************************************************************* 

#include <math.h>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <time.h>
# include <iomanip>
# include <cmath>

using namespace std;

// The line below is updated evey time the moduleSetup file is run.
// Package recompile attempted at: Fri Dec 13 16:05:07 2024

// Changes python array into C array (or rather points to pyarray data)
/*    Assumes PyArray is contiguous in memory.  */
double *pyvector_to_Carray(PyArrayObject *arrayin)
{
/* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Convert a PyArrayObject to a C-Array, assuming PyArray is contiguous in memory.

    Arguments
    ---------
    *arrayin : PyArrayObject
        The memory location of the first element of the array.

    Returns
    -------
    output : *Double
        The memory location of the first element of C-array, map all values to a double.

    Description
    -----------
    This function converts a PyArrayObject to a Double array (C-array).
    It takes as argument the first memory location of the PyArrayObject
    and it returns the first memory location of the C-array.

    Python Prototype
    ----------------
    carray = pyvector_to_Carray(pyarray)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    
    return (double *) arrayin->data;  /* pointer to arrayin data as double */
}


int size_pyvector(PyArrayObject *arrayin)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Return the size of a PyArrayObject.

    Arguments
    ---------
    *arrayin : PyArrayObject
        The memory location of the first element of the array.

    Returns
    -------
    output : integer
        The value of the dimension of a PyArrayObject.

    Description
    -----------
    This function returns the value of the dimension of a PyArrayObject.
    It takes as argument the first memory location of the PyArray.

    Python Prototype
    ----------------
    size = size_pyvector(pyarray)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    return arrayin->dimensions[0];  /* pointer to arrayin data as double */
}

// function to retun amplitude of potential
static PyObject* MT_V(PyObject* self,  PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Return the amplitude of the potential, given the initial values of the fields and the parameters of the model.

    Arguments
    ---------
    fieldsIn : PyArrayObject
        The initial values of the fields.
    params : PyArrayObject
        The values of the parameters of the models.

    Returns
    -------
    output : Python double
        The value of the potential computed with fieldsIn and params.

    Description
    -----------
    This function returns the value of the potential at fieldsIn. 
    It takes as argument the values of the fields and the parameters of the model.

    Python Prototype
    ----------------
    V = PyT.V(fieldsIn, params)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------

    // Declare variables and convert PyObjects to C-objects
    PyArrayObject *fieldsIn, *params;       // Input PyArray Object
    double *Cfields,*Cparams;               // Input doubles
    // Parsing args into variables
    if (!PyArg_ParseTuple(args, "O!O!",  &PyArray_Type, &fieldsIn,&PyArray_Type,&params)) {
        return NULL;}
    Cfields = pyvector_to_Carray(fieldsIn);     // Convert fields to C-array
    Cparams = pyvector_to_Carray(params);       // Convert parameters to C-array
    potential pp;                               // Define potential
    // Get number of fields then check size of fieldsIn
    int nF = pp.getnF(); if (nF!=size_pyvector(fieldsIn)){cout<< "\n \n \n field space array not of correct length \n \n \n";    Py_RETURN_NONE;}
    // Get number of parameters then check size of params
    int nP = pp.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length \n \n \n";  Py_RETURN_NONE;}
    // Vector with fields values
    vector<double> vectIn; vectIn = vector<double>(Cfields, Cfields + nF);
    // Vector with parameters values
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
    // Return the values of the potential
    return Py_BuildValue("d", pp.V(vectIn,Vparams));
}

// function to calculate derivatives of potential
static PyObject* MT_dV(PyObject* self,  PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Return the an array with derivatives of the potential with respect to the fields, computed at fieldsIn.

    Arguments
    ---------
    fieldsIn : PyArrayObject
        The initial values of the fields.
    params : PyArrayObject
        The values of the parameters of the models.

    Returns
    -------
    output : Python array 
        The value of the potential computed with fieldsIn and params.

    Description
    -----------
    This function returns an array with the values of the derivative of potential at fieldsIn. 
    It takes as argument the values of the fields and the parameters of the model.

    Python Prototype
    ----------------
    dV = PyT.dV(fieldsIn, params)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------

    // Declare variables and convert PyObjects to C-objects
    PyArrayObject *fieldsIn, *params;     // Input PyArray Objects
    PyArrayObject *dVI;                   // Output PyArray Objects
    double *Cfields, *dVC, *Cparams ;     // Input C-objects
    // Parsing args into variables
    if (!PyArg_ParseTuple(args, "O!O!",  &PyArray_Type, &fieldsIn,&PyArray_Type,&params)) {
        return NULL;}
    Cfields = pyvector_to_Carray(fieldsIn);     // Convert fields to C-array
    Cparams = pyvector_to_Carray(params);       // Convert params to C-array
    potential pp;                               // Define potential
    // Get number of fields and check size of fieldsIn
    int nF = pp.getnF();if (nF!=size_pyvector(fieldsIn)){cout<< "\n \n \n field space array not of correct length \n \n \n";    Py_RETURN_NONE;}
    // Get number of parameters and check size of params
    int nP = pp.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length \n \n \n";  Py_RETURN_NONE;}
    // Vectori with fields values
    vector<double> vectIn; vectIn = vector<double>(Cfields, Cfields +  nF);
    // Vector with parameters values
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Define dimension of output array
    npy_intp dims[1];
    dims[0]=nF;
    // Define output array
    dVI = (PyArrayObject*) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
    dVC = (double*) dVI->data;
    // Get the derivative of potential
    vector<double> dVect = pp.dV(vectIn,Vparams);
    // Store values of the derivative of the potential
    for(int i=0; i<nF;i++){dVC[i] = dVect[i];}
    // return array
    return PyArray_Return(dVI);
}

// function to calculate 2nd derivatives of potential
static PyObject* MT_ddV(PyObject* self,  PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Return the a matrix with 2nd derivatives of the potential with respect to the fields, computed at fieldsIn.

    Arguments
    ---------
    fieldsIn : PyArrayObject
        The initial values of the fields.
    params : PyArrayObject
        The values of the parameters of the models.

    Returns
    -------
    output : 2D Python array 
        The value of the potential computed with fieldsIn and params.

    Description
    -----------
    This function returns an array with the values of the derivative of potential at fieldsIn. 
    It takes as argument the values of the fields and the parameters of the model.

    Python Prototype
    ----------------
    ddV = PyT.ddV(fieldsIn, params)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------

    // Declare variables and convert PyObjects to C-objects
    PyArrayObject *fieldsIn, *params;        // Input PyArrayObject
    PyArrayObject *ddVI;                     // Output PyArrayObject
    double *Cfields, *ddVC, *Cparams ;       // Input doubles array
    // Parsing arguments into variables
    if (!PyArg_ParseTuple(args, "O!O!",  &PyArray_Type, &fieldsIn,&PyArray_Type,&params)) {
        return NULL;}
    Cfields = pyvector_to_Carray(fieldsIn);     // Convert fields to C-array
    Cparams = pyvector_to_Carray(params);       // Convert parameters to C-array
    potential pp;                               // Define potential
    // Get number of fields then check size of fieldsIn
    int nF = pp.getnF();if (nF!=size_pyvector(fieldsIn)){cout<< "\n \n \n field space array not of correct length \n \n \n";    Py_RETURN_NONE;}
    // Get number of parameters then check size of params
    int nP = pp.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length \n \n \n";  Py_RETURN_NONE;}
    // Vector with fields
    vector<double> vectIn; vectIn = vector<double>(Cfields, Cfields +  nF);
    // Vector with parameters
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Define dimension of output array
    npy_intp dims[2];
    dims[0]=nF; dims[1]=nF;
    // Define output array
    ddVI = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    ddVC = (double*) ddVI->data;
    // Get 2nd derivative of the potential
    vector<double> ddVect = pp.dVV(vectIn,Vparams);
    // Store values of the derivative
    for(int i=0; i<nF;i++){for(int j=0; j<nF;j++){ddVC[i+j*nF] = ddVect[i+j*nF];}}
    // return matrix
    return PyArray_Return(ddVI);
}

// function to calculate Hubble rate
static PyObject* MT_H(PyObject* self,  PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Return the value of the Hubble rate, given the values of the fields and the velocities

    Arguments
    ---------
    fields_dfieldsIn : PyArrayObject
        The initial values of the fields and velocities
    params : PyArrayObject
        The values of the parameters of the models.

    Returns
    -------
    output : Python double
        The value of the Hubble rate computed with fields_dfieldsIn and params.

    Description
    -----------
    This function returns the value of the Hubble rates at fields_dfieldsIn. 
    It takes as argument the values of the fields, the velocities and the parameters of the model.

    Python Prototype
    ----------------
    H = PyT.H(fieldsIn, params)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Declare variables and convert PyObjects to C-objects
    PyArrayObject *fields_dfieldsIn, *params;       // Input PyArray Objects
    double *Cfields_dfields, *Cparams;              // Input doubles 
    // Parsign args into variables
    if (!PyArg_ParseTuple(args, "O!O!",  &PyArray_Type, &fields_dfieldsIn,&PyArray_Type,&params)) {
        return NULL;}
    Cfields_dfields = pyvector_to_Carray(fields_dfieldsIn);     // Convert fields and velocities to C-array
    Cparams = pyvector_to_Carray(params);                       // Convert params to C-array
    model mm;                                                   // Define model
    // Get number of fields then check size of fieldsIn
    int nF = mm.getnF(); if (2*nF!=size_pyvector(fields_dfieldsIn)){cout<< "\n \n \n field space array not of correct length\n \n \n ";    Py_RETURN_NONE;}
    // Get the number of params then check size of params
    int nP = mm.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length \n \n \n";  Py_RETURN_NONE;}
    // Vector with fields and velocities
    vector<double> vectIn; vectIn = vector<double>(Cfields_dfields, Cfields_dfields + 2*nF);
    // Vector with parameters
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
    // Return the value of the Hubble rate
    return Py_BuildValue("d", mm.H(vectIn, Vparams));
}

// function to calculate Epsilon
static PyObject* MT_Ep(PyObject* self,  PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Return the value of the first slow roll parameter, given the values of the fields, velocities and params

    Arguments
    ---------
    fields_dfieldsIn : PyArrayObject
        The initial values of the fields and velocities
    params : PyArrayObject
        The values of the parameters of the models.

    Returns
    -------
    output : Python double
        The value of epsilon rate computed with fields_dfieldsIn and params.

    Description
    -----------
    This function returns the value of epsilon at fields_dfieldsIn. 
    It takes as argument the values of the fields, the velocities and the parameters of the model.

    Python Prototype
    ----------------
    Ep = PyT.Ep(fieldsIn, params)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Declare variables and convert PyObjects to C-objects
    PyArrayObject *fields_dfieldsIn, *params;       // Input PyArray Objects
    double *Cfields_dfields, *Cparams;              // Input doubles 
    // Parsign args into variables
    if (!PyArg_ParseTuple(args, "O!O!",  &PyArray_Type, &fields_dfieldsIn,&PyArray_Type,&params)) {
        return NULL;}
    Cfields_dfields = pyvector_to_Carray(fields_dfieldsIn);     // Convert fields and velocities to C-array
    Cparams = pyvector_to_Carray(params);                       // Convert params to C-array
    model mm;                                                   // Define model
    // Get number of fields then check size of fieldsIn
    int nF = mm.getnF(); if (2*nF!=size_pyvector(fields_dfieldsIn)){cout<< "\n \n \n field space array not of correct length\n \n \n ";    Py_RETURN_NONE;}
    // Get the number of params then check size of params
    int nP = mm.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length \n \n \n";  Py_RETURN_NONE;}
    // Vector with fields and velocities
    vector<double> vectIn; vectIn = vector<double>(Cfields_dfields, Cfields_dfields + 2*nF);
    // Vector with parameters
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
    // Return the value of the epsilon
    return Py_BuildValue("d", mm.Ep(vectIn, Vparams));
}

// function to calculate Epsilon
static PyObject* MT_Eta(PyObject* self,  PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Return the value of the first slow roll parameter, given the values of the fields, velocities and params

    Arguments
    ---------
    fields_dfieldsIn : PyArrayObject
        The initial values of the fields and velocities
    params : PyArrayObject
        The values of the parameters of the models.

    Returns
    -------
    output : Python double
        The value of epsilon rate computed with fields_dfieldsIn and params.

    Description
    -----------
    This function returns the value of epsilon at fields_dfieldsIn. 
    It takes as argument the values of the fields, the velocities and the parameters of the model.

    Python Prototype
    ----------------
    Ep = PyT.Ep(fieldsIn, params)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Declare variables and convert PyObjects to C-objects
    PyArrayObject *fields_dfieldsIn, *params;       // Input PyArray Objects
    double *Cfields_dfields, *Cparams;              // Input doubles 
    // Parsign args into variables
    if (!PyArg_ParseTuple(args, "O!O!",  &PyArray_Type, &fields_dfieldsIn,&PyArray_Type,&params)) {
        return NULL;}
    Cfields_dfields = pyvector_to_Carray(fields_dfieldsIn);     // Convert fields and velocities to C-array
    Cparams = pyvector_to_Carray(params);                       // Convert params to C-array
    model mm;                                                   // Define model
    // Get number of fields then check size of fieldsIn
    int nF = mm.getnF(); if (2*nF!=size_pyvector(fields_dfieldsIn)){cout<< "\n \n \n field space array not of correct length\n \n \n ";    Py_RETURN_NONE;}
    // Get the number of params then check size of params
    int nP = mm.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length \n \n \n";  Py_RETURN_NONE;}
    // Vector with fields and velocities
    vector<double> vectIn; vectIn = vector<double>(Cfields_dfields, Cfields_dfields + 2*nF);
    // Vector with parameters
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
    // Return the value of the epsilon
    return Py_BuildValue("d", mm.Eta(vectIn, Vparams));
}

// function to return number of fields (useful for cross checks)
static PyObject* MT_fieldNumber(PyObject* self,  PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Return the number of fields of a given model
 
    Arguments
    ---------
    None

    Returns
    -------
    nF : integer
        integer representing the number of the fields of a given model.

    Description
    -----------
    This function returns the number of fields of a given model.
    This function doesn't get any argument, it defines a model (mm) object using the class model defined in CppTrans/model.h,
    and then it uses the metod .getnF() to return the number of fields.
    

    Python Prototype
    ----------------
    nF = PyT.fieldNumber()

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/

    if (!PyArg_ParseTuple(args, "")) {
        return NULL;}
    model mm;   // Define model
    // Return the number of fields
    return Py_BuildValue("i",mm.getnF());
}

// function to return number of parameters (useful for cross checks)
static PyObject* MT_paramNumber(PyObject* self,  PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Return the number of parameters of a given model
 
    Arguments
    ---------
    None

    Returns
    -------
    nF : integer
        integer representing the number of the parameters of a given model.

    Description
    -----------
    This function returns the number of parameters of a given model.
    This function doesn't get any argument, it defines a model (mm) object using the class model defined in CppTrans/model.h,
    and then it uses the metod .getnF() to return the number of parameters. 
    

    Python Prototype
    ----------------
    nF = PyT.fieldNumber()

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;}
    model mm; // Define model
    // Return the number of parameters
    return Py_BuildValue("i",mm.getnP());
}

// function to calculate background evolution
static PyObject* MT_backEvolve(PyObject* self,  PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Return the amplitude of the potential, given the initial values of the fields and the parameters of the model.
 
    Arguments
    ---------
    t : PyArrayObject
        1D Array containing the values of each time-step.
    initialCs : PyArrayObject
        The initial values of the fields and velocities.
    params : PyArrayObject
        The values of the parameters of the models.
    tols: PyArrayObject
        1D Array (len=2) containing the absolute and relative tolerance for a numerical integrator Rk45.
    exit : bool
        If True, returns background evolution until the end of inflation (epsilon >1).
        If False, returns background evolution for each value of t.

    Returns
    -------
    backOut : PyArrayObject
        2D array (size_t*, 2*nF). If exit = True, it returns the value of the background evolution up to the end of inflation (epsilon >1).
        If exit = False, it returns the value of the background evolution for each value of t

    Description
    -----------
    This function returns the value of the background evolution of the model, using the derivatives computed
    with the function evolveB in CppTrans/evolve.h
    It computes the fields and velocity trajectory along inflation.
    It takes as argument the array of time values t, the initial conditions for fields and velocities, 
    the parameters of the model, the tolerance array for the numerical integrator, a boolean.
    

    Python Prototype
    ----------------
    back = PyT.backEvolve(t, initialCs, params, tols, exit)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------

    // Declare variables and conver PyObjects to C-objects
    PyArrayObject *initialCs, *t, *backOutT, *params, *tols;   // Input PyArray Objects
    PyArrayObject *backOut;                                    // Output PyArray Object
    double *CinitialCs, *tc, *Cparams, *tolsC ;                // Input doubles array
    bool exit;                                                 // Input boolean
    double abserr, relerr;                                     // Input abs and rel tolerances
    // Parsing args into variables
    if (!PyArg_ParseTuple(args, "O!O!O!O!b",&PyArray_Type, &t, &PyArray_Type, &initialCs, &PyArray_Type, &params, &PyArray_Type, &tols, &exit)){return NULL;}
    tolsC = pyvector_to_Carray(tols);                   // Tols to C-array
    CinitialCs = pyvector_to_Carray(initialCs);         // Ics to C-array
    tc = pyvector_to_Carray(t);                         // Time to C-array
    Cparams = pyvector_to_Carray(params);               // params to C-array
    // Check size of tols
    if (2!=size_pyvector(tols)){cout<< "\n \n \n incorrect tolorances input, using defaults  \n \n \n";
        abserr = pow(10,-8.); relerr = pow(10,-8.);}else {abserr =tolsC[0];relerr = tolsC[1];}
    model mm;   // define model
    // Get number of fields then check size of initialCs
    int nF=mm.getnF(); if (2*nF!=size_pyvector(initialCs)){cout<< "\n \n \n field space array not of correct length \n \n \n";    Py_RETURN_NONE;}
    // Get number of params then check size of params
    int nP = mm.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length \n \n \n";  Py_RETURN_NONE;}

    double N=tc[0];  // Initialize time
    // Vector with initial fields
    vector<double> vectIn; vectIn = vector<double>(CinitialCs, CinitialCs + 2*nF);
    back b(nF, vectIn);         // Instance of back with ics
    
    int flag=-1;
    // Define array to store fields and derivatives
    double *y; y = new double[2*nF];
    double *yp; yp= new double[2*nF];
    // Store initial conditions
    for (int i=0;i<2*nF;i++){y[i] = CinitialCs[i];}
    // Compute evolution of fields and velocities for each time-step in t
    if (exit == false){
        // Get dimension of output array
        int nt = t->dimensions[0];
        // Set up output array
        npy_intp dims[2];
        dims[1]=1+2*nF; dims[0]=nt;
        double * backOutC;
        backOut = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
        backOutC = (double *) PyArray_DATA(backOut);
        // Compute derivatives
        evolveB(N, y, yp, Cparams);
        // run background *********************
        for(int ii=0; ii<nt; ii++ ){
            // evolve background
            while (N<tc[ii]){
                flag = r8_rkf45(evolveB , 2*nF, y, yp, &N, tc[ii], &relerr, abserr, flag, Cparams );
                if (flag== 50){cout<< "\n \n \n Integrator failed at time N = " <<N <<" \n \n \n";  return Py_BuildValue("d", N);}
                flag=-2;
            }
            // Store time
            backOutC[ii*(2*nF+1)]=N;
            // Store field and velocities
            for(int i=0;i< 2*nF;i++){
                backOutC[ii*(2*nF+1)+i+1]=y[i];} // output array
        }
    }
    // Compute the evolution of fields and velocities for each time step until the end of inflation (when espilon >1)
    if (exit == true){
        // Get dimension of t
        int nt = t->dimensions[0];
        // Set up the dimentsion of auxiliary array
        npy_intp dims[2];
        dims[1]=1+2*nF; dims[0]=nt;
        double * backOutCT;
        backOutT = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
        backOutCT = (double *) PyArray_DATA(backOutT);
        // Compute derivatives
        evolveB(N, y, yp, Cparams);
        // vector with fields and velocities
        vector<double> vecy;
        // Vector with parameters
        vector<double> Vparams;
        // run background *********************
        {int ii =0;double eps=0.0; 
        // Stop when eps > 1 or when the time-step is equal to the final time-step in t
        while (eps<1 && ii<nt){
            while (N<tc[ii]){
                // evolve background
                flag = r8_rkf45(evolveB , 2*nF, y, yp, &N, tc[ii], &relerr, abserr, flag, Cparams );
                if (flag== 50){cout<< "\n \n \n Integrator failed at time N = " <<N <<" \n \n \n"; return Py_BuildValue("d", N);}
                flag=-2;}
                // store time
                backOutCT[ii*(2*nF+1)]=N;
                // store fields and velocities
                for(int i=0;i< 2*nF;i++){
                    backOutCT[ii*(2*nF+1)+i+1]=y[i];} // outputs to file at each step
                // Compute epsilon -------------------------------------------------------------------------------------------------------------
                // Store fields and velocities
                vecy = vector<double>(y, y + 2*nF);
                // store parameters 
                Vparams = vector<double>(Cparams, Cparams +  nP);
                // Compute epsilon
                eps = mm.Ep(vecy,Vparams);
                // Update ii
                ii++;
            }
            // Print ii
            cout << ii <<endl;
            // Set up output array
            npy_intp dims2[2];
            dims2[1]=1+2*nF; dims2[0]=ii;
            double * backOutC;
            backOut = (PyArrayObject*) PyArray_SimpleNew(2,dims2,NPY_DOUBLE);
            backOutC = (double *) PyArray_DATA(backOut);
            // Store time
            for(int jj = 0; jj<ii; jj++){backOutC[jj*(2*nF+1)]=tc[jj];
            // Store fields and velocities
            for(int i=0;i< 2*nF;i++){
                backOutC[jj*(2*nF+1)+i+1]=backOutCT[jj*(2*nF+1)+i+1] ;}}
        }
        }
    
    delete[] y; delete[] yp;
    return PyArray_Return(backOut);
}

// function to calculate 2pt evolution
static PyObject* MT_sigEvolve(PyObject* self,  PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Gives the entire evolution of sigma (2pt correlation function) 

    Arguments
    ---------
    t : PyArrayObject*
        1D Array containing the values of each time-step.
    k : double
        The value of the k mode
    initialCs : PyArrayObject*
        1D Array (len=2*nF) containing the initial values of fields and velocities.
    params : PyArrayObject*
        1D Array (len=nP) containing the parameters of the model.
    tols : PyArrayObject*
        1D Array (len=2) containing the absolute and relative tolerance for the numerical integrator RK45.
    full : bool
        If True, returns time value, fields and velocity values, power-spectrum of zeta, and sigma for each time-step;
        if False, returns time value and power-spectrum of zeta for each time step
    Returns
    -------
    sigOut : PyArrayObject
            2D array (if bool = True -> (size_t, 1 +2*nF+1+2*nF*2*nF), if bool false -> (size_t, 1+1)) containing for each time step (on each row): 
            time value, field and velocity values, sigma.

    Description
    -----------
    This function computes the sigma for each value of t.
    It evolve each element of the matrix, starting from the initial conditions (defined when we initialize the object sigma),
    using the derivative computed with the function evolveSig in CppTrans/evolve.h. 
    Then it returns a 2D array storing the matrix at each time step.

    Python Prototype
    ----------------
    twoPt = PyT.sigEvolve(t, k, initialCs, params, tols, full)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------

    // Declare variables and convert PyObjects to C-Objects
    PyArrayObject *initialCs, *t, *params, *tols;   // Input PyArray Objects
    PyArrayObject  *sigOut;                         // Output PyArrayObject 
    double *CinitialCs, *tc, k, *Cparams, *tolsC;   // Input doubles array
    double rtol, atol;                              // Input doubles rel and abs tol
    bool full;                                      // Input boolean
    // Parsing args into variables
    if (!PyArg_ParseTuple(args, "O!dO!O!O!b", &PyArray_Type, &t, &k, &PyArray_Type, &initialCs,&PyArray_Type, &params, &PyArray_Type, &tols,&full)) {
        return NULL;}
    
    CinitialCs = pyvector_to_Carray(initialCs);         // Convert initial conditions to C-array
    tc = pyvector_to_Carray(t);                         // Convert time to C-array
    tolsC = pyvector_to_Carray(tols);                   // Convert tol to C-array
    Cparams = pyvector_to_Carray(params);               // Convert params to C-array

    // Check size tols
    if (2!=size_pyvector(tols)){cout<< "\n \n \n incorrect tolorances input, using defaults  \n \n \n";
        atol = pow(10,-8.); rtol = pow(10,-8.);}
    else {
        atol =tolsC[0];rtol = tolsC[1];}

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Define the model and compute the needed quantities
    model mm;           // define model
    potential pott;     // define poential
    // Define number of fields, check the size of initial conditions array (initialCs)
    int nF=mm.getnF(); if (2*nF!=size_pyvector(initialCs)){cout<< "\n \n \n field space array not of correct length, not proceeding further \n \n \n";    Py_RETURN_NONE;}
    // Vector with ics
    vector<double> vectIn;vectIn = vector<double>(CinitialCs, CinitialCs + 2*nF);
    // Define number of parameters, check the size of parameters array (params)    
    int nP = mm.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length, not proceeding further \n \n \n";  Py_RETURN_NONE;}
    // vector with parameters
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
    
    // we use a scaling below that we rescale back at the end (so the final answer is as if the scaling was never there -- this helps standarise the rtol and atol needed for the same model run with differnet initial conditions
    double kn = 1.0; 
    double kscale = k;    
    double Nstart=tc[0] - log(kscale);
    
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the initial values of sigma and initialize the vector

    sigma sig(nF, kn, Nstart, vectIn, Vparams) ; // instance of sigma object which fixs ics
    double* y; // set up array for ics
    y = new double[2*nF + 2*nF*2*nF];
    for(int i=0; i<2*nF;i++){y[i] = CinitialCs[i];} // fix values of input array
    for(int i=0; i< 2*nF;i++){for(int j=0;j<2*nF;j++){y[2*nF+ i+2*nF*j] = sig.getS(i,j);}}

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the derivative of MPP

    double* paramsIn; // array of parameters to pass to LHS of ODE routine
    paramsIn = new double[1+nP];
    for(int i=0; i<nP;i++) paramsIn[i]=Vparams[i];
    paramsIn[nP]=kn;
    double* yp ; yp = new double [2*nF + 2*nF*2*nF];
    double N=Nstart;  // Initialize time
    evolveSig(N, y, yp, paramsIn);
    
    // Compute zz initial
    // Vector with fields
    vector<double> fieldIn(2*nF);
    fieldIn = vector<double>(y,y+2*nF);
    // Vector gauge transformation
    vector<double> Ni;
    Ni=mm.N1(fieldIn,Vparams,N); // calculate N,i array
    double zz=0;
    zz=0;
    for(int i=0; i<2*nF;i++){for(int j=0; j<2*nF; j++){
        zz=zz+Ni[i]*Ni[j]*y[2*nF + i + j*2*nF];}
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Set output array sigOut    

    int nt = t->dimensions[0]; // size of t (number of rows)
    int size;
    // number of columns
    if (full ==true){size = 1 + 2*nF + 1+ 2*nF*2*nF;}
    if (full ==false){size = 1 + 1;}
    npy_intp dims[2];
    dims[1]=size; dims[0]=nt;
    double * sigOutC;
    sigOut = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    sigOutC = (double *) PyArray_DATA(sigOut);
    
    // define flag
    int flag=-1;
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Evolve sigma    
    for(int ii=0; ii<nt; ii++ ){
        // evolve rho at t = tc[ii]
        while (N<tc[ii]-log(kscale)){
            flag = r8_rkf45(evolveSig , 2*nF+2*nF*2*nF, y, yp, &N, tc[ii]-log(kscale), &rtol, atol, flag, paramsIn );
            if (flag== 50){cout<< "\n \n \n Integrator failed at time N = " <<N <<" \n \n \n"; return Py_BuildValue("d", N);}
            flag = -2;
        }
        // vector with fields
        fieldIn = vector<double>(y,y+2*nF);
        //store time
        sigOutC[ii*size] = N+log(kscale);
        
        // compute zz
        Ni=mm.N1(fieldIn,Vparams,N); // calculate N,i array
        zz=0;
        for(int i=0; i<2*nF;i++){for(int j=0; j<2*nF; j++){
            zz=zz+Ni[i]*Ni[j]*y[2*nF + i + j*2*nF];}}
        // store zz
        sigOutC[ii*size+1] = zz/kscale/kscale/kscale;
        
        // store fields, velocities, sigma
        if(full==true){
            for(int i=0;i<2*nF;i++)
            {
                sigOutC[ii*(size)+i+2]=y[i];
            }
            for(int i=2*nF;i<2*nF+ 2*nF*2*nF;i++)
            {
                sigOutC[ii*(size)+i+2]=y[i]/kscale/kscale/kscale;
            }
        }
    }
    // delete arrays
    delete [] y; delete [] yp;
    delete [] paramsIn;

    return PyArray_Return(sigOut);
}

// function to calculate 2pt evolution of tensor perturbations
static PyObject* MT_gamEvolve(PyObject* self,  PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Gives the entire evolution of gamma (2pt correlation function of tensor perturbation) 

    Arguments
    ---------
    t : PyArrayObject*
        1D Array containing the values of each time-step.
    k : double
        The value of the k mode
    initialCs : PyArrayObject*
        1D Array (len=2*nF) containing the initial values of fields and velocities.
    params : PyArrayObject*
        1D Array (len=nP) containing the parameters of the model.
    tols : PyArrayObject*
        1D Array (len=2) containing the absolute and relative tolerance for the numerical integrator RK45.
    full : bool
        If True, returns time value, fields and velocity values, power-spectrum of zeta, and gamma for each time-step;
        if False, returns time value and power-spectrum of zeta for each time step
    Returns
    -------
    gamOut : PyArrayObject
            2D array (if bool = True -> (size_t, 1 +2*nF+1+2*nF*2*nF), if bool false -> (size_t, 1+1)) containing for each time step (on each row): 
            time value, field and velocity values, sigma.

    Description
    -----------
    This function computes the gamma for each value of t.
    It evolve each element of the matrix, starting from the initial conditions (defined when we initialize the object sigma),
    using the derivative computed with the function evolveGam in CppTrans/evolve.h. 
    Then it returns a 2D array storing the matrix at each time step.

    Python Prototype
    ----------------
    twoG = PyT.gamEvolve(t, k, initialCs, params, tols, full)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------

    // Declare variables and convert PyObjects to C-Objects
    PyArrayObject *initialCs, *t, *params, *tols;   // Input PyArray Objects
    PyArrayObject  *gamOut;                         // Output PyArray Object
    double *CinitialCs, *tc, k, *Cparams, *tolsC;   // Input doble
    double rtol, atol;                              // Input double tols
    bool full;                                      // Input boolean
    // Parsing args into variables
    if (!PyArg_ParseTuple(args, "O!dO!O!O!b", &PyArray_Type, &t, &k, &PyArray_Type, &initialCs,&PyArray_Type, &params, &PyArray_Type, &tols,&full)) {
        return NULL;}
    CinitialCs = pyvector_to_Carray(initialCs);     // Ics to C-arrya
    tc = pyvector_to_Carray(t);                     // Time to C-array
    tolsC = pyvector_to_Carray(tols);               // Tols to C-array
    Cparams = pyvector_to_Carray(params);           // Params to C-Array

    // Check size of tols
    if (2!=size_pyvector(tols)){cout<< "\n \n \n incorrect tolorances input, using defaults  \n \n \n";
        atol = pow(10,-8.); rtol = pow(10,-8.);}
    else {
        atol =tolsC[0];rtol = tolsC[1];}

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Define the model and compute the needed quantities
    model mm;               // Define model
    potential pott;         // Define potential
    // Number of fields, check size of Ics
    int nF=mm.getnF(); if (2*nF!=size_pyvector(initialCs)){cout<< "\n \n \n field space array not of correct length, not proceeding further \n \n \n";    Py_RETURN_NONE;}
	int numT=1;             // Define numer of tensors
    // vector with ics
    vector<double> vectIn; vectIn = vector<double>(CinitialCs, CinitialCs + 2*nF);
    // Number of parameters, check size of params
    int nP = mm.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length, not proceeding further \n \n \n";  Py_RETURN_NONE;}
    // vector with parameters
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
    
    // we use a scaling below that we rescale back at the end (so the final answer is as if the scaling was never there -- this helps standarise the rtol and atol needed for the same model run with differnet initial conditions
    double kn = 1.0; 
    double kscale = k;    
    double Nstart=tc[0] - log(kscale);
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the initial values of rho(MPP) and initialize the vector

    Gamma gam(numT, kn, Nstart, vectIn, Vparams) ; // instance of Gamma object which fixs ics

    double* y; // set up array for ics
    y = new double[2*nF + 2*numT*2*numT];
    // Store intial values
    for(int i=0; i<2*nF;i++){y[i] = CinitialCs[i];} // fix values of input array
    for(int i=0; i< 2*numT;i++){for(int j=0;j<2*numT;j++){y[2*nF+ i+2*numT*j] = gam.getG(i,j);}}
    
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the derivative of MPP

    double* paramsIn; // array of parameters to pass to LHS of ODE routine
    paramsIn = new double[1+nP];
    for(int i=0; i<nP;i++) paramsIn[i]=Vparams[i];
    paramsIn[nP]=kn;
        
    double N=Nstart;    // Initialize time
    double TT=0;        // Initiale TT
    
    // Define vector of derivatives and store derivatives
    double* yp ; yp = new double [2*nF + 2*numT*2*numT];
    evolveGam(N, y, yp, paramsIn);

    // Compute TT
    vector<double> fieldIn(2*nF);           // Vector of fields
    fieldIn = vector<double>(y,y+2*nF);     // Store vector of fields
    TT=0;
    for(int i=0; i<2*numT;i++){for(int j=0; j<2*numT; j++){
        TT=TT+y[2*nF + i + j*2*numT];}
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Set output array gamOut
    
    int nt = t->dimensions[0];          // Numeber of rows
    
    int size;                           // Number of columns
    if (full ==true){size = 1+2*nF + 1+ 2*numT*2*numT;}
    if (full ==false){size = 1 + 1;}
    
    npy_intp dims[2];
    dims[1]=size; dims[0]=nt;
    double * gamOutC;
    gamOut = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    gamOutC = (double *) PyArray_DATA(gamOut);
    
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Evolve Gamma
    int flag=-1;
    for(int ii=0; ii<nt; ii++ ){
        // Evolve gamma at t = tc[ii]
        while (N<tc[ii]-log(kscale)){
            flag = r8_rkf45(evolveGam , 2*nF+2*numT*2*numT, y, yp, &N, tc[ii]-log(kscale), &rtol, atol, flag, paramsIn );
			//cout << y[0] << ' _ '<< y[1] << endl;
            if (flag== 50){cout<< "\n \n \n Integrator failed at time N = " <<N <<" \n \n \n"; return Py_BuildValue("d", N);}
            flag = -2;
        }
        // Vector of fields
        fieldIn = vector<double>(y,y+2*nF);
        // Store time
        gamOutC[ii*size] = N+log(kscale);
        // Store ?
        gamOutC[ii*size+1] =y[2*nF ]/kscale/kscale/kscale;
        // Store Fiedls velocities and Gamma
        if(full==true){
            for(int i=0;i<2*nF;i++)
            {
                gamOutC[ii*(size)+i+2]=y[i];
            }
            for(int i=2*nF;i<2*nF+ 2*numT*2*numT;i++)
            {
                gamOutC[ii*(size)+i+2]=y[i]/kscale/kscale/kscale;
            }
        }
        
    }
    // Delete array
    delete [] y; delete [] yp;
    delete [] paramsIn;

    return PyArray_Return(gamOut);
}

// function to calculate 3pt evolution
static PyObject* MT_alphaEvolve(PyObject* self,  PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Gives the entire evolution of gamma (2pt correlation function of tensor perturbation) 

    Arguments
    ---------
    t : PyArrayObject*
        1D Array containing the values of each time-step.
    k1 : double
        The value of the first k mode
    k2 : double
        The value of the second k mode
    k2 : double
        The value of the third k mode
    initialCs : PyArrayObject*
        1D Array (len=2*nF) containing the initial values of fields and velocities.
    params : PyArrayObject*
        1D Array (len=nP) containing the parameters of the model.
    tols : PyArrayObject*
        1D Array (len=2) containing the absolute and relative tolerance for the numerical integrator RK45.
    full : bool
        If True, returns time value, fields and velocity values, power-spectrum of zeta for k1, power-spectrum of zeta for k2, power-spectrum of zeta for k3,
        bispectrum of zeta, and alpha for each time-step;
        if False, returns time value and power-spectrum of zeta for k1, power-spectrum of zeta for k2, power-spectrum of zeta for k3, 
        and bispectrum of zeta for each time step
    Returns
    -------
    gamOut : PyArrayObject
            2D array (if bool = True -> (size_t, 1 + 4 + (2*nF)**3), if bool false -> (size_t, 1+4)) containing for each time step (on each row): 
            time value, zz1, zz2, zz3, zzz, fields, velocitis and alpha.

    Description
    -----------
    This function computes the alpha for each value of t.
    It evolve each element of the matrix, starting from the initial conditions (defined when we initialize the object alp),
    using the derivative computed with the function alphaSig in CppTrans/evolve.h. 
    Then it returns a 2D array storing the matrix at each time step.

    Python Prototype
    ----------------
    threePt = PyT.alpEvolve(t, k1, k2, k3, initialCs, params, tols, full)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------

    // Declare variables and convert PyObjects to C-Objects
    PyArrayObject *initialCs, *t, *params, *tols;                   // Input PyArrayObjects
    PyArrayObject *alpOut;                                          // Output PyarrayObject
    double k1, k2, k3, Nstart, *CinitialCs, *tc,*Cparams, *tolsC;   // Input doubles   
    double rtol, atol;                                              // Input tols
    bool full;                                                      // Input bool
    // Parsing arguments into variables
    if (!PyArg_ParseTuple(args, "O!dddO!O!O!b", &PyArray_Type, &t, &k1,&k2,&k3, &PyArray_Type, &initialCs,&PyArray_Type,&params,&PyArray_Type,&tols, &full)) {
        return NULL; }
    CinitialCs = pyvector_to_Carray(initialCs);     // Ics into C-array
    tc = pyvector_to_Carray(t);                     // Time into C-array
    Cparams = pyvector_to_Carray(params);           // Params into C-Array
    tolsC = pyvector_to_Carray(tols);               // Tols into C-Array
    //Check tols size
    if (2!=size_pyvector(tols)){cout<< "\n \n \n incorrect tolorances input, using defaults  \n \n \n";
        atol = pow(10,-8.); rtol = pow(10,-8.);}
    else {
        atol =tolsC[0]; rtol = tolsC[1];}
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Define the model and compute the needed quantities
    model mm;               // Define model
    potential pott;         // Define potential
    // Number of fields and check size of ics
    int nF=mm.getnF(); if (2*nF!=size_pyvector(initialCs)){cout<< "\n \n \n field space array not of correct length, not proceeding further \n \n \n";    Py_RETURN_NONE;}
    // Vector with ics
    vector<double> vectIn;vectIn = vector<double>(CinitialCs, CinitialCs+2*nF);
    // Number of parameters and check size of params
    int nP = mm.getnP();if (nP!=size_pyvector(params)){cout<< "\n \n \n  parameters array not of correct length, not proceeding further \n \n \n";  Py_RETURN_NONE;}
    // Vector with parameters
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
 
    // we use a scaling below that we rescale back at the end (so the final answer is as if the scaling was never there -- this helps standarises the rtol and atol needed for the same model run with differnet initial conditions
    
    double kscale = (k1+k2+k3)/3.;
    double k1n = k1/kscale; double k2n = k2/kscale; double k3n = k3/kscale;
    Nstart=tc[0] -log(kscale);
    double N=Nstart; // reset N
    
    // do not alter the comment at the end of the next line -- used by preprocessor
    // ****************************************************************************

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the initial values of Sigma and initialize the array

    sigma sig1(nF, k1n, Nstart, vectIn,Vparams)  ; // 3 instances of sigmas real
    sigma sig2(nF, k2n, Nstart, vectIn,Vparams)  ;
    sigma sig3(nF, k3n, Nstart, vectIn,Vparams)  ;
    sigmaI sig1I(nF, k1n, Nstart, vectIn,Vparams)  ; // 3 instances of sigma imaginary
    sigmaI sig2I(nF, k2n, Nstart, vectIn,Vparams)  ;
    sigmaI sig3I(nF, k3n, Nstart, vectIn,Vparams)  ;
    alpha alp(nF, k1n, k2n, k3n, Nstart, vectIn, Vparams); // instance of alpha
    
    double* y; // array for initial values
    y = new double[2*nF + 6*2*nF*2*nF + 2*nF*2*nF*2*nF];
    // Store initial values
    for(int i=0; i<2*nF;i++){y[i] = CinitialCs[i];}
    for(int i=0; i< 2*nF;i++){for(int j=0;j<2*nF;j++){y[2*nF+ i+2*nF*j] = sig1.getS(i,j);}}
    for(int i=0; i< 2*nF;i++){for(int j=0;j<2*nF;j++){y[2*nF + 1*(2*nF*2*nF)+ i+2*nF*j] = sig2.getS(i,j);}}
    for(int i=0; i< 2*nF;i++){for(int j=0;j<2*nF;j++){y[2*nF + 2*(2*nF*2*nF)+ i+2*nF*j] = sig3.getS(i,j);}}
    for(int i=0; i< 2*nF;i++){for(int j=0;j<2*nF;j++){y[2*nF + 3*(2*nF*2*nF)+ i+2*nF*j] = sig1I.getS(i,j);}}
    for(int i=0; i< 2*nF;i++){for(int j=0;j<2*nF;j++){y[2*nF + 4*(2*nF*2*nF)+ i+2*nF*j] = sig2I.getS(i,j);}}
    for(int i=0; i< 2*nF;i++){for(int j=0;j<2*nF;j++){y[2*nF + 5*(2*nF*2*nF)+ i+2*nF*j] = sig3I.getS(i,j);}}
    for(int i=0; i< 2*nF;i++){for(int j=0;j<2*nF;j++){for(int k=0; k<2*nF;k++){y[2*nF + 6*(2*nF*2*nF)+ i+2*nF*j + 2*nF*2*nF*k] = alp.getA(i,j,k);}}}
    
    
    double* paramsIn2; // array for parameters of RHS of ODE routine
    paramsIn2 = new double[3+nP];
    for(int i=0; i<nP;i++) paramsIn2[i]=Vparams[i];
    paramsIn2[nP]=k1n;
    paramsIn2[nP+1]=k2n;
    paramsIn2[nP+2]=k3n;
    // Compute and store derivatives
    double *yp; yp=new double[2*nF + 6*2*nF*2*nF+  2*nF*2*nF*2*nF];
    evolveAlp(N, y, yp, paramsIn2);

    // Initialize vector to store zz's and zzz
    double ZZZ=0., ZZ1=0., ZZ2=0., ZZ3=0.; //  for zeta zeta calcs
    vector<double> Ni, Nii1, Nii2, Nii3 ; // for N transforms to get to zeta

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Set dimension
    int nt = t->dimensions[0];
    npy_intp dims[2];
    int size;
    if (full==false){size =   5;}
    if (full==true){size =  5+  2*nF + 6*2*nF*2*nF+2*nF*2*nF*2*nF;}
    dims[1]=size; dims[0]=nt;
    double * alpOutC;
    alpOut = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    alpOutC = (double *) PyArray_DATA(alpOut);
    
    int flag=-1; 
    
    // run alpha *******************************************
    vector<double> fieldIn(2*nF);
    
    for(int ii=0; ii<nt; ii++ ){
        // Evolva alpha at t = tc[ii]
        while (N<(tc[ii]-log(kscale))){
            flag = r8_rkf45(evolveAlp, 2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*2*nF, y, yp, &N, tc[ii]-log(kscale), &rtol, atol, flag, paramsIn2);
            if (flag== 50){cout<< "\n \n \n Integrator failed at time N = " <<N <<" \n \n \n"; return Py_BuildValue("d", N);}
            flag=-2;
        }
        // vector with fields
        fieldIn = vector<double>(y,y+2*nF);
        Ni=mm.N1(fieldIn,Vparams,N); // calculate N,i array
        Nii1=mm.N2(fieldIn,Vparams,k1n,k2n,k3n,N); // claculate N,ij array for first arrangement of ks
        Nii2=mm.N2(fieldIn,Vparams,k2n,k1n,k3n,N); // for second
        Nii3=mm.N2(fieldIn,Vparams,k3n,k1n,k2n,N); // for the third
        
        // Compute zz's
        ZZ1=0.;
        ZZ2=0.;
        ZZ3=0.;
        for(int i=0; i<2*nF;i++){for(int j=0; j<2*nF; j++){
            ZZ1=ZZ1+Ni[i]*Ni[j]*(y[2*nF + i + j*2*nF] );
            ZZ2=ZZ2+Ni[i]*Ni[j]*y[2*nF + (2*nF*2*nF) + i + j*2*nF];
            ZZ3=ZZ3+Ni[i]*Ni[j]*y[2*nF + 2*(2*nF*2*nF) + i + j*2*nF];
        }}   
        // Compute zz
        ZZZ=0.;
        for(int i=0; i<2*nF;i++){for(int j=0; j<2*nF;j++){for(int k=0; k<2*nF;k++){
            ZZZ=ZZZ + Ni[i]*Ni[j]*Ni[k]*y[2*nF + 6*(2*nF*2*nF) + i + j*2*nF+ k*2*nF*2*nF];
            for(int l=0; l<2*nF;l++){ZZZ=ZZZ+(Nii1[i+j*2*nF]*Ni[k]*Ni[l]*y[2*nF + 1*(2*nF*2*nF) + i+k*2*nF]*y[2*nF+2*(2*nF*2*nF)+j+l*2*nF]
                                              +Nii2[i+j*2*nF]*Ni[k]*Ni[l]*y[2*nF + 0*(2*nF*2*nF) + i+k*2*nF]*y[2*nF + 2*(2*nF*2*nF) + j+l*2*nF]
                                              +Nii3[i+j*2*nF]*Ni[k]*Ni[l]*y[2*nF + 0*(2*nF*2*nF) + i+k*2*nF]*y[2*nF + 1*(2*nF*2*nF) + j+l*2*nF]);
            }}}}
        // Store time
        alpOutC[ii*size] =  N+log(kscale);
        // Store zz's and zzz
        alpOutC[ii*size+1] = ZZ1/kscale/kscale/kscale;
        alpOutC[ii*size+2] = ZZ2/kscale/kscale/kscale;
        alpOutC[ii*size+3] = ZZ3/kscale/kscale/kscale;
        alpOutC[ii*size+4] = ZZZ/kscale/kscale/kscale/kscale/kscale/kscale;
        // Store fields, velocities, sigmas, alpha
        if(full==true){
            for(int i=0;i<2*nF ;i++){
                alpOutC[ii*size+5+i] =  y[i] ;   }
            
            for(int i=2*nF;i<2*nF + 6*(2*nF*2*nF);i++){
                alpOutC[ii*size+5+i] =  y[i]/kscale/kscale/kscale ;   }

            for(int i=2*nF + 6*(2*nF*2*nF);i<2*nF + 6*(2*nF*2*nF)+ 2*nF*2*nF*2*nF;i++){
                alpOutC[ii*size+5+i] =  y[i]/kscale/kscale/kscale/kscale/kscale/kscale ;   }
        }
        
    }
    // Delete arrays
    delete [] y;  delete [] paramsIn2; delete [] yp;
    
    return PyArray_Return(alpOut);
}

// function to calculate 2pt evolution through the GAMMAs
static PyObject* MT_rhoEvolve(PyObject* self,  PyObject *args)
{
    PyArrayObject *initialCs, *t, *rhoOut, *params, *tols;
    double *CinitialCs, *tc, k, *Cparams, *tolsC;
    double rtol, atol;                                              // Input tols
    bool full;
    if (!PyArg_ParseTuple(args, "O!dO!O!O!b", &PyArray_Type, &t, &k, &PyArray_Type, &initialCs,&PyArray_Type, &params, &PyArray_Type, &tols,&full)) {
        return NULL;}
    CinitialCs = pyvector_to_Carray(initialCs);
    tc = pyvector_to_Carray(t);    
    tolsC = pyvector_to_Carray(tols);
    //Check tols size
    if (2!=size_pyvector(tols)){cout<< "\n \n \n incorrect tolorances input, using defaults  \n \n \n";
        atol = pow(10,-8.); rtol = pow(10,-8.);}
    else {
        atol =tolsC[0]; rtol = tolsC[1];}

    model mm;
    potential pott;
    int nF=mm.getnF(); if (2*nF!=size_pyvector(initialCs)){cout<< "\n \n \n field space array not of correct length, not proceeding further \n \n \n";    Py_RETURN_NONE;}
    vector<double> vectIn = vector<double>(CinitialCs, CinitialCs + 2*nF);
    
    int nP = mm.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length, not proceeding further \n \n \n";  Py_RETURN_NONE;}
    Cparams = pyvector_to_Carray(params);
    vector<double> Vparams = vector<double>(Cparams, Cparams +  nP);
    
    // we use a scaling below that we rescale back at the end (so the final answer is as if the scaling was never there -- this helps standarise the rtol and atol needed for the same model run with differnet initial conditions
    double kn = 1.0; 
    double kscale = k;    
    double Nstart=tc[0] - log(kscale);
    
    // instance of sigma object which fix ics;
    sigma sig(nF, kn, Nstart, vectIn, Vparams);

    // instance of rho object which fixs ics
    Rho1 rho(nF, kn, Nstart, vectIn, Vparams);
    

    // Set up array for 2pt 
    double* y = new double[2*nF + 2*nF*2*nF]; // set up array for ics
    double* s = new double[2*nF + 2*nF*2*nF];
    for(int i=0; i<2*nF;i++){y[i] = CinitialCs[i]; s[i] = CinitialCs[i];} // fix values of input array
    for(int i=0; i<2*nF;i++){for(int j=0;j<2*nF;j++){y[2*nF + i + 2*nF*j] = sig.getS(i, j); s[2*nF + i + 2*nF*j] = sig.getS(i, j);}}

    // Set up array for Gammas (here Rho)
    double* r = new double[2*nF + 2*nF*2*nF];

    for(int i=0; i < 2*nF;i++){r[i] = CinitialCs[i];}
    for(int i=0; i < 2*nF;i++){for(int j=0;j<2*nF;j++){r[2*nF + i + 2*nF*j] = rho.getR(i, j);}}
    

    double* paramsIn; // array of parameters to pass to LHS of ODE routine
    
    paramsIn = new double[1+nP];
    for(int i=0; i<nP;i++){paramsIn[i]=Vparams[i];};
    paramsIn[nP]=kn;
    
    // evolve a 2pt run **************************
    
    double N = Nstart;
    // Set up array for derivatives of 2pt and Gammas(rho)
    double* yp = new double [2*nF + 2*nF*2*nF]; 
    double* rp = new double [2*nF + 2*nF*2*nF];
    vector<double> Ni;
    double zz=0;
    
    int flag=-1;
    
    evolveRho1(N, r, rp, paramsIn);

    vector<double> fieldIn = vector<double>(y,y+2*nF);
    Ni=mm.N1(fieldIn,Vparams,N); // calculate N,i array
    zz=0;
    for(int i = 0; i < 2*nF; i++){
        for(int j = 0; j < 2*nF; j++){
        zz = zz + Ni[i]*Ni[j]*y[2*nF + i + j*2*nF];
        }
    }
    

    int nt = t->dimensions[0];
    
    int size;
    if (full ==true){size = 1 + 2*nF + 1 + 2*nF*2*nF;}
    if (full ==false){size = 1 + 1;}
    
    npy_intp dims[2];
    dims[1]=size; dims[0]=nt;
    double * rhoOutC;
    rhoOut = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    rhoOutC = (double *) PyArray_DATA(rhoOut);
    
    
    // Evolve rho matrices
    for(int ii=0; ii<nt; ii++ ){
        while (N<tc[ii]-log(kscale)){
            flag = r8_rkf45(evolveRho1 , 2*nF + 2*nF*2*nF, r, rp, &N, tc[ii]-log(kscale), &rtol, atol, flag, paramsIn);
            if (flag== 50){cout<< "\n \n \n Integrator failed at time N = " <<N <<" \n \n \n"; return Py_BuildValue("d", N);}
            flag = -2;
        }

        // Update fields
        for(int i = 0; i < 2*nF; i++){y[i] = r[i];};

        double sum = 0.;
        // calculate sigma
        for(int i = 0; i < 2*nF; i++){
            for(int j = 0; j < 2*nF; j++)
            {
                sum = 0.0;
                for(int a=0; a<2*nF; a++){
                    for(int b=0; b<2*nF; b++)
                    {
                        sum = sum + s[2*nF + 2*nF*a + b]*r[2*nF + a + 2*nF*i]*r[2*nF + b + 2*nF*j];
                    }
                }
                y[2*nF + 2*nF*i + j] = sum;
            }
        }

        fieldIn = vector<double>(y,y+2*nF);
        
        rhoOutC[ii*size] = N+log(kscale);
        
        // Calculate zz
        Ni=mm.N1(fieldIn,Vparams,N); // calculate N,i array
        zz=0;
        for(int i=0; i<2*nF;i++){
            for(int j=0; j<2*nF; j++){
                zz=zz+Ni[i]*Ni[j]*y[2*nF + 2*nF*i + j];
            }
        }
        

        rhoOutC[ii*size+1] = zz/kscale/kscale/kscale;
        

        if(full==true){
            // Store the Fields
            for(int i=0; i<2*nF; i++)
            {
                rhoOutC[ii*(size)+i+2] = y[i];
            }
            // Store Sigma's
            for(int i = 2*nF; i<2*nF + 2*nF*2*nF; i++)
            {
                rhoOutC[ii*(size)+i+2] = y[i]/kscale/kscale/kscale;
            }
        }

    }
    
    delete [] y; delete [] yp; delete [] r; delete [] rp; delete [] s;
    delete [] paramsIn;

    return PyArray_Return(rhoOut);
}

// function to calculate 3pt evolution through the GAMMAs
static PyObject* MT_rhoEvolve2(PyObject* self,  PyObject *args)
{
    // printf("Enter the routine\n");
    PyArrayObject *initialCs, *t, *alpROut, *params, *tols;
    double k1, k2, k3, *CinitialCs, *tc, *Cparams, *tolsC;
    bool full;
    if (!PyArg_ParseTuple(args, "O!dddO!O!O!b", &PyArray_Type, &t, &k1,&k2,&k3, &PyArray_Type, &initialCs,&PyArray_Type,&params,&PyArray_Type,&tols, &full)) {
        return NULL; }
    CinitialCs = pyvector_to_Carray(initialCs);
    tc = pyvector_to_Carray(t);
    int nt = t->dimensions[0];

        

    tolsC = pyvector_to_Carray(tols);
    double rtol, atol;
    if (2!=size_pyvector(tols)){cout<< "\n \n \n incorrect tolorances input, using defaults  \n \n \n";
        atol = pow(10,-8.); rtol = pow(10,-8.);}
    else {
        atol =tolsC[0];rtol = tolsC[1];
    }

    
    model mm;
    potential pott;
    int nF=mm.getnF(); if (2*nF!=size_pyvector(initialCs)){cout<< "\n \n \n field space array not of correct length, not proceeding further \n \n \n";    Py_RETURN_NONE;}
    vector<double> vectIn;
    vectIn = vector<double>(CinitialCs, CinitialCs + 2*nF);
    

    int nP = mm.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length, not proceeding further \n \n \n";  Py_RETURN_NONE;}
    Cparams = pyvector_to_Carray(params);
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
    
    // we use a scaling below that we rescale back at the end (so the final answer is as if the scaling was never there -- this helps standarise the rtol and atol needed for the same model run with differnet initial conditions
    double kscale = (k1+k2+k3)/3.; 
    double k1n = k1/kscale; double k2n = k2/kscale; double k3n = k3/kscale;   
    double Nstart=tc[0] - log(kscale);

    double N=Nstart; // Reset N
    
    // instance of sigma object which fix ics;
    sigma sig1(nF, k1n, Nstart, vectIn, Vparams);
    sigma sig2(nF, k2n, Nstart, vectIn, Vparams);
    sigma sig3(nF, k3n, Nstart, vectIn, Vparams);

    // instance of sigma imaginary
    sigmaI sig1I(nF, k1n, Nstart, vectIn,Vparams)  ; 
    sigmaI sig2I(nF, k2n, Nstart, vectIn,Vparams)  ;
    sigmaI sig3I(nF, k3n, Nstart, vectIn,Vparams)  ;

    // instance of alpha objects which fix ics;
    alpha alp(nF, k1n, k2n, k3n, Nstart, vectIn, Vparams);

    // instance of rho 1 object which fixs ics
    Rho1 rho11(nF, k1n, Nstart, vectIn, Vparams);
    Rho1 rho12(nF, k2n, Nstart, vectIn, Vparams);
    Rho1 rho13(nF, k3n, Nstart, vectIn, Vparams);

    // instance of rho 2 object which fixs ics
    Rho2 rho21(nF, k1n, k2n, k3n, Nstart, vectIn, Vparams);
    Rho2 rho22(nF, k2n, k1n, k3n, Nstart, vectIn, Vparams);
    Rho2 rho23(nF, k3n, k1n, k2n, Nstart, vectIn, Vparams);


    // Store fields, rho1 and rho2
    double * r = new double[2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF)];
    // store fields
    for(int i = 0; i < 2*nF; i++){r[i] = CinitialCs[i];}
    // store rho1
    for(int i = 0; i< 2*nF; i++){
        for(int j = 0; j < 2*nF; j++){
            r[2*nF + i + 2*nF*j] = rho11.getR(i,j);
            r[2*nF + 2*nF*2*nF + i + 2*nF*j] = rho12.getR(i,j);
            r[2*nF + 2*(2*nF*2*nF) + i + 2*nF*j] = rho13.getR(i,j);
        }
    }
    // store rho2
    for(int i = 0; i < 2*nF; i++){
        for(int j = 0; j < 2*nF; j++){
            for(int k = 0; k < 2*nF; k++){
                r[2*nF + 3*(2*nF*2*nF) + i + 2*nF*j + 2*nF*2*nF*k] = rho21.getR(i,j,k);
                r[2*nF + 3*(2*nF*2*nF) + (2*nF*2*nF*2*nF) + i + 2*nF*j + 2*nF*2*nF*k] = rho22.getR(i,j,k);
                r[2*nF + 3*(2*nF*2*nF) + 2*(2*nF*2*nF*2*nF) + i + 2*nF*j + 2*nF*2*nF*k] = rho23.getR(i,j,k);
            }
        }
    }


    // Store sigmas
    double *s = new double[6*(2*nF*2*nF)];

    for (int i = 0; i < 2*nF; i++){
        for (int j = 0; j < 2*nF; j++){
            // Real
            s[0*(2*nF*2*nF) + i + 2*nF*j] = sig1.getS(i,j);
            s[1*(2*nF*2*nF) + i + 2*nF*j] = sig2.getS(i,j);
            s[2*(2*nF*2*nF) + i + 2*nF*j] = sig3.getS(i,j);
            // Imaginary
            s[3*(2*nF*2*nF) + i + 2*nF*j] = sig1I.getS(i,j);
            s[4*(2*nF*2*nF) + i + 2*nF*j] = sig2I.getS(i,j);
            s[5*(2*nF*2*nF) + i + 2*nF*j] = sig3I.getS(i,j);
        }
    }


    // Store alpha
    double *ainit = new double[ 2*nF * 2*nF * 2*nF ];
    
    for(int i = 0; i < 2*nF; i++){
        for(int j = 0; j < 2*nF; j++){
            for(int k = 0; k < 2*nF; k++){
                ainit[i + 2*nF*j + (2*nF*2*nF)*k] = alp.getA(i,j,k);
            }
        }
    }

    

    double* paramsIn; // Array for parameters of RHS of ODE routine
    paramsIn = new double[3 + nP];
    for(int i = 0; i < nP; i++){paramsIn[i] = Vparams[i];}
    paramsIn[nP] = k1n;
    paramsIn[nP + 1] = k2n;
    paramsIn[nP + 2] = k3n;

    double ZZZ = 0., ZZ1 = 0., ZZ2 = 0., ZZ3 = 0.; // for zeta zeta calcs
    vector<double> Ni, Nii1, Nii2, Nii3 ; // for N transofrms to get to zeta

    double *rp = new double[2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF)];

    double *y = new double[2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*2*nF];


    npy_intp dims[2];
    int size;
    if (full==false){size = 5;}
    if (full==true){size = 5 + 2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*2*nF;}
    dims[1] = size; dims[0] = nt;
    double* alpROutC;
    alpROut = (PyArrayObject *) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    alpROutC = (double *) PyArray_DATA(alpROut);

    evolveRho2(N, r, rp, paramsIn);
    int flag = -1;

    // Run alpha **************************************************

    vector<double> fieldIn(2*nF);

    for(int ii = 0; ii<nt; ii++){
        // printf("ii = %i\n", ii);

        while (N<(tc[ii]-log(kscale))){
            flag = r8_rkf45(evolveRho2, 2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF), r, rp, &N, tc[ii]-log(kscale), &rtol, atol, flag, paramsIn);
            if (flag== 50){cout<< "\n \n \n Integrator failed at time N = " <<N <<" \n \n \n"; return Py_BuildValue("d", N);}
            flag=-2;
        }        

        // store fields
        for(int i = 0; i < 2*nF; i++){y[i] = r[i];}
 
        // Calculate and store sigma
        for(int i = 0; i < 2*nF; i++){
            for(int j = 0; j < 2*nF; j++)
            {
                // Reset
                double sumR1 = 0.;
                double sumR2 = 0.;
                double sumR3 = 0.;
                double sumI1 = 0.;
                double sumI2 = 0.;
                double sumI3 = 0.;
                // Calculate sigma
                for(int a=0; a<2*nF; a++){
                    for(int b=0; b<2*nF; b++)
                    {
                        // Real
                        sumR1 = sumR1 + s[0*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 0*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 0*(2*nF*2*nF) + b + 2*nF*j];
                        sumR2 = sumR2 + s[1*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 1*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 1*(2*nF*2*nF) + b + 2*nF*j];
                        sumR3 = sumR3 + s[2*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 2*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 2*(2*nF*2*nF) + b + 2*nF*j];
                        // Imaginary
                        sumI1 = sumI1 + s[3*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 0*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 0*(2*nF*2*nF) + b + 2*nF*j];
                        sumI2 = sumI2 + s[4*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 1*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 1*(2*nF*2*nF) + b + 2*nF*j];
                        sumI3 = sumI3 + s[5*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 2*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 2*(2*nF*2*nF) + b + 2*nF*j];
                    }
                }
                // Store
                y[2*nF + 0*(2*nF*2*nF) + 2*nF*i + j] = sumR1;
                y[2*nF + 1*(2*nF*2*nF) + 2*nF*i + j] = sumR2;
                y[2*nF + 2*(2*nF*2*nF) + 2*nF*i + j] = sumR3;
                y[2*nF + 3*(2*nF*2*nF) + 2*nF*i + j] = sumI1;
                y[2*nF + 4*(2*nF*2*nF) + 2*nF*i + j] = sumI2;
                y[2*nF + 5*(2*nF*2*nF) + 2*nF*i + j] = sumI3;
            }
        }
        

        // Calculate and store alpha
        double term0, term1, term2, term3;

        for (int a = 0; a < 2*nF; a++){
            for(int b = 0; b < 2*nF; b++){
                for(int c = 0; c < 2*nF; c++){
                    // reset term
                    term0 = 0.;
                    term1 = 0.;
                    term2 = 0.;
                    term3 = 0.;
                    // sum over i,j,k
                    for(int i = 0; i < 2*nF; i++){
                        for(int j = 0; j < 2*nF; j++){
                            for(int k = 0; k < 2*nF; k++){

                                term0 = term0 + r[2*nF + 2*nF*a + i]*r[2*nF + (2*nF*2*nF) + 2*nF*b + j]*r[2*nF + 2*(2*nF*2*nF) + 2*nF*c + k]*ainit[i + 2*nF*j + (2*nF*2*nF)*k];
                                // sum over l
                                for(int l = 0; l < 2*nF; l++){
                                    term1 = term1 + r[2*nF + 3*(2*nF*2*nF) + (2*nF*2*nF)*a + 2*nF*i + j] * r[2*nF + (2*nF*2*nF) + 2*nF*b + k] * r[2*nF + 2*(2*nF*2*nF) + 2*nF*c + l]*( s[(2*nF*2*nF) + 2*nF*i + k]*s[2*(2*nF*2*nF) + 2*nF*j + l] - s[4*(2*nF*2*nF) + 2*nF*i + k]*s[5*(2*nF*2*nF) + 2*nF*j + l] );
                                    term2 = term2 + r[2*nF + 3*(2*nF*2*nF) + (2*nF*2*nF*2*nF) + (2*nF*2*nF)*b + 2*nF*i + j] * r[2*nF + 2*nF*a + k] * r[2*nF + 2*(2*nF*2*nF) + 2*nF*c + l]*( s[2*nF*k + i]*s[2*(2*nF*2*nF) + 2*nF*j + l] - s[3*(2*nF*2*nF) + 2*nF*k + i]*s[5*(2*nF*2*nF) + 2*nF*j + l] );
                                    term3 = term3 + r[2*nF + 3*(2*nF*2*nF) + 2*(2*nF*2*nF*2*nF) + (2*nF*2*nF)*c + 2*nF*i + j] * r[2*nF + 2*nF*a + k] * r[2*nF + (2*nF*2*nF) + 2*nF*b + l]*( s[2*nF*k + i]*s[(2*nF*2*nF) + 2*nF*l + j] - s[3*(2*nF*2*nF) + 2*nF*k + i]*s[4*(2*nF*2*nF) + 2*nF*l + j] );
                                }
                            }
                        }
                    }
                    // Store alpha
                    y[2*nF + 6*(2*nF*2*nF) + (2*nF*2*nF)*a + 2*nF*b + c] = term0 + term1 + term2 + term3;
                }
            }
        }

        fieldIn = vector<double>(y,y+2*nF);
        Ni=mm.N1(fieldIn,Vparams,N); // calculate N,i array
        Nii1=mm.N2(fieldIn,Vparams,k1n,k2n,k3n,N); // claculate N,ij array for first arrangement of ks
        Nii2=mm.N2(fieldIn,Vparams,k2n,k1n,k3n,N); // for second
        Nii3=mm.N2(fieldIn,Vparams,k3n,k1n,k2n,N); // etc
        
        ZZ1=0.;
        ZZ2=0.;
        ZZ3=0.;
        for(int i=0; i<2*nF;i++){for(int j=0; j<2*nF; j++){
            ZZ1=ZZ1+Ni[i]*Ni[j]*(y[2*nF + 2*nF*i + j] );
            ZZ2=ZZ2+Ni[i]*Ni[j]*y[2*nF + (2*nF*2*nF) + 2*nF*i + j];
            ZZ3=ZZ3+Ni[i]*Ni[j]*y[2*nF + 2*(2*nF*2*nF) + 2*nF*i + j];
        }}   


        ZZZ=0.;
        for(int i=0; i<2*nF;i++){for(int j=0; j<2*nF;j++){for(int k=0; k<2*nF;k++){
            ZZZ=ZZZ + Ni[i]*Ni[j]*Ni[k]*y[2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*i + j*2*nF+ k];
            for(int l=0; l<2*nF;l++){ZZZ=ZZZ+(Nii1[i+j*2*nF]*Ni[k]*Ni[l]*y[2*nF + 1*(2*nF*2*nF) + 2*nF*i+k]*y[2*nF + 2*(2*nF*2*nF)+2*nF*j+l]
                                              +Nii2[i+j*2*nF]*Ni[k]*Ni[l]*y[2*nF + 0*(2*nF*2*nF) + 2*nF*i+k]*y[2*nF + 2*(2*nF*2*nF) + 2*nF*j+l]
                                              +Nii3[i+j*2*nF]*Ni[k]*Ni[l]*y[2*nF + 0*(2*nF*2*nF) + 2*nF*i+k]*y[2*nF + 1*(2*nF*2*nF) + 2*nF*j+l]);
        }}}}
        

        alpROutC[ii*size] =  N+log(kscale);
        alpROutC[ii*size+1] = ZZ1/kscale/kscale/kscale;
        alpROutC[ii*size+2] = ZZ2/kscale/kscale/kscale;
        alpROutC[ii*size+3] = ZZ3/kscale/kscale/kscale;
        alpROutC[ii*size+4] = ZZZ/kscale/kscale/kscale/kscale/kscale/kscale;
        
        if(full==true){
            for(int i=0; i<2*nF ;i++){
                alpROutC[ii*size+5+i] =  y[i] ;   }
            
            for(int i=2*nF; i<2*nF + 6*(2*nF*2*nF); i++){
                alpROutC[ii*size+5+i] =  y[i]/kscale/kscale/kscale ;   }

            for(int i=2*nF + 6*(2*nF*2*nF); i<2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*2*nF; i++){
                alpROutC[ii*size+5+i] =  y[i]/kscale/kscale/kscale/kscale/kscale/kscale ;   }
        }
        
    }
    
    delete [] y;  delete [] paramsIn; delete [] rp; delete [] r;
    delete [] s; delete [] ainit;
    
    return PyArray_Return(alpROut);
}



static PyObject* MT_MPP2(PyObject* self,  PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Gives the entire evolution of the MPP matrix with two indices 

    Arguments
    ---------
    t : PyArrayObject*
        1D Array containing the values of each time-step.
    k : double
        The value of the k mode
    initialCs : PyArrayObject*
        1D Array (len=2*nF) containing the initial values of fields and velocities.
    params : PyArrayObject*
        1D Array (len=nP) containing the parameters of the model.
    tols : PyArrayObject*
        1D Array (len=2) containing the absolute and relative tolerance for the numerical integrator RK45.
    Returns
    -------
    rhoOut : PyArrayObject
            2D array (size_t, 1+2*nF+2*nF*2*nF) containing for each time step (on each row): 
            time value, field and velocity values, element of MPP.

    Description
    -----------
    This function computes the MPP 2 indices matrix for each value of t.
    It evolve each element of the matrix, starting from the initial conditions (defined when we initialize the object MPP2),
    using the derivative computed with the function evolveRho2. Then it returns a 2D array storing the matrix at each time step.

    Python Prototype
    ----------------
    mpp2 = PyT.MPP2(t, k, initialCs, params, tols)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------

    // Convert PyObjects to C-Objects
    PyArrayObject *initialCs, *t, *params, *tols;   // Input PyArrayObject
    PyArrayObject *rhoOut;                          // Output PyArrayObject
    double *CinitialCs, *tc, k, *Cparams, *tolsC;   // Input doubles array
    double rtol, atol;                              // Input doubles relative tol and abs tol
    // Parsing args into variables
    if (!PyArg_ParseTuple(args, "O!dO!O!O!", &PyArray_Type, &t, &k, &PyArray_Type, &initialCs,&PyArray_Type, &params, &PyArray_Type, &tols)) {
        return NULL;}
    CinitialCs = pyvector_to_Carray(initialCs); // Convert initial values in a C-array
    tc = pyvector_to_Carray(t);                 // Convert time to a C-array
    tolsC = pyvector_to_Carray(tols);           // Convert tolerance to a C-array
    Cparams = pyvector_to_Carray(params);       // Convert params to C-array

    // Check size tols 
    if (2!=size_pyvector(tols)){cout<< "\n \n \n incorrect tolorances input, using defaults  \n \n \n";
        atol = pow(10,-8.); rtol = pow(10,-8.);}
    else {
        atol =tolsC[0]; rtol = tolsC[1];
    }
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Define the model and compute the needed quantities
    model mm;         // define model
    potential pott;   // define potential
    // Define number of fields, check the size of initial conditions array (initialCs)
    int nF=mm.getnF(); if (2*nF!=size_pyvector(initialCs)){cout<< "\n \n \n field space array not of correct length, not proceeding further \n \n \n";    Py_RETURN_NONE;}
    // vector with ics
    vector<double> vectIn; vectIn = vector<double>(CinitialCs, CinitialCs + 2*nF);
    // Define number of parameters, check the size of parameters array (params)
    int nP = mm.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length, not proceeding further \n \n \n";  Py_RETURN_NONE;}
    // vector with parameters
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
    
    // we use a scaling below that we rescale back at the end (so the final answer is as if the scaling was never there -- this helps standarise the rtol and atol needed for the same model run with differnet initial conditions
    double kn = 1.0; 
    double kscale = k;    
    double Nstart=tc[0] - log(kscale);
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the initial values of rho(MPP) and initialize the vector

    // instance of MPP object which fixs initial conditions
    Rho1 rho(nF, kn, Nstart, vectIn, Vparams);
    
    // Set up array for MPP2 (here rho)
    double* r = new double[2*nF + 2*nF*2*nF];                                                   //create array
    for(int i=0; i < 2*nF;i++){r[i] = CinitialCs[i];}                                           // store fields and velocities
    for(int i=0; i < 2*nF;i++){for(int j=0;j<2*nF;j++){r[2*nF + i + 2*nF*j] = rho.getR(i, j);}} //store initial rho
    
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the derivative of MPP

    // Set parameters
    double* paramsIn; // array of parameters to pass to LHS of ODE routine
    paramsIn = new double[1+nP]; // define size of paramsIn
    for(int i=0; i<nP;i++){paramsIn[i]=Vparams[i];}; // Initialize paramsIn
    paramsIn[nP]=kn;    
    double N = Nstart; // reset N

    // Set up array for derivatives of 2pt and Gammas(rho)
    double* rp; rp = new double [2*nF + 2*nF*2*nF]; // declare and define size rp
    // Compute derivative, store in rp
    evolveRho1(N, r, rp, paramsIn);
    // Declare variables for powerspectrum of zeta
    vector<double> Ni;   // set array for gauge transformation
    double zz=0;         // power spectrum of zz
    int flag=-1;
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Set output array rhoOut

    // Set dimension
    int nt = t->dimensions[0]; //size of t (number of rows)
    int size;
    size = 1 + 2*nF + 2*nF*2*nF; //number of columns (time, fields, mpp2)
    npy_intp dims[2];
    dims[1]=size; dims[0]=nt;

    // Define output array
    double * rhoOutC;
    rhoOut = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    rhoOutC = (double *) PyArray_DATA(rhoOut);
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Evolve rho matrices (MPP)
    for(int ii=0; ii<nt; ii++ ){
        // Evolve rho at t = tc[ii]
        while (N<tc[ii]-log(kscale)){
            flag = r8_rkf45(evolveRho1, 2*nF + 2*nF*2*nF, r, rp, &N, tc[ii]-log(kscale), &rtol, atol, flag, paramsIn );
            if (flag == 50){cout<< "\n \n \n Integrator failed at time N = " <<N <<" \n \n \n"; return Py_BuildValue("d", N);}
            flag = -2;
        }
        // Store timestep
        rhoOutC[ii*size] = N+log(kscale);        
        // Store the Fields, Velocities and Rho
        for(int i=0; i<2*nF + 2*nF*2*nF; i++)
        {
            rhoOutC[ii*size + 1 + i] = r[i];
        }
    }
    // Delete used arrays
    delete [] r; delete [] rp;
    delete [] paramsIn;

    return PyArray_Return(rhoOut);
}

static PyObject* MT_MPPSigma(PyObject* self, PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Gives the evolution of the phase space two-point function (sigma) using the MPP formalism. 
    Arguments
    ---------
    t : PyArrayObject*
        1D Array containing the values of each time-step.
    k : double
        The value of the k mode
    initialCs : PyArrayObject*
        1D Array (len=2*nF) containing the initial values of fields and velocities.
    params : PyArrayObject*
        1D Array (len=nP) containing the parameters of the model.
    rho : PyArrayObject*
        2D Array (len=2) containing the value of MPP2 needed to compute Sigma.
    full : bool
        If True, the function returns an array with dimension size = 2+2*nF+2*nF*2*nF),
        if False, the function returns an array with dimension size = 2.
    ts : double
        Optional parameter. It specifies the desired time-step at which we compute sigma.
        If not given, the function returns the entire evolution.
    Returns
    -------
    sigOut : PyArrayObject*
        If ts is given,  1D array (len=size). If ts is not given, 2D array (t_size, size).
    
    Description
    -----------
    This function computes the two-point correlation function using the MPP computed with the function MT_MPP2.
    The bool variable full controls number of columns of the returning array. If full = True, the function returns the time value, the 2pt function of zeta, 
    the values of the fields and the velocities, the phase-space two-point correlation function.

    Python Prototype
    ----------------
    twoPt = PyT.MPPSigma(t, k, backExitMinus, params, rho, full, ts)
    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------

    // Convert PyObjects to C-Objects
    PyArrayObject *rho, *params, *initialCs, *t;            // Input PyArrayObject
    PyArrayObject *sigOut;                                  // Output PyarrayObject
    double k, *CinitialCs, *Cparams, *tc;                   // Input doubles array
    double Ns = -1;                                         // Optional double
    bool full;                                              // Input boolean
    // Parsing args into variables
    if (!PyArg_ParseTuple(args, "O!dO!O!O!b|d", &PyArray_Type, &t, &k, &PyArray_Type, &initialCs,&PyArray_Type, &params, &PyArray_Type, &rho, &full, &Ns))
    {printf("Err Args\n"); return NULL;}

    CinitialCs = pyvector_to_Carray(initialCs);     // Convert initial values in C-array
    tc = pyvector_to_Carray(t);                     // Convert time to C-array
    Cparams = pyvector_to_Carray(params);           // Convert params to C-array
    double *mppRho = pyvector_to_Carray(rho);       // Convert rho to C-array
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Define the model and compute the needed quantities
    model mm;           // define model
    potential pott;     // define potential
    // define number of fields then check the size of initial conditions array (initialCs)
    int nF=mm.getnF(); if (2*nF!=size_pyvector(initialCs)){cout<< "\n \n \n field space array not of correct length, not proceeding further \n \n \n";    Py_RETURN_NONE;}
    //vector with initial conditions
    vector<double> vectIn; vectIn = vector<double>(CinitialCs, CinitialCs + 2*nF);
    // define number of parameters then check the size of parameters array (params)
    int nP = mm.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length, not proceeding further \n \n \n";  Py_RETURN_NONE;}
    // vector with parameters
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
    
    // we use a scaling below that we rescale back at the end (so the final answer is as if the scaling was never there -- this helps standarise the rtol and atol needed for the same model run with differnet initial conditions
    double kn = 1.0; 
    double kscale = k;    
    double Nstart=tc[0] - log(kscale);
    double N = Nstart;  // reset N
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the initial values of Sigma and initialize the array

    // Set parametes
    double* paramsIn; // array of parameters to pass to LHS of ODE routine
    paramsIn = new double[1+nP];    // define size of paramsIN
    for(int i=0; i<nP;i++){paramsIn[i]=Vparams[i];} // initialize paramsIN
    paramsIn[nP]=kn;

    // Declare variables for powerspectrum of zeta
    vector<double> Ni;   // set array for gauge transformation
    double zz=0;         // power spectrum of zz

    // Instance of Sigma object wich fixs initial conditions
    sigma sig(nF, kn, Nstart, vectIn, Vparams);

    // Set up array for sigma
    double* y = new double[2*nF + 2*nF*2*nF]; // set up array for evolution
    double* s = new double[2*nF + 2*nF*2*nF]; // set up array for ics
    for(int i=0; i<2*nF;i++){y[i] = CinitialCs[i]; s[i] = CinitialCs[i];} // fix values of input array
    for(int i=0; i<2*nF;i++){for(int j=0;j<2*nF;j++){y[2*nF + i + 2*nF*j] = sig.getS(i, j); s[2*nF + i + 2*nF*j] = sig.getS(i, j);}}
    // Set up field vector
    vector<double> fieldIn(2*nF);
    fieldIn = vector<double>(y,y+2*nF);
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Set dimension
    int nt = t->dimensions[0];
    int size;
    if(full==false){size = 1 + 1;}
    if(full==true){size = 1 + 2*nF + 1 + 2*nF*2*nF;}
    int sizeR = 1 + 2*nF + 2*nF*2*nF;

    // If Ns is not given, the function returns the entire evolution
    if(Ns == -1){
        // Set up the output array sigOut
        npy_intp dims[2];
        dims[1]=size; dims[0]=nt;   // size: number of columns, nt: number of rows
        double * sigOutC;
        sigOut = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
        sigOutC = (double *) PyArray_DATA(sigOut);

        // Set the array to store rho
        vector<double> r;

        // Compute sigma for each time-step
        for(int ii=0; ii<nt; ii++ ){
            N = tc[ii] - log(kscale);
            // Store rho[ii]
            r = vector<double>(mppRho + ii*sizeR + 1, mppRho + (ii+1)*sizeR);

            // Update fields
            for(int i = 0; i < 2*nF; i++){y[i] = r[i];};

            double sum = 0.;
            // calculate sigma
            for(int i = 0; i < 2*nF; i++){
                for(int j = 0; j < 2*nF; j++)
                {
                    sum = 0.0;
                    for(int a=0; a<2*nF; a++){
                        for(int b=0; b<2*nF; b++)
                        {
                            sum = sum + s[2*nF + 2*nF*a + b]*r[2*nF + a + 2*nF*i]*r[2*nF + b + 2*nF*j];
                        }
                    }
                    y[2*nF + 2*nF*i + j] = sum;
                }
            }
            fieldIn = vector<double>(y,y+2*nF);
            sigOutC[ii*size] = N + log(kscale);
            // Calculate zz
            Ni=mm.N1(fieldIn,Vparams,N); // calculate N,i array
            zz=0;
            for(int i=0; i<2*nF;i++){
                for(int j=0; j<2*nF; j++){
                    zz=zz+Ni[i]*Ni[j]*y[2*nF + 2*nF*i + j];
                }
            }
            //store zz
            sigOutC[ii*size+1] = zz/kscale/kscale/kscale;
            // Store the Fields
                if (full==true){
                for(int i=0; i<2*nF; i++)
                {
                    sigOutC[ii*(size)+i+2] = y[i];
                }
                // Store Sigma's
                for(int i = 2*nF; i<2*nF + 2*nF*2*nF; i++)
                {
                    sigOutC[ii*(size)+i+2] = y[i]/kscale/kscale/kscale;
                }
            }
        }
        // delete arrays
        delete [] y;  delete [] s;
        delete [] paramsIn;

        return PyArray_Return(sigOut);
    }
    // If Ns is given, the function returns sigma[t[Ns]]
    else
    {
        // Set dimension and create array sigOut (1D array in this case)
        npy_intp dims[1];
        dims[0] = size;
        double * sigOutC;
        sigOut = (PyArrayObject*) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
        sigOutC = (double *) PyArray_DATA(sigOut);
        // Covert rho to double array
        double *mppRho = pyvector_to_Carray(rho);
        // Store the index
        int ii = int(Ns);
        N = tc[ii] - log(kscale);
        // Store rho[ii] into vector
        vector<double> r;
        r = vector<double>(mppRho + 1 + ii*sizeR, mppRho + (ii+1)*sizeR);

        // Update fields
        for(int i = 0; i < 2*nF; i++){y[i] = r[i];};
        // calculate sigma
        double sum = 0.;
        for(int i = 0; i < 2*nF; i++){
            for(int j = 0; j < 2*nF; j++)
            {
                sum = 0.0;
                for(int a=0; a<2*nF; a++){
                    for(int b=0; b<2*nF; b++)
                    {
                        sum = sum + s[2*nF + 2*nF*a + b]*r[2*nF + a + 2*nF*i]*r[2*nF + b + 2*nF*j];
                    }
                }
                y[2*nF + 2*nF*i + j] = sum;
            }
        }
        fieldIn = vector<double>(y,y+2*nF);
        // Store time
        sigOutC[0] = N+log(kscale);
        // Calculate zz
        Ni=mm.N1(fieldIn,Vparams,N); // calculate N,i array
        zz=0;
        for(int i=0; i<2*nF;i++){
            for(int j=0; j<2*nF; j++){
                zz=zz+Ni[i]*Ni[j]*y[2*nF + 2*nF*i + j];
            }
        }
        // Store zz
        sigOutC[1] = zz/kscale/kscale/kscale;
        // Store the Fields
        for(int i=0; i<2*nF; i++)
        {
            sigOutC[i+2] = y[i];
        }
        // Store Sigma's
        for(int i = 2*nF; i<2*nF + 2*nF*2*nF; i++)
        {
            sigOutC[i+2] = y[i]/kscale/kscale/kscale;
        }
        // delete time
        delete [] y;  delete [] s;
        delete [] paramsIn;

        return PyArray_Return(sigOut);
    }
}

static PyObject* MT_MPP3(PyObject* self, PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Gives the entire evolution of 3 MPP matrices with three indices (one for each permutation of the k's)
    and three MPP matrices with two indices (one for each k)

    Arguments
    ---------
    t : PyArrayObject*
        1D Array containing the values of each time-step.
    k1 : double
        The value of the first k mode.
    k2 : double
        The value of the second k mode.
    k3 : double
        The value of the third k mode.    
    initialCs : PyArrayObject*
        1D Array (len=2*nF) containing the initial values of fields and velocities.
    params : PyArrayObject*
        1D Array (len=nP) containing the parameters of the model.
    tols : PyArrayObject*
        1D Array (len=2) containing the absolute and relative tolerance for the numerical integrator RK45.
        
    Returns
    -------
    rhoOut : PyArrayObject
            2D array (size_t, 1 + 2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF)) containing for each time step (on each row): 
            time value, field and velocity values, 3 MPP2, 3 MPP3.

    Description
    -----------
    This function computes the MPP 3 indices matrix for each value of t.
    It evolves 3 MPP2 matrices (1 for each k mode), and 3 MPP3 matrices (1 for each permutation of the three k modes), 
    using the function evolveRho3 to compute the derivatives of all matrices.

    Python Prototype
    ----------------
    mpp3 = PyT.MPP3(t, k1, k2, k3, initialCs, params, tols)

    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------

    cout << "Enter MPP\n";

    // Convert PyObjects to C-Objects
    PyArrayObject *initialCs, *t, *params, *tols;           // Input PyArrayObject
    PyArrayObject *rhoOut;                                  // Output PyArrayObject
    double k1, k2, k3, *CinitialCs, *tc, *Cparams, *tolsC;  // Input doubles
    double rtol, atol;                                      // Input rel and abs tolerance
    // Parsing args into variables
    if (!PyArg_ParseTuple(args, "O!dddO!O!O!", &PyArray_Type, &t, &k1,&k2,&k3, &PyArray_Type, &initialCs,&PyArray_Type,&params,&PyArray_Type,&tols)) {
    return NULL; }
    CinitialCs = pyvector_to_Carray(initialCs);     // Convert initial values in a C-array
    tc = pyvector_to_Carray(t);                     // Convert time in a C-array
    tolsC = pyvector_to_Carray(tols);               // Convert tolerance to a C-array
    Cparams = pyvector_to_Carray(params);           // Convert params to c-array
    // Check size tols
    if (2!=size_pyvector(tols)){cout<< "\n \n \n incorrect tolorances input, using defaults  \n \n \n";
        atol = pow(10,-8.); rtol = pow(10,-8.);}
    else {
        atol =tolsC[0];rtol = tolsC[1];
    }
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Define the model and compute the needed quantities
    model mm;           // define model
    potential pott;     // define potential
    // Define number of fields then check the size of ics array (initialCs)
    int nF=mm.getnF(); if (2*nF!=size_pyvector(initialCs)){cout<< "\n \n \n field space array not of correct length, not proceeding further \n \n \n";    Py_RETURN_NONE;}
    // define vector with initial conditions
    vector<double> vectIn; vectIn = vector<double>(CinitialCs, CinitialCs + 2*nF);
    // define number of parameters then check the size of parameters array (params)
    int nP = mm.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length, not proceeding further \n \n \n";  Py_RETURN_NONE;}
    // define vector with parameters
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
    
    // we use a scaling below that we rescale back at the end (so the final answer is as if the scaling was never there -- this helps standarise the rtol and atol needed for the same model run with differnet initial conditions
    double kscale = (k1+k2+k3)/3.; 
    double k1n = k1/kscale; double k2n = k2/kscale; double k3n = k3/kscale;   
    double Nstart = tc[0] - log(kscale);
    double N = Nstart; // Reset N
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the initial values of rho(MPP) and initialize the vector

    // instance of rho1 object which fixs ics
    Rho1 rho11(nF, k1n, Nstart, vectIn, Vparams);
    Rho1 rho12(nF, k2n, Nstart, vectIn, Vparams);
    Rho1 rho13(nF, k3n, Nstart, vectIn, Vparams);

    // instance of rho2 object which fixs ics
    Rho2 rho21(nF, k1n, k2n, k3n, Nstart, vectIn, Vparams);
    Rho2 rho22(nF, k2n, k1n, k3n, Nstart, vectIn, Vparams);
    Rho2 rho23(nF, k3n, k1n, k2n, Nstart, vectIn, Vparams);

    // Store fields, rho1 and rho2
    double * r = new double[2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF)];
    // store fields
    for(int i = 0; i < 2*nF; i++){r[i] = CinitialCs[i];}
    // store rho1
    for(int i = 0; i< 2*nF; i++){
        for(int j = 0; j < 2*nF; j++){
            r[2*nF + i + 2*nF*j] = rho11.getR(i,j);
            r[2*nF + 2*nF*2*nF + i + 2*nF*j] = rho12.getR(i,j);
            r[2*nF + 2*(2*nF*2*nF) + i + 2*nF*j] = rho13.getR(i,j);
        }
    }
    // store rho2
    for(int i = 0; i < 2*nF; i++){
        for(int j = 0; j < 2*nF; j++){
            for(int k = 0; k < 2*nF; k++){
                r[2*nF + 3*(2*nF*2*nF) + i + 2*nF*j + 2*nF*2*nF*k] = rho21.getR(i,j,k);
                r[2*nF + 3*(2*nF*2*nF) + (2*nF*2*nF*2*nF) + i + 2*nF*j + 2*nF*2*nF*k] = rho22.getR(i,j,k);
                r[2*nF + 3*(2*nF*2*nF) + 2*(2*nF*2*nF*2*nF) + i + 2*nF*j + 2*nF*2*nF*k] = rho23.getR(i,j,k);
            }
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the derivative of MPP

    // Set parameters
    double* paramsIn; // Array for parameters of RHS of ODE routine
    paramsIn = new double[3 + nP];
    for(int i = 0; i < nP; i++){paramsIn[i] = Vparams[i];}
    paramsIn[nP] = k1n;
    paramsIn[nP + 1] = k2n;
    paramsIn[nP + 2] = k3n;

    // Set up array for derivatives of MPP2 and MPP3
    double *rp = new double[2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF)];
    evolveRho2(N, r, rp, paramsIn);

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Set output array rhoOut

    // Set dimension
    npy_intp dims[2];
    int size;                   // number of columns
    int nt = t->dimensions[0];  // number of rows
    size = 1 + 2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF);
    dims[1] = size; dims[0] = nt;
    // Define output array
    double* rhoOutC;
    rhoOut = (PyArrayObject *) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    rhoOutC = (double *) PyArray_DATA(rhoOut);

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Evolve rho matrices (Fields, Velocities, MPP2 and MPP3)
    int flag = -1;
    for(int ii = 0; ii<nt; ii++){
        while (N < (tc[ii]-log(kscale))){
            if (ii == 0){
            flag = r8_rkf45(evolveRho2, 2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF), r, rp, &N, tc[ii]-log(kscale), &rtol, atol, flag, paramsIn);
            if (abs(flag) == 0) {
                // Success: Integration step completed successfully.
                flag = -2;
                continue;
            } else if (flag == 3) {
                std::cerr << "Warning: Relative error tolerance was too small. Temporarily relaxing tolerances for one step." << std::endl;
                // RELERR was too small; RELERR has been increased appropriately.
                double originalRtol = rtol;
                double originalAtol = atol;

                rtol *= 10;
                flag = -1;
                flag = r8_rkf45(evolveRho2, 2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF), r, rp, &N, tc[ii]-log(kscale), &rtol, atol, flag, paramsIn);
                if (flag != 0){
                    cout<< "\n \n \n Integrator failed at time N = " <<N <<" \n \n \n"; 
                    return Py_BuildValue("d", N);
                }

                rtol = originalRtol;
                flag = -2;
                
            } else if (flag == 6) {
                // Accuracy could not be achieved; temporarily relax tolerances.
                std::cerr << "Warning: Requested accuracy could not be achieved. Temporarily relaxing tolerances for one step." << std::endl;
                double originalRtol = rtol;
                double originalAtol = atol;
                int countRetry = 0;
                int maxRetry = 5;
    
                while (countRetry < maxRetry) {
                    rtol *= 10;
                    atol *= 10;
                    flag = -1;
                    flag = r8_rkf45(evolveRho2, 2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF), r, rp, &N, tc[ii]-log(kscale), &rtol, atol, flag, paramsIn);
                    if (flag == 0){
                        rtol = originalRtol;
                        atol = originalAtol;
                        flag = -2;
                        break;
                    }
                    else {
                        countRetry += 1;
                    }
                }

                rtol = originalRtol;
                atol = originalAtol;
            }
            }
        }


        // while (N < (tc[ii]-log(kscale))){
        //     flag = r8_rkf45(evolveRho2, 2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF), r, rp, &N, tc[ii]-log(kscale), &rtol, atol, flag, paramsIn);
        //     if (flag == 50){cout<< "\n \n \n Integrator failed at time N = " <<N <<" \n \n \n"; return Py_BuildValue("d", N);}
        //     flag=-2;
        // }
        // Store timestep
        rhoOutC[ii*size] =  N+log(kscale);
        // Store the MPP matrices
        for(int i=0; i<2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF);i++){
            rhoOutC[ii*size+1+i] =  r[i] ; 
        }
    }
    // Delete arrays
    delete [] paramsIn; delete [] rp; delete [] r;
    
    return PyArray_Return(rhoOut);
}

static PyObject* MT_MPPAlpha(PyObject* self, PyObject *args)
{
    /* -----------------------------------------------------------------------------------------------------------------------------------------------------
    Gives the evolution of the phase space three-point function (alpha) using the MPP formalism. 
    Arguments
    ---------
    t : PyArrayObject*
        1D Array containing the values of each time-step.
    k1 : double
        The value of the first k mode.
    k2 : double
        The value of the second k mode.
    k3 : double
        The value of the third k mode. 
    initialCs : PyArrayObject*
        1D Array (len=2*nF) containing the initial values of fields and velocities.
    params : PyArrayObject*
        1D Array (len=nP) containing the parameters of the model.
    rho : PyArrayObject*
        2D Array containing the values of the MPPs needed.
    full : bool
        If True, the function returns an array with dimension size = 5 + 2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*2*nF),
        if False, the function returns an array with dimension size = 5.
    ts : double
        Optional parameter. It specifies the desired time-step at which we compute sigma.
        If not given, the function returns the entire evolution.
    Returns
    -------
    alpOut : PyArrayObject*
        If ts is given,  1D array (len=size). If ts is not given, 2D array (t_size, size).
    
    Description
    -----------
    This function computes the three-point correlation function using the MPP computed with the function MT_MPP3.
    The bool variable full controls number of columns of the returning array. If full = True, the function returns the time value, the value of Pz(k1), 
    the value of Pz(k2), the value of Pz(k3), the value of fnl(k1,k2,k3), the values of the fields and the velocities, the three real 2pt functions, the three imaginary 2pt functions,
    the phase-space three correlation function. If full = False, the function returns the time value, the value of Pz(k1), 
    the value of Pz(k2), the value of Pz(k3), the value of fnl(k1,k2,k3). 
    The optional parameter ts controls the number of rows of the returning array. If given, the function returns the output only for t = t[ts].
    If not given, the function returns the output for each value in t. 

    Python Prototype
    ----------------
    threePt = PyT.MPPAlpha(t, k1, k2, k3, backExitMinus, params, rho, full, ts)
    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Convert PyObjects to C-Objects
    PyArrayObject *initialCs, *t, *params, *MppRho;         // Input PyArrayObject
    PyArrayObject *alpROut;                                 // Output PyArrayObject
    double k1, k2, k3, *CinitialCs, *tc, *Cparams, *rho;    // Input doubles array
    double Ns = -1;                                         // Optional double
    bool full;                                              // Input boolean
    // Parsing args into variables
    if (!PyArg_ParseTuple(args, "O!dddO!O!O!b|d", &PyArray_Type, &t, &k1,&k2,&k3, &PyArray_Type, &initialCs,&PyArray_Type,&params,&PyArray_Type,&MppRho, &full, &Ns)) {
        return NULL; }
    CinitialCs = pyvector_to_Carray(initialCs);             // Convert ics in C-array
    tc = pyvector_to_Carray(t);                             // Convert time in C-array
    rho = pyvector_to_Carray(MppRho);                       // Convert rho in C-array
    Cparams = pyvector_to_Carray(params);                   // Convert params in C-array

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Define the model and compute the needed quantities
    model mm;                   // Define model
    potential pott;             // Define potential
    // Define number of fields then check size ics (initialCs)
    int nF=mm.getnF(); if (2*nF!=size_pyvector(initialCs)){cout<< "\n \n \n field space array not of correct length, not proceeding further \n \n \n";    Py_RETURN_NONE;}
    // vector with ics
    vector<double> vectIn; vectIn = vector<double>(CinitialCs, CinitialCs + 2*nF);
    // Define number of parameters then check size of parameters (params)
    int nP = mm.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length, not proceeding further \n \n \n";  Py_RETURN_NONE;}
    // vector parameters
    vector<double> Vparams; Vparams = vector<double>(Cparams, Cparams +  nP);
    
    // we use a scaling below that we rescale back at the end (so the final answer is as if the scaling was never there -- this helps standarise the rtol and atol needed for the same model run with differnet initial conditions
    double kscale = (k1+k2+k3)/3.; 
    double k1n = k1/kscale; double k2n = k2/kscale; double k3n = k3/kscale;   
    double Nstart=tc[0] - log(kscale);

    double N=Nstart; // Reset N
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the initial values of rho(MPP) and initialize the vector
    
    // instance of sigma object which fix ics;
    sigma sig1(nF, k1n, Nstart, vectIn, Vparams);
    sigma sig2(nF, k2n, Nstart, vectIn, Vparams);
    sigma sig3(nF, k3n, Nstart, vectIn, Vparams);

    // instance of sigma imaginary
    sigmaI sig1I(nF, k1n, Nstart, vectIn,Vparams)  ; 
    sigmaI sig2I(nF, k2n, Nstart, vectIn,Vparams)  ;
    sigmaI sig3I(nF, k3n, Nstart, vectIn,Vparams)  ;

    // instance of alpha objects which fix ics;
    alpha alp(nF, k1n, k2n, k3n, Nstart, vectIn, Vparams);

    // Define vector to store rho
    vector<double> r;

    // Store sigmas
    double *s = new double[6*(2*nF*2*nF)];
    for (int i = 0; i < 2*nF; i++){
        for (int j = 0; j < 2*nF; j++){
            // Real
            s[0*(2*nF*2*nF) + i + 2*nF*j] = sig1.getS(i,j);
            s[1*(2*nF*2*nF) + i + 2*nF*j] = sig2.getS(i,j);
            s[2*(2*nF*2*nF) + i + 2*nF*j] = sig3.getS(i,j);
            // Imaginary
            s[3*(2*nF*2*nF) + i + 2*nF*j] = sig1I.getS(i,j);
            s[4*(2*nF*2*nF) + i + 2*nF*j] = sig2I.getS(i,j);
            s[5*(2*nF*2*nF) + i + 2*nF*j] = sig3I.getS(i,j);
        }
    }

    // Store alpha
    double *ainit = new double[ 2*nF * 2*nF * 2*nF ];
    for(int i = 0; i < 2*nF; i++){
        for(int j = 0; j < 2*nF; j++){
            for(int k = 0; k < 2*nF; k++){
                ainit[i + 2*nF*j + (2*nF*2*nF)*k] = alp.getA(i,j,k);
            }
        }
    }
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the derivative of MPP

    // Set parameters
    double* paramsIn;                                           // Array for parameters of RHS of ODE routine
    paramsIn = new double[3 + nP];                              // Define size of params In
    for(int i = 0; i < nP; i++){paramsIn[i] = Vparams[i];}      // Initialize paramsIn
    paramsIn[nP] = k1n;
    paramsIn[nP + 1] = k2n;
    paramsIn[nP + 2] = k3n;

    double ZZZ = 0., ZZ1 = 0., ZZ2 = 0., ZZ3 = 0.; // for zeta zeta calcs
    vector<double> Ni, Nii1, Nii2, Nii3 ; // for N transofrms to get to zeta

    // array for alpha evolution
    double *y = new double[2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*2*nF];

    int sizeR = 1 + 2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF);

    int flag = -1;

    // Run alpha **************************************************

    vector<double> fieldIn(2*nF);   //vector to store fields for gauge coeffiecients

    // If Ns is not given the function returns the entire evolution for alpha
    if(Ns == -1){
        // Set dimension output
        npy_intp dims[2];
        int nt = t->dimensions[0];
        int size;
        if (full==false){size = 5;}
        if (full==true){size = 5 + 2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*2*nF;}
        dims[1] = size; dims[0] = nt;
        double* alpROutC;
        alpROut = (PyArrayObject *) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
        alpROutC = (double *) PyArray_DATA(alpROut);

        // Evolution of alpha
        for(int ii=0; ii<nt; ii++){
            N = tc[ii] - log(kscale);    // Compute time-step
            r = vector<double>(rho + 1 + ii*sizeR, rho + (ii+1)*sizeR); //Store value of rho at each time-step
            // store fields
            for(int i = 0; i < 2*nF; i++){y[i] = r[i];}
            // Calculate and store sigma
            for(int i = 0; i < 2*nF; i++){
                for(int j = 0; j < 2*nF; j++)
                {
                    // Reset
                    double sumR1 = 0.;
                    double sumR2 = 0.;
                    double sumR3 = 0.;
                    double sumI1 = 0.;
                    double sumI2 = 0.;
                    double sumI3 = 0.;
                    // Calculate sigma
                    for(int a=0; a<2*nF; a++){
                        for(int b=0; b<2*nF; b++)
                        {
                            // Real
                            sumR1 = sumR1 + s[0*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 0*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 0*(2*nF*2*nF) + b + 2*nF*j];
                            sumR2 = sumR2 + s[1*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 1*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 1*(2*nF*2*nF) + b + 2*nF*j];
                            sumR3 = sumR3 + s[2*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 2*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 2*(2*nF*2*nF) + b + 2*nF*j];
                            // Imaginary
                            sumI1 = sumI1 + s[3*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 0*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 0*(2*nF*2*nF) + b + 2*nF*j];
                            sumI2 = sumI2 + s[4*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 1*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 1*(2*nF*2*nF) + b + 2*nF*j];
                            sumI3 = sumI3 + s[5*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 2*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 2*(2*nF*2*nF) + b + 2*nF*j];
                        }
                    }
                    // Store
                    y[2*nF + 0*(2*nF*2*nF) + 2*nF*i + j] = sumR1;
                    y[2*nF + 1*(2*nF*2*nF) + 2*nF*i + j] = sumR2;
                    y[2*nF + 2*(2*nF*2*nF) + 2*nF*i + j] = sumR3;
                    y[2*nF + 3*(2*nF*2*nF) + 2*nF*i + j] = sumI1;
                    y[2*nF + 4*(2*nF*2*nF) + 2*nF*i + j] = sumI2;
                    y[2*nF + 5*(2*nF*2*nF) + 2*nF*i + j] = sumI3;
                }
            }
            
            // Calculate and store alpha
            double term0, term1, term2, term3;
            for (int a = 0; a < 2*nF; a++){
                for(int b = 0; b < 2*nF; b++){
                    for(int c = 0; c < 2*nF; c++){
                        // reset term
                        term0 = 0.;
                        term1 = 0.;
                        term2 = 0.;
                        term3 = 0.;
                        // sum over i,j,k
                        for(int i = 0; i < 2*nF; i++){
                            for(int j = 0; j < 2*nF; j++){
                                for(int k = 0; k < 2*nF; k++){

                                    term0 = term0 + r[2*nF + 2*nF*a + i]*r[2*nF + (2*nF*2*nF) + 2*nF*b + j]*r[2*nF + 2*(2*nF*2*nF) + 2*nF*c + k]*ainit[i + 2*nF*j + (2*nF*2*nF)*k];
                                    // sum over l
                                    for(int l = 0; l < 2*nF; l++){
                                        term1 = term1 + r[2*nF + 3*(2*nF*2*nF) + (2*nF*2*nF)*a + 2*nF*i + j] * r[2*nF + (2*nF*2*nF) + 2*nF*b + k] * r[2*nF + 2*(2*nF*2*nF) + 2*nF*c + l]*( s[(2*nF*2*nF) + 2*nF*i + k]*s[2*(2*nF*2*nF) + 2*nF*j + l] - s[4*(2*nF*2*nF) + 2*nF*i + k]*s[5*(2*nF*2*nF) + 2*nF*j + l] );
                                        term2 = term2 + r[2*nF + 3*(2*nF*2*nF) + (2*nF*2*nF*2*nF) + (2*nF*2*nF)*b + 2*nF*i + j] * r[2*nF + 2*nF*a + k] * r[2*nF + 2*(2*nF*2*nF) + 2*nF*c + l]*( s[2*nF*k + i]*s[2*(2*nF*2*nF) + 2*nF*j + l] - s[3*(2*nF*2*nF) + 2*nF*k + i]*s[5*(2*nF*2*nF) + 2*nF*j + l] );
                                        term3 = term3 + r[2*nF + 3*(2*nF*2*nF) + 2*(2*nF*2*nF*2*nF) + (2*nF*2*nF)*c + 2*nF*i + j] * r[2*nF + 2*nF*a + k] * r[2*nF + (2*nF*2*nF) + 2*nF*b + l]*( s[2*nF*k + i]*s[(2*nF*2*nF) + 2*nF*l + j] - s[3*(2*nF*2*nF) + 2*nF*k + i]*s[4*(2*nF*2*nF) + 2*nF*l + j] );
                                    }
                                }
                            }
                        }
                        // Store alpha
                        y[2*nF + 6*(2*nF*2*nF) + (2*nF*2*nF)*a + (2*nF)*b + c] = term0 + term1 + term2 + term3;
                    }
                }
            }

            fieldIn = vector<double>(y,y+2*nF);
            Ni=mm.N1(fieldIn,Vparams,N); // calculate N,i array
            Nii1=mm.N2(fieldIn,Vparams,k1n,k2n,k3n,N); // claculate N,ij array for first arrangement of ks
            Nii2=mm.N2(fieldIn,Vparams,k2n,k1n,k3n,N); // for second
            Nii3=mm.N2(fieldIn,Vparams,k3n,k1n,k2n,N); // etc
            
            // Compute zeta-zeta
            ZZ1=0.;
            ZZ2=0.;
            ZZ3=0.;
            for(int i=0; i<2*nF;i++){for(int j=0; j<2*nF; j++){
                ZZ1=ZZ1+Ni[i]*Ni[j]*(y[2*nF + 2*nF*i + j] );
                ZZ2=ZZ2+Ni[i]*Ni[j]*y[2*nF + (2*nF*2*nF) + 2*nF*i + j];
                ZZ3=ZZ3+Ni[i]*Ni[j]*y[2*nF + 2*(2*nF*2*nF) + 2*nF*i + j];
            }}   
            // Compute zeta-zeta-zeta
            ZZZ=0.;
            for(int i=0; i<2*nF;i++){for(int j=0; j<2*nF;j++){for(int k=0; k<2*nF;k++){
                ZZZ=ZZZ + Ni[i]*Ni[j]*Ni[k]*y[2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*i + j*2*nF+ k];
                for(int l=0; l<2*nF;l++){ZZZ=ZZZ+(Nii1[i+j*2*nF]*Ni[k]*Ni[l]*y[2*nF + 1*(2*nF*2*nF) + 2*nF*i+k]*y[2*nF + 2*(2*nF*2*nF)+2*nF*j+l]
                                                +Nii2[i+j*2*nF]*Ni[k]*Ni[l]*y[2*nF + 0*(2*nF*2*nF) + 2*nF*i+k]*y[2*nF + 2*(2*nF*2*nF) + 2*nF*j+l]
                                                +Nii3[i+j*2*nF]*Ni[k]*Ni[l]*y[2*nF + 0*(2*nF*2*nF) + 2*nF*i+k]*y[2*nF + 1*(2*nF*2*nF) + 2*nF*j+l]);
            }}}}
            
            // Store output array
            alpROutC[ii*size] =  N+log(kscale);                                 // time-step
            alpROutC[ii*size+1] = ZZ1/kscale/kscale/kscale;                     // zz(k1)
            alpROutC[ii*size+2] = ZZ2/kscale/kscale/kscale;                     // zz(k2)              
            alpROutC[ii*size+3] = ZZ3/kscale/kscale/kscale;                     // zz(k2)
            alpROutC[ii*size+4] = ZZZ/kscale/kscale/kscale/kscale/kscale/kscale;// zzz(k1,k2,k3)
            
            if(full==true){
                // store fields and velocities
                for(int i=0; i<2*nF ;i++){
                    alpROutC[ii*size+5+i] =  y[i];}
                // store sigmas
                for(int i=2*nF; i<2*nF + 6*(2*nF*2*nF); i++){
                    alpROutC[ii*size+5+i] =  y[i]/kscale/kscale/kscale;}
                // store alpha
                for(int i=2*nF + 6*(2*nF*2*nF); i<2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*2*nF; i++){
                    alpROutC[ii*size+5+i] =  y[i]/kscale/kscale/kscale/kscale/kscale/kscale;}
            }
        }
    } 
    // If Ns is given, the fuction return alpha[t(Ns)]
    else{
        // Set dimension and create array alpROut
        npy_intp dims[1];
        int size = 5 + 2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*2*nF;;
        dims[0] = size;
        double* alpROutC;
        alpROut = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
        alpROutC = (double *) PyArray_DATA(alpROut);

        // Define time-step
        int ii = int(Ns);
        N = tc[ii] - log(kscale); // time
        r = vector<double>(rho + 1 + ii*sizeR, rho + (ii+1)*sizeR); //rho

        // store fields
        for(int i = 0; i < 2*nF; i++){y[i] = r[i];}

        // Calculate and store sigma
        for(int i = 0; i < 2*nF; i++){
            for(int j = 0; j < 2*nF; j++)
            {
                // Reset
                double sumR1 = 0.;
                double sumR2 = 0.;
                double sumR3 = 0.;
                double sumI1 = 0.;
                double sumI2 = 0.;
                double sumI3 = 0.;
                // Calculate sigma
                for(int a=0; a<2*nF; a++){
                    for(int b=0; b<2*nF; b++)
                    {
                        // Real
                        sumR1 = sumR1 + s[0*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 0*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 0*(2*nF*2*nF) + b + 2*nF*j];
                        sumR2 = sumR2 + s[1*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 1*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 1*(2*nF*2*nF) + b + 2*nF*j];
                        sumR3 = sumR3 + s[2*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 2*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 2*(2*nF*2*nF) + b + 2*nF*j];
                        // Imaginary
                        sumI1 = sumI1 + s[3*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 0*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 0*(2*nF*2*nF) + b + 2*nF*j];
                        sumI2 = sumI2 + s[4*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 1*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 1*(2*nF*2*nF) + b + 2*nF*j];
                        sumI3 = sumI3 + s[5*(2*nF*2*nF) + 2*nF*a + b]*r[2*nF + 2*(2*nF*2*nF) + a + 2*nF*i]*r[2*nF + 2*(2*nF*2*nF) + b + 2*nF*j];
                    }
                }
                // Store
                y[2*nF + 0*(2*nF*2*nF) + 2*nF*i + j] = sumR1;
                y[2*nF + 1*(2*nF*2*nF) + 2*nF*i + j] = sumR2;
                y[2*nF + 2*(2*nF*2*nF) + 2*nF*i + j] = sumR3;
                y[2*nF + 3*(2*nF*2*nF) + 2*nF*i + j] = sumI1;
                y[2*nF + 4*(2*nF*2*nF) + 2*nF*i + j] = sumI2;
                y[2*nF + 5*(2*nF*2*nF) + 2*nF*i + j] = sumI3;
            }
        }
        

        // Calculate and store alpha
        double term0, term1, term2, term3;

        for (int a = 0; a < 2*nF; a++){
            for(int b = 0; b < 2*nF; b++){
                for(int c = 0; c < 2*nF; c++){
                    // reset term
                    term0 = 0.;
                    term1 = 0.;
                    term2 = 0.;
                    term3 = 0.;
                    // sum over i,j,k
                    for(int i = 0; i < 2*nF; i++){
                        for(int j = 0; j < 2*nF; j++){
                            for(int k = 0; k < 2*nF; k++){

                                term0 = term0 + r[2*nF + 2*nF*a + i]*r[2*nF + (2*nF*2*nF) + 2*nF*b + j]*r[2*nF + 2*(2*nF*2*nF) + 2*nF*c + k]*ainit[i + 2*nF*j + (2*nF*2*nF)*k];
                                // sum over l
                                for(int l = 0; l < 2*nF; l++){
                                    term1 = term1 + r[2*nF + 3*(2*nF*2*nF) + (2*nF*2*nF)*a + 2*nF*i + j] * r[2*nF + (2*nF*2*nF) + 2*nF*b + k] * r[2*nF + 2*(2*nF*2*nF) + 2*nF*c + l]*( s[(2*nF*2*nF) + 2*nF*i + k]*s[2*(2*nF*2*nF) + 2*nF*j + l] - s[4*(2*nF*2*nF) + 2*nF*i + k]*s[5*(2*nF*2*nF) + 2*nF*j + l] );
                                    term2 = term2 + r[2*nF + 3*(2*nF*2*nF) + (2*nF*2*nF*2*nF) + (2*nF*2*nF)*b + 2*nF*i + j] * r[2*nF + 2*nF*a + k] * r[2*nF + 2*(2*nF*2*nF) + 2*nF*c + l]*( s[2*nF*k + i]*s[2*(2*nF*2*nF) + 2*nF*j + l] - s[3*(2*nF*2*nF) + 2*nF*k + i]*s[5*(2*nF*2*nF) + 2*nF*j + l] );
                                    term3 = term3 + r[2*nF + 3*(2*nF*2*nF) + 2*(2*nF*2*nF*2*nF) + (2*nF*2*nF)*c + 2*nF*i + j] * r[2*nF + 2*nF*a + k] * r[2*nF + (2*nF*2*nF) + 2*nF*b + l]*( s[2*nF*k + i]*s[(2*nF*2*nF) + 2*nF*l + j] - s[3*(2*nF*2*nF) + 2*nF*k + i]*s[4*(2*nF*2*nF) + 2*nF*l + j] );
                                }
                            }
                        }
                    }
                    // Store alpha
                    y[2*nF + 6*(2*nF*2*nF) + (2*nF*2*nF)*a + 2*nF*b + c] = term0 + term1 + term2 + term3;
                }
            }
        }

        fieldIn = vector<double>(y,y+2*nF);
        Ni=mm.N1(fieldIn,Vparams,N); // calculate N,i array
        Nii1=mm.N2(fieldIn,Vparams,k1n,k2n,k3n,N); // claculate N,ij array for first arrangement of ks
        Nii2=mm.N2(fieldIn,Vparams,k2n,k1n,k3n,N); // for second
        Nii3=mm.N2(fieldIn,Vparams,k3n,k1n,k2n,N); // etc
        
        // Compute zz
        ZZ1=0.;
        ZZ2=0.;
        ZZ3=0.;
        for(int i=0; i<2*nF;i++){for(int j=0; j<2*nF; j++){
            ZZ1=ZZ1+Ni[i]*Ni[j]*(y[2*nF + 2*nF*i + j] );
            ZZ2=ZZ2+Ni[i]*Ni[j]*y[2*nF + (2*nF*2*nF) + 2*nF*i + j];
            ZZ3=ZZ3+Ni[i]*Ni[j]*y[2*nF + 2*(2*nF*2*nF) + 2*nF*i + j];
        }}   

        // Compute zzz
        ZZZ=0.;
        for(int i=0; i<2*nF;i++){for(int j=0; j<2*nF;j++){for(int k=0; k<2*nF;k++){
            ZZZ=ZZZ + Ni[i]*Ni[j]*Ni[k]*y[2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*i + j*2*nF+ k];
            for(int l=0; l<2*nF;l++){ZZZ=ZZZ+(Nii1[i+j*2*nF]*Ni[k]*Ni[l]*y[2*nF + 1*(2*nF*2*nF) + 2*nF*i+k]*y[2*nF + 2*(2*nF*2*nF)+2*nF*j+l]
                                              +Nii2[i+j*2*nF]*Ni[k]*Ni[l]*y[2*nF + 0*(2*nF*2*nF) + 2*nF*i+k]*y[2*nF + 2*(2*nF*2*nF) + 2*nF*j+l]
                                              +Nii3[i+j*2*nF]*Ni[k]*Ni[l]*y[2*nF + 0*(2*nF*2*nF) + 2*nF*i+k]*y[2*nF + 1*(2*nF*2*nF) + 2*nF*j+l]);
        }}}}
        

        alpROutC[0] = N+log(kscale);                                    // time-step
        alpROutC[1] = ZZ1/kscale/kscale/kscale;                         // zz(k1)
        alpROutC[2] = ZZ2/kscale/kscale/kscale;                         // zz(k2)
        alpROutC[3] = ZZ3/kscale/kscale/kscale;                         // zz(k3)
        alpROutC[4] = ZZZ/kscale/kscale/kscale/kscale/kscale/kscale;    // zzz(k1,k2,k3)

        // Store fields
        if(full==true){
                    for(int i=0; i<2*nF ;i++){
            alpROutC[5+i] =  y[i] ;   }
        // Store sigma
        for(int i=2*nF; i<2*nF + 6*(2*nF*2*nF); i++){
            alpROutC[5+i] =  y[i]/kscale/kscale/kscale ;   }
        // Store alpha
        for(int i=2*nF + 6*(2*nF*2*nF); i<2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*2*nF; i++){
            alpROutC[5+i] =  y[i]/kscale/kscale/kscale/kscale/kscale/kscale ;   }
        }
    }

    // Delete vectors
    delete [] y;  delete [] paramsIn;
    delete [] s; delete [] ainit;
    
    return PyArray_Return(alpROut);
}

static PyObject* MT_getSigma(PyObject* self, PyObject *args)
{
    PyArrayObject *initialCs, *t, *params;
    PyArrayObject *sigOut;
    double *CinitialCs, *Ct, k, *Cparams;

    if (!PyArg_ParseTuple(args, "O!dO!O!", &PyArray_Type, &t, &k, &PyArray_Type, &initialCs,&PyArray_Type, &params))
    { return NULL;}

    CinitialCs = pyvector_to_Carray(initialCs);
    Ct = pyvector_to_Carray(t);
    Cparams = pyvector_to_Carray(params);

    model mm;
    potential pott;
    int nF = mm.getnF();
    int nP = mm.getnP();
    vector<double> vectIn = vector<double>(CinitialCs, CinitialCs + 2*nF);
    vector<double> Vparams = vector<double>(Cparams, Cparams + nP);
    double kn = 1.0;
    double Nstart = Ct[0] - log(k);

    int nt = 2;
    int size = 2*nF*2*nF;
    npy_intp dims[2];
    dims[0] = nt; dims[1] = size;
    double* sigOutC;
    sigOut = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    sigOutC = (double *) PyArray_DATA(sigOut);

    sigma sigR(nF, kn, Nstart, vectIn, Vparams);
    sigmaI sigI(nF, kn, Nstart, vectIn, Vparams);
    for(int i = 0; i < 2*nF; i++){
        for(int j =0; j <2*nF; j++){
            sigOutC[i + 2*nF*j] = sigR.getS(i,j)/k/k/k;
            sigOutC[2*nF*2*nF + i + 2*nF*j] = sigI.getS(i,j)/k/k/k;
        }
    }

    return PyArray_Return(sigOut);

}
 
static char PyTrans_docs[] =
    "This is PyTrans, a package for solving the moment transport equations of inflationary cosmology\n";

// **************************************************************************************
static PyMethodDef PyTransLNC_methods[] = {{"H", (PyCFunction)MT_H,    METH_VARARGS, PyTrans_docs},{"Ep", (PyCFunction)MT_Ep,    METH_VARARGS, PyTrans_docs},{"Eta", (PyCFunction)MT_Eta,    METH_VARARGS, PyTrans_docs},{"nF", (PyCFunction)MT_fieldNumber,        METH_VARARGS, PyTrans_docs},{"nP", (PyCFunction)MT_paramNumber,        METH_VARARGS, PyTrans_docs},{"V", (PyCFunction)MT_V,            METH_VARARGS, PyTrans_docs},{"dV", (PyCFunction)MT_dV,                METH_VARARGS, PyTrans_docs},  {"ddV", (PyCFunction)MT_ddV,                METH_VARARGS, PyTrans_docs},  {"backEvolve", (PyCFunction)MT_backEvolve,        METH_VARARGS, PyTrans_docs},  {"sigEvolve", (PyCFunction)MT_sigEvolve,        METH_VARARGS, PyTrans_docs},  {"gamEvolve", (PyCFunction)MT_gamEvolve,        METH_VARARGS, PyTrans_docs},    {"alphaEvolve", (PyCFunction)MT_alphaEvolve,        METH_VARARGS, PyTrans_docs},  {"rhoEvolve", (PyCFunction)MT_rhoEvolve,        METH_VARARGS, PyTrans_docs}, {"rhoEvolve2", (PyCFunction)MT_rhoEvolve2,        METH_VARARGS, PyTrans_docs}, {"MPP2", (PyCFunction)MT_MPP2,        METH_VARARGS, PyTrans_docs}, {"MPPSigma", (PyCFunction)MT_MPPSigma,        METH_VARARGS, PyTrans_docs}, {"MPP3", (PyCFunction)MT_MPP3,        METH_VARARGS, PyTrans_docs},{"MPPAlpha", (PyCFunction)MT_MPPAlpha,        METH_VARARGS, PyTrans_docs},{"getSigma", (PyCFunction)MT_getSigma, METH_VARARGS, PyTrans_docs},{NULL, NULL, 0, NULL}};//FuncDef
// do not alter the comment at the end of the preceding line -- it is used by the preprocessor

#ifdef __cplusplus
extern "C" {
#endif

// **************************************************************************************    
static struct PyModuleDef PyTransModule = {PyModuleDef_HEAD_INIT, "PyTransLNC", PyTrans_docs, -1, PyTransLNC_methods}; //modDef
// do not alter the comment at the end of the preceding line -- it is used by the preprocessor

// **************************************************************************************
PyMODINIT_FUNC PyInit_PyTransLNC(void)    {    PyObject *m = PyModule_Create(&PyTransModule); import_array(); return m;} //initFunc

// do not alter the comment at the end of the preceding line -- it is used by the preprocessor

#ifdef __cplusplus
}
#endif
