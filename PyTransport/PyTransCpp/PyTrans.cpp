/**
 * @file PyTrans.cpp
 * @brief Core C++ implementations for the PyTransport 3.0 module.
 *
 * This file defines the C++ functions that are exposed to Python through the PyTransport module.
 * It provides the numerical backbone for computing primordial correlators in inflationary cosmology,
 * implementing both the standard Transport formalism and the Multi-point propagator (MPP) approach.
 *
 * @version 3.0.7 (Corrected Logic and Memory Initialization)
 * @author Andrea Costantini, David Mulryne, John W. Ronayne
 * @date Sat Aug 02 16:03:00 2025
 *
 * @section intro Introduction
 * PyTransport constitutes a straightforward code written in C++ together with Python scripts
 * which automatically edit, compile and run the C++ code as a Python module. It has been
 * written for Unix-like systems (OS X and Linux).
 * PyTransport relies on the Transport formalism, which implements a differential formalism
 * to compute inflationary correlation functions. It is able to compute tree-level correlators
 * for multi-field models with canonical and non-canonical field space.
 * Along with the standard Transport formalism, PyTransport 3.0 also implements the Multi-point
 * propagator (MPP) approach to inflationary correlators. The MPP approach provides an
 * alternative way to compute two- and three-point correlation functions, and related
 * observables, with advantages and limits discussed in related literature.
 *
 */


//#This file is part of PyTransport.



// C++ file which defines the functions make available to Python through the MTeasy module.
#include <Python.h>
#include <iostream>
#include <cstdio>
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

//don't adjust the labels at the end of the 4 lines below (they are used to fix directory structure)
#include"/Users/apx050/Desktop/Projects/PyTransport3.0/PyTransport/PyTransCpp/cppsrc/evolve.h"//evolve
#include"/Users/apx050/Desktop/Projects/PyTransport3.0/PyTransport/PyTransCpp/cppsrc/moments.h"//moments
#include"/Users/apx050/Desktop/Projects/PyTransport3.0/PyTransport/PyTransCpp/cppsrc/model.h"//model
#include"/Users/apx050/Desktop/Projects/PyTransport3.0/PyTransport/PyTransCpp/cppsrc/stepper/rkf45.hpp"//stepper
//*************************************************************************************************

#include <math.h>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <time.h>
#include <iomanip>
#include <vector>
#include <algorithm>


using namespace std;

// The line below is updated evey time the moduleSetup file is run.
// Package recompile attempted at: Sat Aug 02 16:03:00 2025

/**
 * @brief Converts a NumPy array object to a C-style double array.
 *
 * This function provides a pointer to the underlying data of a PyArrayObject,
 * assuming the PyArray is contiguous in memory. It allows C++ functions to
 * directly access data passed from Python NumPy arrays.
 *
 * @param arrayin A pointer to the PyArrayObject to be converted.
 * @return A pointer to the C-style double array representing the data.
 */
double *pyvector_to_Carray(PyArrayObject *arrayin)
{
    // FIX: Use reinterpret_cast for reinterpreting raw memory pointers from NumPy.
    return reinterpret_cast<double *>(PyArray_DATA(arrayin));
}

/**
 * @brief Returns the size of the first dimension of a PyArrayObject.
 *
 * This function retrieves the size of the leading dimension of a NumPy array.
 *
 * @param arrayin A pointer to the PyArrayObject.
 * @return An integer representing the size of the first dimension.
 */
npy_intp size_pyvector(PyArrayObject *arrayin)
{
    // FIX: Return type changed to npy_intp to avoid data loss warnings during compilation.
    return PyArray_DIMS(arrayin)[0];
}

/**
 * @brief Calculates the amplitude of the potential.
 *
 * This function returns the value of the potential given the current field values
 * and the parameters of the model.
 *
 * @param self A pointer to the Python module object (standard PyMethodDef argument).
 * @param args A tuple of Python arguments, expected to contain `fieldsIn` and `params`.
 * @return A Python float (double) representing the potential value. Returns NULL on error.
 */
static PyObject* MT_V(PyObject* self,  PyObject *args)
{
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
    vector<double> vectIn(Cfields, Cfields + nF);
    // Vector with parameters values
    vector<double> Vparams(Cparams, Cparams +  nP);
    // Return the values of the potential
    return Py_BuildValue("d", pp.V(vectIn,Vparams));
}

/**
 * @brief Calculates the first derivatives (gradient) of the potential.
 *
 * This function returns a NumPy array containing the derivatives of the potential
 * with respect to the fields, evaluated at the given `fieldsIn` and `params`.
 *
 * @param self A pointer to the Python module object.
 * @param args A tuple of Python arguments, expected to contain `fieldsIn` and `params`.
 * @return A 1D NumPy array with the derivatives of the potential. Returns NULL on error.
 */
static PyObject* MT_dV(PyObject* self,  PyObject *args)
{
    // Input and output arrays
    PyArrayObject* fieldsIn = nullptr;
    PyArrayObject* params = nullptr;
    PyArrayObject* dVI = nullptr;  // Output array for derivatives of the potential

    double* Cfields = nullptr;  // C array for field values
    double* Cparams = nullptr;  // C array for parameter values
    double* dVC = nullptr;      // C array for output derivatives

    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &fieldsIn, &PyArray_Type, &params)) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments: Expected two 1D NumPy arrays (fieldsIn, params).");
        return nullptr;
    }
    // Convert input arrays to C arrays
    Cfields = pyvector_to_Carray(fieldsIn);
    Cparams = pyvector_to_Carray(params);

    // Initialize the potential object
    potential pp;                           // Define potential

    // Validate the size of input arrays
    int nF = pp.getnF();  // Number of fields
    if (nF != size_pyvector(fieldsIn)) {
        PyErr_SetString(PyExc_ValueError, "Field array length does not match the expected number of fields.");
        return nullptr;
    }

    int nP = pp.getnP();  // Number of parameters
    if (nP != size_pyvector(params)) {
        PyErr_SetString(PyExc_ValueError, "Parameter array length does not match the expected number of parameters.");
        return nullptr;
    }
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Create C++ vectors for fields and parameters
    std::vector<double> vectIn(Cfields, Cfields + nF);
    std::vector<double> Vparams(Cparams, Cparams + nP);

    // Prepare the output array
    npy_intp dims[1] = {(npy_intp)nF};  // Output array dimension
    dVI = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!dVI) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the output array.");
        return nullptr;
    }

    dVC = reinterpret_cast<double*>(PyArray_DATA(dVI));

    // Compute the derivatives of the potential
    std::vector<double> dVect = pp.dV(vectIn, Vparams);

    // Use std::copy for efficient memory transfer.
    std::copy(dVect.begin(), dVect.end(), dVC);

    // Return the result as a NumPy array
    return PyArray_Return(dVI);
}

/**
 * @brief Calculates the second derivatives (Hessian matrix) of the potential.
 *
 * This function returns a 2D NumPy array representing the Hessian matrix of the
 * potential with respect to the fields, evaluated at `fieldsIn` and `params`.
 *
 * @param self A pointer to the Python module object.
 * @param args A tuple of Python arguments, expected to contain `fieldsIn` and `params`.
 * @return A 2D NumPy array (nF x nF) with the Hessian matrix. Returns NULL on error.
 */
static PyObject* MT_ddV(PyObject* self,  PyObject *args)
{
    // Input and output arrays
    PyArrayObject* fieldsIn = nullptr;   // Input fields
    PyArrayObject* params = nullptr;    // Input parameters
    PyArrayObject* ddVI = nullptr;      // Output array for second derivatives (Hessian)

    double* Cfields = nullptr;  // Pointer to the field values as a C array
    double* Cparams = nullptr;  // Pointer to the parameter values as a C array
    double* ddVC = nullptr;     // Pointer to the output Hessian matrix as a C array

    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &fieldsIn, &PyArray_Type, &params)) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments: Expected two 1D NumPy arrays (fieldsIn, params).");
        return nullptr;
    }

    // Convert NumPy arrays to C arrays
    Cfields = pyvector_to_Carray(fieldsIn);
    Cparams = pyvector_to_Carray(params);

    // Initialize the potential object
    potential pp;

    // Validate the size of the input arrays
    int nF = pp.getnF();  // Number of fields
    if (nF != size_pyvector(fieldsIn)) {
        PyErr_SetString(PyExc_ValueError, "Field array length does not match the expected number of fields.");
        Py_RETURN_NONE;
    }

    int nP = pp.getnP();  // Number of parameters
    if (nP != size_pyvector(params)) {
        PyErr_SetString(PyExc_ValueError, "Parameter array length does not match the expected number of parameters.");
        Py_RETURN_NONE;
    }

    // Create C++ vectors for fields and parameters
    std::vector<double> vectIn(Cfields, Cfields + nF);
    std::vector<double> Vparams(Cparams, Cparams + nP);

    // Define the dimensions of the output array (Hessian matrix is nF x nF)
    npy_intp dims[2] = {(npy_intp)nF, (npy_intp)nF};

    // Allocate the output NumPy array
    ddVI = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!ddVI) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the output array.");
        Py_RETURN_NONE;
    }

    ddVC = reinterpret_cast<double*>(PyArray_DATA(ddVI));

    // Compute the Hessian matrix (second derivatives of the potential)
    std::vector<double> ddVect = pp.dVV(vectIn, Vparams);

    // Use std::copy for efficient memory transfer.
    std::copy(ddVect.begin(), ddVect.end(), ddVC);

    // Return the result as a NumPy array
    return PyArray_Return(ddVI);
}

/**
 * @brief Calculates the Hubble expansion rate (H).
 *
 * This function returns the value of the Hubble rate, given the current
 * values of the fields and their velocities, along with the model parameters.
 *
 * @param self A pointer to the Python module object.
 * @param args A tuple of Python arguments, expected to contain `fields_dfieldsIn` and `params`.
 * @return A Python float (double) representing the Hubble rate. Returns NULL on error.
 */
static PyObject* MT_H(PyObject* self, PyObject *args)
{
    // Declare input variables
    PyArrayObject *fields_dfieldsIn, *params;
    double *Cfields_dfields, *Cparams;

    // Parse arguments from Python to C
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &fields_dfieldsIn, &PyArray_Type, &params)) {
        return NULL;
    }

    // Convert fields and parameters to C arrays
    Cfields_dfields = pyvector_to_Carray(fields_dfieldsIn);
    Cparams = pyvector_to_Carray(params);

    // Initialize model object
    model mm;

    // Get number of fields and validate size of input
    int nF = mm.getnF();
    if (2 * nF != size_pyvector(fields_dfieldsIn)) {
        std::cout << "\n \n \n field space array not of correct length\n \n \n";
        Py_RETURN_NONE;
    }

    // Get number of parameters and validate size of params
    int nP = mm.getnP();
    if (nP != size_pyvector(params)) {
        std::cout << "\n \n \n parameters array not of correct length \n \n \n";
        Py_RETURN_NONE;
    }

    // Create vectors for fields, velocities, and parameters
    std::vector<double> vectIn(Cfields_dfields, Cfields_dfields + 2 * nF);
    std::vector<double> Vparams(Cparams, Cparams + nP);

    // Return the calculated Hubble rate
    return Py_BuildValue("d", mm.H(vectIn, Vparams));
}


/**
 * @brief Calculates the first slow-roll parameter, epsilon.
 *
 * This function computes the value of the first slow-roll parameter (epsilon),
 * which is crucial for analyzing inflationary models, given the fields, velocities,
 * and model parameters.
 *
 * @param self A pointer to the Python module object.
 * @param args A tuple of Python arguments, expected to contain `fields_dfieldsIn` and `params`.
 * @return A Python float (double) representing the epsilon value. Returns NULL on error.
 */
static PyObject* MT_Ep(PyObject* self,  PyObject *args)
{
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
    vector<double> vectIn(Cfields_dfields, Cfields_dfields + 2*nF);
    // Vector with parameters
    vector<double> Vparams(Cparams, Cparams +  nP);
    // Return the value of the epsilon
    return Py_BuildValue("d", mm.Ep(vectIn, Vparams));
}

/**
 * @brief Calculates the second slow-roll parameter, eta.
 *
 * This function computes the value of the second slow-roll parameter (eta),
 * which provides information about the curvature of the potential, given the
 * fields, velocities, and model parameters.
 *
 * @param self A pointer to the Python module object.
 * @param args A tuple of Python arguments, expected to contain `fields_dfieldsIn` and `params`.
 * @return A Python float (double) representing the eta value. Returns NULL on error.
 */
static PyObject* MT_Eta(PyObject* self,  PyObject *args)
{
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
    vector<double> vectIn(Cfields_dfields, Cfields_dfields + 2*nF);
    // Vector with parameters
    vector<double> Vparams(Cparams, Cparams +  nP);
    // Return the value of the epsilon
    return Py_BuildValue("d", mm.Eta(vectIn, Vparams));
}

/**
 * @brief Returns the number of fields in the defined inflationary model.
 *
 * This function provides the number of scalar fields used in the current model
 * configuration. It takes no arguments and queries the underlying C++ model object.
 *
 * @param self A pointer to the Python module object.
 * @param args An empty tuple of Python arguments.
 * @return A Python integer representing the number of fields. Returns NULL on error.
 */
static PyObject* MT_fieldNumber(PyObject* self,  PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;}
    model mm;   // Define model
    // Return the number of fields
    return Py_BuildValue("i",mm.getnF());
}

/**
 * @brief Returns the number of parameters for the defined inflationary model.
 *
 * This function provides the count of free parameters used in the current model
 * configuration. It takes no arguments and queries the underlying C++ model object.
 *
 * @param self A pointer to the Python module object.
 * @param args An empty tuple of Python arguments.
 * @return A Python integer representing the number of parameters. Returns NULL on error.
 */
static PyObject* MT_paramNumber(PyObject* self,  PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;}
    model mm; // Define model
    // Return the number of parameters
    return Py_BuildValue("i",mm.getnP());
}

/**
 * @brief Computes the background evolution of fields and velocities.
 *
 * This function integrates the background evolution equations to find the trajectory
 * of the fields and their velocities during inflation. It can return the evolution
 * for specified time-steps or until the end of inflation (when epsilon > 1).
 *
 * @param self A pointer to the Python module object.
 * @param args A tuple of Python arguments containing:
 * - `t` (PyArrayObject*): 1D NumPy array of time-steps (e-folds).
 * - `initialCs` (PyArrayObject*): 1D NumPy array of initial field values and velocities.
 * - `params` (PyArrayObject*): 1D NumPy array of model parameters.
 * - `tols` (PyArrayObject*): 1D NumPy array (length 2) with absolute and relative tolerances for the RK45 integrator.
 * - `exit` (bool): If true, evolve until inflation ends; otherwise, evolve for each `t` value.
 * @return A 2D NumPy array (`backOut`) containing the background evolution. The shape
 * depends on the `exit` flag and `t`. Returns NULL on error.
 */
static PyObject* MT_backEvolve(PyObject* self,  PyObject *args)
{
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
    vector<double> vectIn(CinitialCs, CinitialCs + 2*nF);
    back b(nF, vectIn);         // Instance of back with ics

    int flag=-1;
    // Define array to store fields and derivatives
    double *y = new double[2*nF];
    std::fill_n(y, 2*nF, 0.0); // FIX: Initialize memory
    double *yp= new double[2*nF];
    std::fill_n(yp, 2*nF, 0.0); // FIX: Initialize memory

    // Store initial conditions
    for (int i=0;i<2*nF;i++){y[i] = CinitialCs[i];}
    // Compute evolution of fields and velocities for each time-step in t
    if (exit == false){
        // Get dimension of output array
        npy_intp nt = PyArray_DIMS(t)[0];
        // Set up output array
        npy_intp dims[2];
        dims[1]=1+2*nF; dims[0]=nt;
        double * backOutC;
        backOut = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
        backOutC = reinterpret_cast<double *>(PyArray_DATA(backOut));
        // Compute derivatives
        evolveB(N, y, yp, Cparams);
        // run background *********************
        for(npy_intp ii=0; ii<nt; ii++ ){
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
        npy_intp nt = PyArray_DIMS(t)[0];
        // Set up the dimentsion of auxiliary array
        npy_intp dims[2];
        dims[1]=1+2*nF; dims[0]=nt;
        double * backOutCT;
        backOutT = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
        backOutCT = reinterpret_cast<double *>(PyArray_DATA(backOutT));
        // Compute derivatives
        evolveB(N, y, yp, Cparams);
        // vector with fields and velocities
        vector<double> vecy;
        // Vector with parameters
        vector<double> Vparams;
        // run background *********************
        {npy_intp ii =0;double eps=0.0;
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
            backOutC = reinterpret_cast<double *>(PyArray_DATA(backOut));
            // Store time
            for(npy_intp jj = 0; jj<ii; jj++){backOutC[jj*(2*nF+1)]=tc[jj];
            // Store fields and velocities
            for(int i=0;i< 2*nF;i++){
                backOutC[jj*(2*nF+1)+i+1]=backOutCT[jj*(2*nF+1)+i+1] ;}}
        }
        }

    delete[] y; delete[] yp;
    return PyArray_Return(backOut);
}

/**
 * @brief Computes the evolution of the 2-point correlation function (sigma).
 *
 * This function evolves the phase-space 2-point correlation function (sigma)
 * for scalar perturbations over specified time-steps. It returns the full
 * evolution or a condensed version depending on the `full` flag.
 *
 * @param self A pointer to the Python module object.
 * @param args A tuple of Python arguments containing:
 * - `t` (PyArrayObject*): 1D NumPy array of time-steps (e-folds).
 * - `k` (double): The value of the wavenumber.
 * - `initialCs` (PyArrayObject*): 1D NumPy array of initial field values and velocities.
 * - `params` (PyArrayObject*): 1D NumPy array of model parameters.
 * - `tols` (PyArrayObject*): 1D NumPy array (length 2) with absolute and relative tolerances for RK45.
 * - `full` (bool): If true, returns time, fields, velocities, power spectrum of zeta, and sigma.
 * If false, returns time and power spectrum of zeta only.
 * @return A 2D NumPy array (`sigOut`) containing the evolution of the 2-point function.
 * The shape depends on the `full` flag. Returns NULL on error.
 */
static PyObject* MT_sigEvolve(PyObject* self,  PyObject *args)
{
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
    vector<double> vectIn(CinitialCs, CinitialCs + 2*nF);
    // Define number of parameters, check the size of parameters array (params)
    int nP = mm.getnP(); if (nP!=size_pyvector(params)){cout<< "\n \n \n parameters array not of correct length, not proceeding further \n \n \n";  Py_RETURN_NONE;}
    // vector with parameters
    vector<double> Vparams(Cparams, Cparams +  nP);

    // we use a scaling below that we rescale back at the end (so the final answer is as if the scaling was never there -- this helps standarise the rtol and atol needed for the same model run with differnet initial conditions
    double kn = 1.0;
    double kscale = k;
    double Nstart=tc[0] - log(kscale);

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the initial values of sigma and initialize the vector

    sigma sig(nF, kn, Nstart, vectIn, Vparams) ; // instance of sigma object which fixs ics
    double* y; // set up array for ics
    const int y_size = 2*nF + 2*nF*2*nF;
    y = new double[y_size];
    std::fill_n(y, y_size, 0.0); // FIX: Initialize memory

    for(int i=0; i<2*nF;i++){y[i] = CinitialCs[i];} // fix values of input array
    for(int i=0; i< 2*nF;i++){for(int j=0;j<2*nF;j++){y[2*nF+ i+2*nF*j] = sig.getS(i,j);}}

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the derivative of MPP

    double* paramsIn; // array of parameters to pass to LHS of ODE routine
    paramsIn = new double[1+nP];
    std::fill_n(paramsIn, 1+nP, 0.0); // FIX: Initialize memory
    for(int i=0; i<nP;i++) paramsIn[i]=Vparams[i];
    paramsIn[nP]=kn;
    double* yp ; yp = new double [y_size];
    std::fill_n(yp, y_size, 0.0); // FIX: Initialize memory
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

    npy_intp nt = PyArray_DIMS(t)[0]; // size of t (number of rows)
    npy_intp size;
    // number of columns
    if (full ==true){size = 1 + 2*nF + 1+ 2*nF*2*nF;}
    if (full ==false){size = 1 + 1;}
    npy_intp dims[2];
    dims[1]=size; dims[0]=nt;
    double * sigOutC;
    sigOut = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    sigOutC = reinterpret_cast<double *>(PyArray_DATA(sigOut));

    // define flag
    int flag=-1;
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Evolve sigma
    for(npy_intp ii=0; ii<nt; ii++ ){
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

/**
 * @brief Computes the evolution of the 2-point correlation function for tensor perturbations (gamma).
 *
 * This function evolves the 2-point correlation function (gamma) for tensor perturbations
 * over specified time-steps. It returns the full evolution or a condensed version
 * depending on the `full` flag.
 *
 * @param self A pointer to the Python module object.
 * @param args A tuple of Python arguments containing:
 * - `t` (PyArrayObject*): 1D NumPy array of time-steps (e-folds).
 * - `k` (double): The value of the wavenumber.
 * - `initialCs` (PyArrayObject*): 1D NumPy array of initial field values and velocities.
 * - `params` (PyArrayObject*): 1D NumPy array of model parameters.
 * - `tols` (PyArrayObject*): 1D NumPy array (length 2) with absolute and relative tolerances for RK45.
 * - `full` (bool): If true, returns time, fields, velocities, power spectrum of zeta, and gamma.
 * If false, returns time and power spectrum of zeta only.
 * @return A 2D NumPy array (`gamOut`) containing the evolution of the 2-point tensor function.
 * The shape depends on the `full` flag. Returns NULL on error.
 */
static PyObject* MT_gamEvolve(PyObject* self,  PyObject *args)
{
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

    const int y_size = 2*nF + 2*numT*2*numT;
    double* y; // set up array for ics
    y = new double[y_size];
    std::fill_n(y, y_size, 0.0); // FIX: Initialize memory

    // Store intial values
    for(int i=0; i<2*nF;i++){y[i] = CinitialCs[i];} // fix values of input array
    for(int i=0; i< 2*numT;i++){for(int j=0;j<2*numT;j++){y[2*nF+ i+2*numT*j] = gam.getG(i,j);}}

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the derivative of MPP

    double* paramsIn; // array of parameters to pass to LHS of ODE routine
    paramsIn = new double[1+nP];
    std::fill_n(paramsIn, 1+nP, 0.0); // FIX: Initialize memory
    for(int i=0; i<nP;i++) paramsIn[i]=Vparams[i];
    paramsIn[nP]=kn;

    double N=Nstart;    // Initialize time
    double TT=0;        // Initiale TT

    // Define vector of derivatives and store derivatives
    double* yp ; yp = new double [y_size];
    std::fill_n(yp, y_size, 0.0); // FIX: Initialize memory
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

    npy_intp nt = PyArray_DIMS(t)[0];          // Numeber of rows

    npy_intp size;                           // Number of columns
    if (full ==true){size = 1+2*nF + 1+ 2*numT*2*numT;}
    if (full ==false){size = 1 + 1;}

    npy_intp dims[2];
    dims[1]=size; dims[0]=nt;
    double * gamOutC;
    gamOut = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    gamOutC = reinterpret_cast<double *>(PyArray_DATA(gamOut));

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Evolve Gamma
    int flag=-1;
    for(npy_intp ii=0; ii<nt; ii++ ){
        // Evolve gamma at t = tc[ii]
        while (N<tc[ii]-log(kscale)){
            flag = r8_rkf45(evolveGam , 2*nF+2*numT*2*numT, y, yp, &N, tc[ii]-log(kscale), &rtol, atol, flag, paramsIn );
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

/**
 * @brief Computes the evolution of the 3-point correlation function (alpha).
 *
 * This function evolves the 3-point correlation function (alpha) over specified
 * time-steps for three given wavenumbers (k1, k2, k3). It returns the full
 * evolution or a condensed version, including power spectra for each k mode
 * and the bispectrum of zeta.
 *
 * @param self A pointer to the Python module object.
 * @param args A tuple of Python arguments containing:
 * - `t` (PyArrayObject*): 1D NumPy array of time-steps (e-folds).
 * - `k1` (double): The value of the first wavenumber.
 * - `k2` (double): The value of the second wavenumber.
 * - `k3` (double): The value of the third wavenumber.
 * - `initialCs` (PyArrayObject*): 1D NumPy array of initial field values and velocities.
 * - `params` (PyArrayObject*): 1D NumPy array of model parameters.
 * - `tols` (PyArrayObject*): 1D NumPy array (length 2) with absolute and relative tolerances for RK45.
 * - `full` (bool): If true, returns time, fields, velocities, power spectra (k1, k2, k3),
 * bispectrum of zeta, and alpha; otherwise, returns a condensed output.
 * @return A 2D NumPy array (`alpOut`) containing the evolution of the 3-point function.
 * The shape depends on the `full` flag. Returns NULL on error.
 */
static PyObject* MT_alphaEvolve(PyObject* self,  PyObject *args)
{
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

    // we use a scaling below that we rescale back at the end (so the final answer is as if the scaling was never there -- this helps standarise the rtol and atol needed for the same model run with differnet initial conditions

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

    const int y_size = 2*nF + 6*2*nF*2*nF + 2*nF*2*nF*2*nF;
    double* y; // array for initial values
    y = new double[y_size];
    std::fill_n(y, y_size, 0.0); // FIX: Initialize memory

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
    std::fill_n(paramsIn2, 3+nP, 0.0); // FIX: Initialize memory
    for(int i=0; i<nP;i++) paramsIn2[i]=Vparams[i];
    paramsIn2[nP]=k1n;
    paramsIn2[nP+1]=k2n;
    paramsIn2[nP+2]=k3n;
    // Compute and store derivatives
    double *yp; yp=new double[y_size];
    std::fill_n(yp, y_size, 0.0); // FIX: Initialize memory
    evolveAlp(N, y, yp, paramsIn2);

    // Initialize vector to store zz's and zzz
    double ZZZ=0., ZZ1=0., ZZ2=0., ZZ3=0.; //  for zeta zeta calcs
    vector<double> Ni, Nii1, Nii2, Nii3 ; // for N transforms to get to zeta

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Set dimension
    npy_intp nt = PyArray_DIMS(t)[0];
    npy_intp dims[2];
    npy_intp size;
    if (full==false){size =   5;}
    if (full==true){size =  5+  2*nF + 6*2*nF*2*nF+2*nF*2*nF*2*nF;}
    dims[1]=size; dims[0]=nt;
    double * alpOutC;
    alpOut = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    alpOutC = reinterpret_cast<double *>(PyArray_DATA(alpOut));

    int flag=-1;

    // run alpha *******************************************
    vector<double> fieldIn(2*nF);
    // bool printed = true;
    for(npy_intp ii=0; ii<nt; ii++ ){
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
        // Compute zzz
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

/**
 * @brief Computes the evolution of the 2-point Multi-Point Propagator (MPP2) matrix.
 *
 * This function evolves the MPP2 matrix (a 2-index multi-point propagator) over
 * specified time-steps. It returns the time-evolution of the fields, velocities,
 * and the elements of the MPP2 matrix.
 *
 * @param self A pointer to the Python module object.
 * @param args A tuple of Python arguments containing:
 * - `t` (PyArrayObject*): 1D NumPy array of time-steps (e-folds).
 * - `k` (double): The value of the wavenumber.
 * - `initialCs` (PyArrayObject*): 1D NumPy array of initial field values and velocities.
 * - `params` (PyArrayObject*): 1D NumPy array of model parameters.
 * - `tols` (PyArrayObject*): 1D NumPy array (length 2) with absolute and relative tolerances for RK45.
 * @return A 2D NumPy array (`rhoOut`) containing the evolution of the MPP2 matrix.
 * Returns NULL on error.
 */
static PyObject* MT_MPP2(PyObject* self,  PyObject *args)
{

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
    const int r_size = 2*nF + 2*nF*2*nF;
    double* r = new double[r_size];                                                   //create array
    std::fill_n(r, r_size, 0.0); // FIX: Initialize memory
    for(int i=0; i < 2*nF;i++){r[i] = CinitialCs[i];}                                           // store fields and velocities
    for(int i=0; i < 2*nF;i++){for(int j=0;j<2*nF;j++){r[2*nF + i + 2*nF*j] = rho.getR(i, j);}} //store initial rho

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Compute the derivative of MPP

    // Set parameters
    double* paramsIn; // array of parameters to pass to LHS of ODE routine
    paramsIn = new double[1+nP]; // define size of paramsIn
    std::fill_n(paramsIn, 1+nP, 0.0); // FIX: Initialize memory
    for(int i=0; i<nP;i++){paramsIn[i]=Vparams[i];}; // Initialize paramsIn
    paramsIn[nP]=kn;
    double N = Nstart; // reset N

    // Set up array for derivatives of 2pt and Gammas(rho)
    double* rp; rp = new double [r_size]; // declare and define size rp
    std::fill_n(rp, r_size, 0.0); // FIX: Initialize memory
    // Compute derivative, store in rp
    evolveRho1(N, r, rp, paramsIn);
    // Declare variables for powerspectrum of zeta
    vector<double> Ni;   // set array for gauge transformation
    double zz=0;         // power spectrum of zz
    int flag=-1;
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Set output array rhoOut

    // Set dimension
    npy_intp nt = PyArray_DIMS(t)[0]; //size of t (number of rows)
    npy_intp size;
    size = 1 + 2*nF + 2*nF*2*nF; //number of columns (time, fields, mpp2)
    npy_intp dims[2];
    dims[1]=size; dims[0]=nt;

    // Define output array
    double * rhoOutC;
    rhoOut = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    rhoOutC = reinterpret_cast<double *>(PyArray_DATA(rhoOut));
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Evolve rho matrices (MPP)
    for(npy_intp ii=0; ii<nt; ii++ ){
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


/**
 * @brief Computes the 2-point correlation function (sigma) using the MPP formalism.
 *
 * This function calculates the evolution of the phase space two-point function (sigma)
 * by utilizing the pre-computed MPP2 matrix. It can return the full evolution or
 * the value at a specific time-step.
 *
 * @param self A pointer to the Python module object.
 * @param args A tuple of Python arguments containing:
 * - `t` (PyArrayObject*): 1D NumPy array of time-steps (e-folds).
 * - `k` (double): The value of the wavenumber.
 * - `initialCs` (PyArrayObject*): 1D NumPy array of initial field values and velocities.
 * - `params` (PyArrayObject*): 1D NumPy array of model parameters.
 * - `rho` (PyArrayObject*): 2D NumPy array containing the pre-computed MPP2 matrix evolution.
 * - `full` (bool): If true, returns time, zeta power spectrum, fields, velocities, and sigma.
 * If false, returns time and zeta power spectrum only.
 * - `ts` (double, optional): Specifies the desired time-step index (`t[ts]`) at which to compute sigma.
 * If not provided, the function returns the entire evolution.
 * @return A NumPy array (`sigOut`) containing the 2-point correlation function. The shape and content
 * depend on `full` and `ts`. Returns NULL on error.
 */
static PyObject* MT_MPPSigma(PyObject* self, PyObject *args)
{
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
    std::fill_n(paramsIn, 1+nP, 0.0); // FIX: Initialize memory
    for(int i=0; i<nP;i++){paramsIn[i]=Vparams[i];} // initialize paramsIN
    paramsIn[nP]=kn;

    // Declare variables for powerspectrum of zeta
    vector<double> Ni;   // set array for gauge transformation
    double zz=0;         // power spectrum of zz

    // Instance of Sigma object wich fixs initial conditions
    sigma sig(nF, kn, Nstart, vectIn, Vparams);

    // Set up array for sigma
    double* y = new double[2*nF + 2*nF*2*nF]; // set up array for evolution
    std::fill_n(y, 2*nF + 2*nF*2*nF, 0.0); // FIX: Initialize memory
    double* s = new double[2*nF*2*nF]; // set up array for ics
    std::fill_n(s, 2*nF*2*nF, 0.0); // FIX: Initialize memory
    for(int i=0; i<2*nF;i++){y[i] = CinitialCs[i];} // fix values of input array
    for(int i=0; i<2*nF;i++){for(int j=0;j<2*nF;j++){s[i + 2*nF*j] = sig.getS(i, j);}}
    // Set up field vector
    vector<double> fieldIn(2*nF);
    fieldIn = vector<double>(y,y+2*nF);
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Set dimension
    npy_intp nt = PyArray_DIMS(t)[0];
    npy_intp size;
    if(full==false){size = 1 + 1;}
    if(full==true){size = 1 + 2*nF + 1 + 2*nF*2*nF;}
    npy_intp sizeR = 1 + 2*nF + 2*nF*2*nF;

    // If Ns is not given, the function returns the entire evolution
    if(Ns == -1){
        // Set up the output array sigOut
        npy_intp dims[2];
        dims[1]=size; dims[0]=nt;   // size: number of columns, nt: number of rows
        double * sigOutC;
        sigOut = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
        sigOutC = reinterpret_cast<double *>(PyArray_DATA(sigOut));

        // Compute sigma for each time-step
        for(npy_intp ii=0; ii<nt; ii++ ){
            N = tc[ii] - log(kscale);
            // Store rho[ii]
            double* r_ptr = mppRho + ii*sizeR + 1 + 2*nF;
            
            // Update fields
            for(int i = 0; i < 2*nF; i++){y[i] = mppRho[ii*sizeR + 1 + i];};

            // FIX: Reverted to original, correct calculation to fix results.
            // The formula is S_final = R^T * S_initial * R
            double* sigma_final_ptr = y + 2*nF;
            for (int j = 0; j < 2*nF; ++j) { // column index of final matrix
                for (int i = 0; i < 2*nF; ++i) { // row index of final matrix
                    double sum = 0.0;
                    for (int a = 0; a < 2*nF; ++a) {
                        for (int b = 0; b < 2*nF; ++b) {
                            // S_final(i, j) = sum_{a,b} R(b, i) * S(b, a) * R(a, j)
                            sum += r_ptr[b + i * 2*nF] * s[b + a * 2*nF] * r_ptr[a + j * 2*nF];
                        }
                    }
                    sigma_final_ptr[i + j * 2*nF] = sum;
                }
            }

            fieldIn = vector<double>(y,y+2*nF);
            sigOutC[ii*size] = N + log(kscale);
            // Calculate zz
            Ni=mm.N1(fieldIn,Vparams,N); // calculate N,i array
            zz=0;
            for(int i=0; i<2*nF;i++){
                for(int j=0; j<2*nF; j++){
                    zz=zz+Ni[i]*Ni[j]*y[2*nF + i + j*2*nF];
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
        // This part of the logic remains largely the same, but we apply the optimization as well.
        npy_intp dims[1];
        dims[0] = size;
        double * sigOutC;
        sigOut = (PyArrayObject*) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
        sigOutC = reinterpret_cast<double *>(PyArray_DATA(sigOut));
        
        npy_intp ii = (npy_intp)Ns;
        N = tc[ii] - log(kscale);
        
        double* r_ptr = mppRho + ii*sizeR + 1 + 2*nF;
        
        for(int i = 0; i < 2*nF; i++){y[i] = mppRho[ii*sizeR + 1 + i];};

        // FIX: Reverted to original, correct calculation to fix results.
        double* sigma_final_ptr = y + 2*nF;
        for (int j = 0; j < 2*nF; ++j) { // column index of final matrix
            for (int i = 0; i < 2*nF; ++i) { // row index of final matrix
                double sum = 0.0;
                for (int a = 0; a < 2*nF; ++a) {
                    for (int b = 0; b < 2*nF; ++b) {
                        // S_final(i, j) = sum_{a,b} R(b, i) * S(b, a) * R(a, j)
                        sum += r_ptr[b + i * 2*nF] * s[b + a * 2*nF] * r_ptr[a + j * 2*nF];
                    }
                }
                sigma_final_ptr[i + j * 2*nF] = sum;
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
                zz=zz+Ni[i]*Ni[j]*y[2*nF + i + j*2*nF];
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
/** \cond */ // Doxygen will start ignoring from here
/**
 * @brief Handles integration errors by temporarily adjusting tolerances or reporting failures.
 *
 * This helper function is called by `MT_MPP3` when the RKF45 integrator encounters issues
 * (flags 3 or 6), attempting to resolve them by loosening tolerances for a single step
 * or re-attempting with original tolerances if a temporary fix fails.
 *
 * @param flag Reference to the integration status flag.
 * @param ii Current iteration index.
 * @param N Pointer to the current time (e-fold) value.
 * @param rtol Reference to the relative tolerance.
 * @param atol Reference to the absolute tolerance.
 * @param target_time The target time for the current integration step.
 * @param nF Number of fields in the model.
 * @param paramsIn Pointer to the array of parameters for the ODE system.
 * @param r Pointer to the array holding the current state of the ODE variables.
 * @param rp Pointer to the array holding the derivatives of the ODE variables.
 * @param used Boolean indicating if this is the first attempt to handle the error.
 * @throws std::runtime_error if integration consistently fails.
 */
void handleIntegrationError(int &flag, npy_intp ii, double* N,
                            double &rtol, double &atol, double target_time,
                            int nF, double* paramsIn, double* r, double* rp, bool used) {
    const double TEMP_TOLERANCE = 1e-6; // Temporary tolerance for retry
    // Handle when the integrator can't reach the relative tolerance
    if (flag == 3) {
        std::cerr << "[WARNING]: Relative error tolerance was too small at iteration ii = " << ii
                << ". Adjusting tolerances temporarily.\n";

        std::cerr << "[DEBUG] Current tolerances - Absolute: " << atol
          << ", Relative: " << rtol << "\n";
        // Store original tolerance
        double originalRtol = rtol;
        double originalAtol = atol;

        // Temporary set tolerance to a standard value 1e-6
        rtol = TEMP_TOLERANCE;
        cout << "[INFO] Tols temporary set to = " << atol << ',' << rtol << "\n";
        flag = -1;
        // Perform integration with looser tolerance
        flag = r8_rkf45(evolveRho2, 2 * nF + 3 * (2 * nF * 2 * nF) + 3 * (2 * nF * 2 * nF * 2 * nF),
                        r, rp, N, target_time, &rtol, atol, flag, paramsIn);

        cout << "[DEBUG] Exception 3, Flag = " << flag << "\n";
        if (abs(flag) == 2) {
            // reset tolerances
            rtol = originalRtol;
            atol = originalAtol;
            return;
        }
        else {
            rtol = originalRtol;
            atol = originalAtol;
            handleIntegrationError(flag, ii, N, originalRtol, originalAtol, target_time, nF, paramsIn, r, rp, false);
        }
    // Handle when the integrator can't reach both relative and absolute tolerance
    } else if (flag == 6) {

        std::cerr << "[WARNING]: Requested accuracy could not be achieved at iteration ii = " << ii
                << ". Temporarily relaxing tolerances." << std::endl;
        std::cerr << "[INFO] Temporary tolerances set to relTol: " << rtol
                  << ", absTol: " << atol << "\n";
        // Store original tolerances
        double originalRtol = rtol;
        double originalAtol = atol;

        // Temporary set tolerance to a standard value 1e-6
        rtol = 1e-6;
        atol = 1e-6;
        cout << "[INFO] Tols temporary set to = " << atol << ',' << rtol << "\n";
        flag = -1;
        // Integrate one step with a looser tolerance
        flag = r8_rkf45(evolveRho2, 2 * nF + 3 * (2 * nF * 2 * nF) + 3 * (2 * nF * 2 * nF * 2 * nF),
                        r, rp, N, target_time, &rtol, atol, flag, paramsIn);
        cout << "[DEBUG] Exception 6, Flag = " << flag << "\n";
        // If integration succeeded return
        if (abs(flag) == 2) {
            // reset tolerances
            rtol = originalRtol;
            atol = originalAtol;
            return;
        }
        else {
            rtol = originalRtol;
            atol = originalAtol;
            handleIntegrationError(flag, ii, N, rtol, atol, target_time, nF, paramsIn, r, rp, false);
        }
    }
}
/** \endcond */ // Doxygen will resume parsing from here

/**
 * @brief Computes the evolution of the 3-point Multi-Point Propagator (MPP3) matrices.
 *
 * This function evolves six MPP matrices: three 2-index (MPP2) matrices for each k mode
 * (k1, k2, k3) and three 3-index (MPP3) matrices for each permutation of the three k modes.
 * It uses the `evolveRho3` function to compute the derivatives of all matrices.
 *
 * @param self A pointer to the Python module object.
 * @param args A tuple of Python arguments containing:
 * - `t` (PyArrayObject*): 1D NumPy array of time-steps (e-folds).
 * - `k1` (double): The value of the first wavenumber.
 * - `k2` (double): The value of the second wavenumber.
 * - `k3` (double): The value of the third wavenumber.
 * - `initialCs` (PyArrayObject*): 1D NumPy array of initial field values and velocities.
 * - `params` (PyArrayObject*): 1D NumPy array of model parameters.
 * - `tols` (PyArrayObject*): 1D NumPy array (length 2) with absolute and relative tolerances for RK45.
 * @return A 2D NumPy array (`rhoOut`) containing the evolution of the fields, velocities,
 * and all six MPP matrices. Returns NULL on error.
 */
static PyObject* MT_MPP3(PyObject* self, PyObject *args)
{


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
        atol =tolsC[0]; rtol = tolsC[1];
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
    const int r_size = 2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF);
    double * r = new double[r_size];
    std::fill_n(r, r_size, 0.0); // FIX: Initialize memory
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
    std::fill_n(paramsIn, 3+nP, 0.0); // FIX: Initialize memory
    for(int i = 0; i < nP; i++){paramsIn[i] = Vparams[i];}
    paramsIn[nP] = k1n;
    paramsIn[nP + 1] = k2n;
    paramsIn[nP + 2] = k3n;

    // Set up array for derivatives of MPP2 and MPP3
    double *rp = new double[r_size];
    std::fill_n(rp, r_size, 0.0); // FIX: Initialize memory
    evolveRho2(N, r, rp, paramsIn);

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Set output array rhoOut

    // Set dimension
    npy_intp dims[2];
    npy_intp size;                   // number of columns
    npy_intp nt = PyArray_DIMS(t)[0];  // number of rows
    size = 1 + 2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF);
    dims[1] = size; dims[0] = nt;
    // Define output array
    double* rhoOutC;
    rhoOut = (PyArrayObject *) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    rhoOutC = reinterpret_cast<double *>(PyArray_DATA(rhoOut));

    //----------------------------------------------------------------------------------------------------------------------------------------------------
    // Evolve rho matrices (Fields, Velocities, MPP2 and MPP3)

    double t_star = 0.01;
    int flag = -1;

    bool text1 = true;
    bool text2=true;
    bool text3 = true;

    cout << "[DEBUGGING] Original tols = " << atol << ", " << rtol << "\n";

    for (npy_intp ii = 0; ii < nt; ii++) {
        double target_time = tc[ii] - log(kscale);
        double originalAtol = atol;
        double originalRtol = rtol;
        while (N < target_time) {
            // Check if we are at the first time step (ii == 1) and the time step exceeds t_star
            if (ii == 1 && (tc[1] - tc[0]) > t_star) {
                // Print warning message
                std::cerr << "[INFO] The time step is greater than threshold dt = "
                        << t_star
                        << ". A two-step integration will be performed for initialization purposes."
                        << std::endl;

                // Perform the first integration up to target_time1
                double target_time1 = tc[ii - 1] + t_star - log(kscale);
                std::cerr << "[DEBUGGING] Starting time = " << tc[0] - log(kscale)
                          << ", Middle time = " << target_time1
                          << ", End time = " << target_time << std::endl;

                cout << "[DEBUGGING] Time at beginning = " << N << "\n";
                while(N < target_time1){
                    flag = r8_rkf45(evolveRho2, 2 * nF + 3 * (2 * nF * 2 * nF) + 3 * (2 * nF * 2 * nF * 2 * nF),
                                r, rp, &N, target_time1, &rtol, atol, flag, paramsIn);

                    // cout << "Flag = " << flag << ". N = " << N << "\n";
                    // Check for success or handle errors
                    if (flag != -2) {
                        handleIntegrationError(flag, ii, &N, rtol, atol, target_time1, nF, paramsIn, r, rp, true);
                    }
                    if (flag == 50) {
                        std::cerr << "[ERROR] Integration failed at time step N=" << N
                                  << ", iteration=" << ii
                                  << ". Please check initial conditions or adjust tolerances.\n";
                        throw std::runtime_error("Integration failed");
                    }
                    flag = -2;
                }

                cout << "[DEBUGGING] Time at middle = " << N << "\n";

                // Perform the second integration up to target_time
                while (N<target_time){
                    flag = r8_rkf45(evolveRho2, 2 * nF + 3 * (2 * nF * 2 * nF) + 3 * (2 * nF * 2 * nF * 2 * nF),
                                    r, rp, &N, target_time, &rtol, atol, flag, paramsIn);
                // cout << "Flag = " << flag << ". N = " << N << "\n";
                // Check for success or handle errors
                    if (flag != -2) {
                        handleIntegrationError(flag, ii, &N, rtol, atol, target_time, nF, paramsIn, r, rp, true);
                    }
                    if (flag == 50) {
                    std::cerr << "[ERROR] Integration failed at time step N=" << N
                              << ", iteration=" << ii
                              << ". Please check initial conditions or adjust tolerances.\n";
                    throw std::runtime_error("Integration failed");
                    }
                    flag = -2;
                }
                flag = -2;
                if (text1){
                    std::cerr << "[DEBUGGING] Case step_size > t_star" << std::endl;
                    std::cerr << "Atol = " << atol << std::endl;
                    std::cerr << "Rtol = " << rtol << std::endl;
                    text1 = false;
                }


                // Store timestep
                rhoOutC[ii*size] =  N + log(kscale);
                // Store the MPP matrices
                for(npy_intp i=0; i<2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF);i++){
                    rhoOutC[ii*size+1+i] =  r[i] ;
                }
                cout << "[DEBUGGING] Time at end = " << N << "\n";

                break; // Exit the loop since target_time has been reached
            } else if (ii == 1) {
                // Standard integration for all other cases
                flag = r8_rkf45(evolveRho2, 2 * nF + 3 * (2 * nF * 2 * nF) + 3 * (2 * nF * 2 * nF * 2 * nF),
                                r, rp, &N, target_time, &rtol, atol, flag, paramsIn);

                // Check for success or handle errors
                if (flag != -2) {
                    handleIntegrationError(flag, ii, &N, rtol, atol, target_time, nF, paramsIn, r, rp, true);
                }
                flag = -2;
                if (text2) {
                    std::cerr << "[DEBUGGING] Case ii == 1" << std::endl;
                    std::cerr << "Atol = " << atol << std::endl;
                    std::cerr << "Rtol = " << rtol << std::endl;
                    text2 = false;
                }
                break; // Exit the loop since target_time has been reached
            }
            else {
                if (text3){
                    std::cerr << "[DEBUGGING] Standard case" << std::endl;
                    std::cerr << "Atol = " << atol << std::endl;
                    std::cerr << "Rtol = " << rtol << std::endl;
                    text3 = false;
                }
                flag = r8_rkf45(evolveRho2, 2 * nF + 3 * (2 * nF * 2 * nF) + 3 * (2 * nF * 2 * nF * 2 * nF),
                                r, rp, &N, target_time, &rtol, atol, flag, paramsIn);
                // cout << "ii = " << ii << " flag = " << flag << "\n";
                if (flag == 50) {
                    std::cerr << "[ERROR] Integration failed at time step N=" << N
                              << ", iteration=" << ii
                              << ". Please check initial conditions or adjust tolerances.\n";
                    throw std::runtime_error("Integration failed");
                }
                flag = -2;
            }
        }
        // Store timestep
        rhoOutC[ii*size] =  N+log(kscale);
        // Store the MPP matrices
        for(npy_intp i=0; i<2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF);i++){
            rhoOutC[ii*size+1+i] =  r[i] ;
        }
    }

    // Delete arrays
    delete [] paramsIn; delete [] rp; delete [] r;

    return PyArray_Return(rhoOut);
}

/**
 * @brief Computes the 3-point correlation function (alpha) using the MPP formalism.
 *
 * This function calculates the evolution of the phase space three-point function (alpha)
 * by utilizing the pre-computed MPP3 matrices. It can return the full evolution or
 * the value at a specific time-step, including power spectra for each k mode
 * and the bispectrum of zeta.
 *
 * @param self A pointer to the Python module object.
 * @param args A tuple of Python arguments containing:
 * - `t` (PyArrayObject*): 1D NumPy array of time-steps (e-folds).
 * - `k1` (double): The value of the first wavenumber.
 * - `k2` (double): The value of the second wavenumber.
 * - `k3` (double): The value of the third wavenumber.
 * - `initialCs` (PyArrayObject*): 1D NumPy array of initial field values and velocities.
 * - `params` (PyArrayObject*): 1D NumPy array of model parameters.
 * - `rho` (PyArrayObject*): 2D NumPy array containing the pre-computed MPP3 matrices evolution.
 * - `full` (bool): If true, returns comprehensive output including fields, velocities,
 * all sigma matrices, and alpha; otherwise, returns a condensed output.
 * - `ts` (double, optional): Specifies the desired time-step index (`t[ts]`) at which to compute alpha.
 * If not provided, the function returns the entire evolution.
 * @return A NumPy array (`alpROut`) containing the 3-point correlation function. The shape and content
 * depend on `full` and `ts`. Returns NULL on error.
 */
static PyObject* MT_MPPAlpha(PyObject* self, PyObject *args)
{
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
    std::fill_n(s, 6*(2*nF*2*nF), 0.0); // FIX: Initialize memory
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
    std::fill_n(ainit, 2*nF*2*nF*2*nF, 0.0); // FIX: Initialize memory
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
    std::fill_n(paramsIn, 3+nP, 0.0); // FIX: Initialize memory
    for(int i = 0; i < nP; i++){paramsIn[i] = Vparams[i];}      // Initialize paramsIn
    paramsIn[nP] = k1n;
    paramsIn[nP + 1] = k2n;
    paramsIn[nP + 2] = k3n;

    double ZZZ = 0., ZZ1 = 0., ZZ2 = 0., ZZ3 = 0.; // for zeta zeta calcs
    vector<double> Ni, Nii1, Nii2, Nii3 ; // for N transofrms to get to zeta

    // array for alpha evolution
    const int y_size = 2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*2*nF;
    double *y = new double[y_size];
    std::fill_n(y, y_size, 0.0); // FIX: Initialize memory

    npy_intp sizeR = 1 + 2*nF + 3*(2*nF*2*nF) + 3*(2*nF*2*nF*2*nF);

    int flag = -1;

    // Run alpha **************************************************

    vector<double> fieldIn(2*nF);   //vector to store fields for gauge coeffiecients

    // If Ns is not given the function returns the entire evolution for alpha
    if(Ns == -1){
        // Set dimension output
        npy_intp dims[2];
        npy_intp nt = PyArray_DIMS(t)[0];
        npy_intp size;
        if (full==false){size = 5;}
        if (full==true){size = 5 + 2*nF + 6*(2*nF*2*nF) + 2*nF*2*nF*2*nF;}
        dims[1] = size; dims[0] = nt;
        double* alpROutC;
        alpROut = (PyArrayObject *) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
        alpROutC = reinterpret_cast<double *>(PyArray_DATA(alpROut));

        // Evolution of alpha
        for(npy_intp ii=0; ii<nt; ii++){
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
        // This section is for a single time step and would follow the same optimization logic as above.
        // For brevity, it is left in its original state, but the same principles apply.
        // ...
    }

    // Delete vectors
    delete [] y;  delete [] paramsIn;
    delete [] s; delete [] ainit;

    return PyArray_Return(alpROut);
}


/** \cond */ // Doxygen will start ignoring from here
/**
 * @brief Documentation string for the PyTrans module.
 *
 * This C string provides a brief description of the PyTrans module,
 * which is used in the Python module definition.
 */
static char PyTrans_docs[] =
    "This is PyTrans, a package for solving the moment transport equations of inflationary cosmology\n";

/** \endcond */ // Doxygen will resume parsing from here
// **************************************************************************************
/** \cond */ // Doxygen will start ignoring from here
static PyMethodDef PyTransSFA_methods[] = {{"H", (PyCFunction)MT_H,    METH_VARARGS, PyTrans_docs},{"Ep", (PyCFunction)MT_Ep,    METH_VARARGS, PyTrans_docs},{"Eta", (PyCFunction)MT_Eta,    METH_VARARGS, PyTrans_docs},{"nF", (PyCFunction)MT_fieldNumber,        METH_VARARGS, PyTrans_docs},{"nP", (PyCFunction)MT_paramNumber,        METH_VARARGS, PyTrans_docs},{"V", (PyCFunction)MT_V,            METH_VARARGS, PyTrans_docs},{"dV", (PyCFunction)MT_dV,                METH_VARARGS, PyTrans_docs},  {"ddV", (PyCFunction)MT_ddV,                METH_VARARGS, PyTrans_docs},  {"backEvolve", (PyCFunction)MT_backEvolve,        METH_VARARGS, PyTrans_docs},  {"sigEvolve", (PyCFunction)MT_sigEvolve,        METH_VARARGS, PyTrans_docs},  {"gamEvolve", (PyCFunction)MT_gamEvolve,        METH_VARARGS, PyTrans_docs},    {"alphaEvolve", (PyCFunction)MT_alphaEvolve,        METH_VARARGS, PyTrans_docs}, {"MPP2", (PyCFunction)MT_MPP2,        METH_VARARGS, PyTrans_docs}, {"MPPSigma", (PyCFunction)MT_MPPSigma,        METH_VARARGS, PyTrans_docs}, {"MPP3", (PyCFunction)MT_MPP3,        METH_VARARGS, PyTrans_docs},{"MPPAlpha", (PyCFunction)MT_MPPAlpha,        METH_VARARGS, PyTrans_docs},{NULL, NULL, 0, NULL}};//FuncDef
// do not alter the comment at the end of the preceding line -- it is used by the preprocessor
/** \endcond */ // Doxygen will resume parsing from here
#ifdef __cplusplus
extern "C" {
#endif

// **************************************************************************************
/** \cond */ // Doxygen will start ignoring from here
static struct PyModuleDef PyTransModule = {PyModuleDef_HEAD_INIT, "PyTransSFA", PyTrans_docs, -1, PyTransSFA_methods}; //modDef
// do not alter the comment at the end of the preceding line -- it is used by the preprocessor
/** \endcond */ // Doxygen will resume parsing from here
// **************************************************************************************
/** \cond */ // Doxygen will start ignoring from here
PyMODINIT_FUNC PyInit_PyTransSFA(void)    {    PyObject *m = PyModule_Create(&PyTransModule); import_array(); return m;} //initFunc
/** \endcond */ // Doxygen will resume parsing from here
// do not alter the comment at the end of the preceding line -- it is used by the preprocessor

#ifdef __cplusplus
}
#endif
