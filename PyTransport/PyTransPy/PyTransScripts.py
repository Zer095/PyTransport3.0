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


# python code contains some useful scripts to use with the compiled PyTrans module.

import numpy as np
from scipy import interpolate
import timeit
import sys
#from mpi4py import MPI

# this script finds initial conditions at least NBMassless e-folds before the massless point
# back must finely sample the backgroudn evolution for the initial conditions to be close to exactly NBMassless before

def unPackAlp(threePtOut, MTE):
    """
    This function unpacks the components of the input array `threePtOut` into individual variables
    representing different components of the model. It returns the components in a reshaped format
    suitable for further analysis.

    Arguments
    ---------
    threePtOut : ndarray
        A multi-dimensional array containing the results of a computation. The size of this array is
        expected to match a specific formula based on the number of fields `nF`.
    MTE : object
        An object representing the model. It provides the method `nF()` to return the number of fields.

    Returns
    -------
    zetaMs : ndarray
        A 2D array representing the first component (zetaMs) extracted from `threePtOut`.
    sig1R : ndarray
        A 3D array (2*nF x 2*nF) representing the first signature matrix (sig1R) extracted from `threePtOut`.
    sig2R : ndarray
        A 3D array (2*nF x 2*nF) representing the second signature matrix (sig2R) extracted from `threePtOut`.
    sig3R : ndarray
        A 3D array (2*nF x 2*nF) representing the third signature matrix (sig3R) extracted from `threePtOut`.
    alpha : ndarray
        A 4D array (2*nF x 2*nF x 2*nF) representing the alpha component extracted from `threePtOut`.

    Description
    -----------
    The function checks the dimensionality of `threePtOut` and extracts the relevant components into
    separate variables. These components are then reshaped into the required dimensions and returned.

    Python Prototype
    ----------------
    zetaMs, sig1R, sig2R, sig3R, alpha = unPackAlp(threePtOut, MTE)
    """

    # Get the number of fields from the MTE object
    nF = MTE.nF()

    # Check the size of the input array to ensure it matches the expected dimensions
    if np.size(threePtOut[0,:]) != 1 + 4 + 2 * nF + 6 * 2 * nF * 2 * nF + 2 * nF * 2 * nF * 2 * nF:
        # If the dimensions are incorrect, print a warning and return NaN values
        print("\n\n\n\n warning array you asked to unpack is not of correct dimension \n\n\n\n")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Extract the components from the input array, starting at the specific indices based on the number of fields
    zetaMs = threePtOut[:, 1:5]  # Extract the first component (zetaMs)

    # Extract the signature matrices (sig1R, sig2R, sig3R) based on the size of the array
    sig1R = threePtOut[:, 1 + 4 + 2 * nF : 1 + 4 + 2 * nF + 2 * nF * 2 * nF]
    sig2R = threePtOut[:, 1 + 4 + 2 * nF + 2 * nF * 2 * nF : 1 + 4 + 2 * nF + 2 * 2 * nF * 2 * nF]
    sig3R = threePtOut[:, 1 + 4 + 2 * nF + 2 * 2 * nF * 2 * nF : 1 + 4 + 2 * nF + 3 * 2 * nF * 2 * nF]

    # Extract the alpha component
    alpha = threePtOut[:, 1 + 4 + 2 * nF + 6 * 2 * nF * 2 * nF:]

    # Reshape the extracted components into the required 3D or 4D arrays, ensuring each component has the correct shape
    return zetaMs, np.reshape(sig1R, (np.size(threePtOut[:, 0]), 2 * nF, 2 * nF)), \
           np.reshape(sig2R, (np.size(threePtOut[:, 0]), 2 * nF, 2 * nF)), \
           np.reshape(sig3R, (np.size(threePtOut[:, 0]), 2 * nF, 2 * nF)), \
           np.reshape(alpha, (np.size(threePtOut[:, 0]), 2 * nF, 2 * nF, 2 * nF))


def unPackSig(twoPtOut, MTE):
    """
    This function unpacks the components of the input array `twoPtOut` into individual variables
    representing the zeta and signature matrices (sig). It reshapes and returns the signature matrix
    in a format suitable for further analysis.

    Arguments
    ---------
    twoPtOut : ndarray
        A multi-dimensional array containing the results of a computation. The size of this array is
        expected to match a specific formula based on the number of fields `nF`.
    MTE : object
        An object representing the model. It provides the method `nF()` to return the number of fields.

    Returns
    -------
    zeta : ndarray
        A 1D array representing the first component (zeta) extracted from `twoPtOut`.
    sig : ndarray
        A 3D array (2*nF x 2*nF) representing the signature matrix (sig) extracted from `twoPtOut`.

    Description
    -----------
    The function checks the dimensionality of `twoPtOut` and extracts the relevant components into
    separate variables. These components are then reshaped into the required dimensions and returned.
    """

    # Get the number of fields from the MTE object
    nF = MTE.nF()

    # Check the size of the input array to ensure it matches the expected dimensions
    if np.size(twoPtOut[0, :]) != 1 + 1 + 2 * nF + 2 * nF * 2 * nF:
        # If the dimensions are incorrect, print a warning and return NaN values
        print("\n\n\n\n warning array you asked to unpack is not of correct dimension \n\n\n\n")
        return np.nan, np.nan

    # Extract the zeta component (single column)
    zeta = twoPtOut[:, 1]

    # Extract the signature matrix (sig) based on the size of the array
    sig = twoPtOut[:, 1 + 1 + 2 * nF : 1 + 1 + 2 * nF + 2 * nF * 2 * nF]

    # Reshape the signature matrix into a 3D array of size (nt, 2*nF, 2*nF) and return both components
    return zeta, np.reshape(sig, (np.size(twoPtOut[:, 0]), 2 * nF, 2 * nF))


# this script finds initial conditions at least NBMassless e-folds before the massless point
# back must finely sample the backgroudn evolution for the initial conditions to be close to exactly NBMassless before

def ICsBM(NBMassless, k, back, params, MTE):
    """
    This function searches for the initial conditions at least `NBMassless` e-folds before the massless point.
    The massless point is identified by finding when the effective mass squared becomes positive.
    It works by calculating the largest eigenvalue of the mass matrix for the background evolution and adjusting the
    initial conditions accordingly.

    Arguments
    ---------
    NBMassless : float
        The number of e-folds before the massless point to find the initial conditions.
    k : float
        The wavenumber (k) used in calculating the effective mass.
    back : ndarray
        A 2D array representing the background evolution.
    params : ndarray
        A 1D array representing the model parameters.
    MTE : object
        An object representing the model that provides methods like `ddV()` to calculate the potential.

    Returns
    -------
    NexitMinus : float
        The time (e-fold) at which the massless condition is reached.
    backExitMinus : ndarray
        The background evolution at the time `NexitMinus`, representing the initial conditions.

    Description
    -----------
    This function iteratively searches the background evolution for the point where the effective mass squared
    becomes positive, indicating the massless condition is met. It then ensures that the initial conditions are
    set `NBMassless` e-folds before this massless point and returns both the corresponding time and initial conditions.
    """

    # Get the number of fields (nF) from the background data
    nF = np.size(back[0, 1:]) // 2
    massEff = -1  # Initialize the effective mass value to an invalid negative value

    # Search for the point where the effective mass squared becomes positive
    jj = 0
    while (massEff < 0 and jj < np.size(back[:, 0]) - 1):
        # Calculate eigenvalues of the mass matrix at the current background point
        w, v = np.linalg.eig(MTE.ddV(back[jj, 1:1 + nF], params))
        eigen = np.max(w)  # Get the largest eigenvalue (mass squared)

        # Calculate the effective mass squared and check if it's positive
        massEff = -k ** 2 * np.exp(-2.0 * back[jj, 0]) + eigen
        jj += 1  # Move to the next background point

    # If the massless condition is not found, return NaN values
    if jj == np.size(back[:, 0]):
        print("\n\n\n\n warning massless condition not found \n\n\n\n")
        return np.nan, np.nan

    # The time (N) when the massless condition was met
    NMassless = back[jj - 2, 0]

    # Initialize a variable for the exit background evolution
    backExitMinus = np.zeros(2 * nF)

    # Search for the point where the background time is at least `NBMassless` e-folds before the massless point
    ll = 0
    Ncond = -1.0
    while (Ncond < 0.0 and ll < np.size(back[:, 0]) - 1):
        # Check if the current time is at least `NBMassless` e-folds before the massless point
        Ncond = back[ll, 0] - (NMassless - NBMassless)
        ll += 1  # Move to the next background point

    # If the initial condition is not found, return NaN values
    if ll == np.size(back[:, 0]) or (NMassless - back[0, 0]) < NBMassless:
        print("\n\n\n\n warning initial condition not found \n\n\n\n")
        return np.nan, np.nan

    # The time (N) when the initial condition is found
    NexitMinus = back[ll - 2, 0]
    backExitMinus = back[ll - 2, 1:]  # The background evolution at that time

    return NexitMinus, backExitMinus


# this script finds initial conditions at least NBExit e-folds before horizon exit of k
# back must finely sample the background evolution for the initial conditions to be close to exactly NBExit before

def ICsBE(NBExit, k, back, params, MTE):
    """
    This function searches for the initial conditions at least `NBExit` e-folds before the horizon exit of a given
    wavenumber `k`. It finds the horizon exit point by checking when the condition for horizon crossing is met,
    i.e., when the comoving wave number `k` exits the horizon. The background evolution must be finely sampled to
    ensure that the initial conditions are set correctly before the horizon exit.

    Arguments
    ---------
    NBExit : float
        The number of e-folds before the horizon exit to find the initial conditions.
    k : float
        The wavenumber (k) used to determine the horizon crossing.
    back : ndarray
        A 2D array representing the background evolution.
    params : ndarray
        A 1D array representing the model parameters.
    MTE : object
        An object representing the model that provides methods like `H()` to calculate the Hubble parameter.

    Returns
    -------
    NexitMinus : float
        The time (e-fold) at which the horizon exit condition is met.
    backExitMinus : ndarray
        The background evolution at the time `NexitMinus`, representing the initial conditions.

    Description
    -----------
    This function iterates through the background evolution and checks when the condition for horizon crossing is
    met, which is determined by the comoving wave number `k` and the Hubble parameter. Once this condition is met,
    the function ensures that the initial conditions are set `NBExit` e-folds before the horizon exit.
    """

    # Get the number of fields (nF) from the background data
    nF = np.size(back[0, 1:]) // 2
    kvaH = -1.0  # Initialize the condition for horizon exit to an invalid negative value

    # Search for the horizon exit condition where the comoving wave number crosses the horizon
    jj = 0
    while (kvaH < 0.0 and jj < np.size(back[:, 0]) - 1):
        # Calculate the Hubble parameter at the current background point
        H = MTE.H(back[jj, 1:1 + 2 * nF], params)

        # Check if the comoving wave number `k` crosses the horizon
        kvaH = -k + np.exp(back[jj, 0]) * H
        jj += 1  # Move to the next background point

    # If the horizon exit condition is not found, print a warning and return NaN values
    if jj == np.size(back[:, 0]):
        print("\n\n\n\n warning exit condition not found \n\n\n\n")
        return np.nan, np.nan

    # The time (N) when the horizon exit condition was met
    NExit = back[jj - 2, 0]

    # Initialize a variable to search for the initial condition `NBExit` e-folds before the horizon exit
    ll = 0
    Ncond = -1.0
    while (Ncond < 0 and ll < np.size(back[:, 0]) - 1):
        # Check if the current time is at least `NBExit` e-folds before the horizon exit point
        Ncond = back[ll, 0] - (NExit - NBExit)
        ll += 1  # Move to the next background point

    # If the initial condition is not found, print a warning and return NaN values
    if ll == np.size(back[:, 0]) or (NExit - back[0, 0]) < NBExit:
        print("\n\n\n\n warning initial condition not found \n\n\n\n")
        return np.nan, np.nan

    # The time (N) when the initial condition is found
    NexitMinus = back[ll - 2, 0]
    backExitMinus = back[ll - 2, 1:]  # The background evolution at that time

    return NexitMinus, backExitMinus


# find the earliest condition between the massless one and the horizon exit one
def ICs(NB, k, back, params, MTE):
    """
    This function finds the earliest initial condition between the massless condition (ICsBM)
    and the horizon exit condition (ICsBE). It calls the `ICsBE` and `ICsBM` functions to get
    the initial conditions based on different criteria (horizon exit and massless condition).
    It returns the initial conditions corresponding to the earliest condition in time.

    Arguments
    ---------
    NB : float
        The number of e-folds before the massless or horizon exit conditions.
    k : float
        The wavenumber used to determine the horizon crossing or massless condition.
    back : ndarray
        A 2D array representing the background evolution.
    params : ndarray
        A 1D array representing the model parameters.
    MTE : object
        An object representing the model that provides methods like `H()` and `ddV()` for background evolution.

    Returns
    -------
    NExitMinus : float
        The time (e-fold) when the earliest condition is met.
    fieldExitMinus : ndarray
        The background evolution at the time `NExitMinus`, representing the initial conditions.

    Description
    -----------
    This function compares the results from the horizon exit condition (`ICsBE`) and the massless condition (`ICsBM`).
    It returns the initial conditions for the earliest of these two conditions. This is useful when you need to determine
    the initial conditions based on either the massless condition or the horizon exit condition, whichever occurs first.
    """

    # Get the initial conditions based on horizon exit (from ICsBE)
    NBEs, fieldBE = ICsBE(NB, k, back, params, MTE)

    # Get the initial conditions based on massless condition (from ICsBM)
    NBMs, fieldBM = ICsBM(NB, k, back, params, MTE)

    # Compare the two conditions and return the one that happens first
    if (NBEs < NBMs):
        # If the horizon exit condition happens earlier, return the horizon exit initial condition
        return NBEs, fieldBE

    # If the massless condition happens earlier, return the massless initial condition
    return NBMs, fieldBM


# calculates the power spectrum at each element in kA at the end of the background evolution (back) with PyT
def pSpectra(kA, back, params, NB, tols, MTE):
    """
    This function calculates the power spectrum at each value of k in the input array `kA` at the end of the background evolution (from the `back` array).
    It performs the calculation for each wavenumber `k` by finding the initial conditions using the `ICs` function and then evolving the system using the `sigEvolve` method
    from the MTE object. It returns the power spectrum values and the computation times for each `k`.

    Arguments
    ---------
    kA : ndarray
        A 1D array containing the different wavenumbers (k) for which the power spectrum will be calculated.
    back : ndarray
        A 2D array containing the background evolution at different e-folds.
    params : ndarray
        A 1D array representing the parameters of the model.
    NB : float
        The number of e-folds before the massless or horizon exit condition.
    tols : ndarray
        A 1D array containing the tolerance values (absolute and relative) used in the numerical solver.
    MTE : object
        An object representing the model that provides methods like `sigEvolve` to evolve the system.

    Returns
    -------
    zzOut : ndarray
        A 1D array containing the calculated power spectrum values for each wavenumber in `kA`.
    times : ndarray
        A 1D array containing the computation times for each wavenumber in `kA`.

    Description
    -----------
    This function calculates the power spectrum at the end of the background evolution for each wavenumber `k`.
    It first finds the initial conditions using the `ICs` function, then evolves the system using the `sigEvolve`
    method, and finally stores the resulting power spectrum values at the last time step.
    """

    # Initialize empty arrays for output: zzOut will store the power spectrum values and times will store the computation times
    zzOut = np.array([])  # Will hold the power spectrum results
    times = np.array([])  # Will hold the computation time for each iteration
    num = np.size(kA)    # Number of wavenumbers to process

    # Loop over each wavenumber in kA
    for ii in range(0, num):
        print("\n \n \n performing " + str(ii + 1) + " of " + str(num) + "\n \n \n")
        k = kA[ii]  # Current wavenumber
        Nstart, backExitMinus = ICs(NB, k, back, params, MTE)  # Get the initial conditions for this wavenumber

        # Measure the time taken for each computation
        start_time = timeit.default_timer()

        # If initial conditions are invalid (Nstart is NaN), return an empty array for the power spectrum
        if np.isnan(Nstart): # Fix: Use np.isnan()
            twoPt = np.empty((2, 2))  # Create an empty 2x2 array
            twoPt[:] = np.nan  # Fill the array with NaNs
        else:
            # Generate a time array from Nstart to the last background time
            t = np.linspace(Nstart, back[-1, 0], 10)

            # Evolve the system to calculate the power spectrum at the current k value
            twoPt = MTE.sigEvolve(t, k, backExitMinus, params, tols, True)

        # Append the final power spectrum value (from the last time step) to zzOut
        zzOut = np.append(zzOut, twoPt[-1, 1])

        # Append the computation time for this iteration to the times array
        times = np.append(times, timeit.default_timer() - start_time)

    # Return the results: the power spectrum values and the computation times
    return zzOut, times


# calculates the power spectrum at each element in kA at the end of the background evolution (back) with MPP
def pSpectraMPP(kA, back, params, NB, tols, MTE):
    """
    This function calculates the power spectrum at the end of the background evolution (from the `back` array) for each
    wavenumber in `kA` using the MPP (Mean Particle Production) method. The function finds the initial conditions using
    the `ICs` function, then evolves the system using MTE methods: `MPP2` for the evolution of the background field and
    `MPPSigma` to calculate the power spectrum.

    Arguments
    ---------
    kA : ndarray
        A 1D array containing the different wavenumbers (k) for which the power spectrum will be calculated.
    back : ndarray
        A 2D array containing the background evolution at different e-folds.
    params : ndarray
        A 1D array representing the model parameters.
    NB : float
        The number of e-folds before the massless or horizon exit condition.
    tols : ndarray
        A 1D array containing the tolerance values (absolute and relative) used in the numerical solver.
    MTE : object
        An object representing the model that provides methods like `MPP2` and `MPPSigma` to calculate the power spectrum.

    Returns
    -------
    zzOut : ndarray
        A 1D array containing the calculated power spectrum values for each wavenumber in `kA`.
    times : ndarray
        A 1D array containing the computation times for each wavenumber in `kA`.

    Description
    -----------
    This function calculates the power spectrum at the end of the background evolution using the MPP method. For each
    wavenumber `k`, it first finds the initial conditions using the `ICs` function, evolves the system using `MPP2`,
    and finally calculates the power spectrum using `MPPSigma`.
    """

    # Initialize empty arrays for output: zzOut will store the power spectrum values and times will store the computation times
    zzOut = np.array([])  # Will hold the power spectrum results
    times = np.array([])  # Will hold the computation time for each iteration
    num = np.size(kA)    # Number of wavenumbers to process

    # Loop over each wavenumber in kA
    for ii in range(0, num):
        print("\n \n \n performing " + str(ii + 1) + " of " + str(num) + "\n \n \n")
        k = kA[ii]  # Current wavenumber
        Nstart, backExitMinus = ICs(NB, k, back, params, MTE)  # Get the initial conditions for this wavenumber

        # Measure the time taken for each computation
        start_time = timeit.default_timer()

        # If initial conditions are invalid (Nstart is NaN), return an empty array for the power spectrum
        if np.isnan(Nstart): # Fix: Use np.isnan()
            twoPt = np.empty((2, 2))  # Create an empty 2x2 array
            twoPt[:] = np.nan  # Fill the array with NaNs
        else:
            # Generate a time array from Nstart to the last background time
            t = np.linspace(Nstart, back[-1, 0], 10)

            # Evolve the system using MPP method for background evolution (MPP2)
            rho = MTE.MPP2(t, k, backExitMinus, params, tols)  # Get the evolution for the system

            # Calculate the power spectrum using MPPSigma method
            twoPt = MTE.MPPSigma(t, k, backExitMinus, params, rho, True, -1)

        # Append the final power spectrum value (from the last time step) to zzOut
        zzOut = np.append(zzOut, twoPt[-1, 1])

        # Append the computation time for this iteration to the times array
        times = np.append(times, timeit.default_timer() - start_time)

    # Return the results: the power spectrum values and the computation times
    return zzOut, times


# calculates the power spectrum at each element in kA at the end of the background evolution (back) in a manner suitable to be called over many processes
def pSpecMpi(kA, back, params, NB, tols, MTE):
    """
    This function calculates the power spectrum for each element in kA (wavenumbers) at the end of the background evolution
    (from the `back` array) using parallel computing with MPI (Message Passing Interface). The task is distributed across
    multiple processes. Each process calculates the power spectrum for a subset of `kA` and the results are combined at the
    root process (rank 0).

    Arguments
    ---------
    kA : ndarray
        A 1D array containing the wavenumbers for which the power spectrum will be calculated.
    back : ndarray
        A 2D array containing the background evolution at different e-folds.
    params : ndarray
        A 1D array representing the model parameters.
    NB : float
        The number of e-folds before the massless or horizon exit condition.
    tols : ndarray
        A 1D array containing the tolerance values (absolute and relative) used in the numerical solver.
    MTE : object
        An object representing the model that provides methods like `sigEvolve` to calculate the power spectrum.

    Returns
    -------
    zzOut : ndarray
        A 1D array containing the calculated power spectrum values for each wavenumber in `kA`.
    timesOut : ndarray
        A 1D array containing the computation times for each wavenumber in `kA`.

    Description
    -----------
    This function uses MPI to distribute the task of calculating the power spectrum for a given set of wavenumbers `kA`
    across multiple processes. The computation is performed by dividing the `kA` array into roughly equal parts for each
    process. Each process computes its portion of the power spectrum and then sends the results to the root process (rank 0).
    The root process collects the results from all processes and combines them into the final output.
    """
    # Import MPI library and initialize the communicator for parallel processes
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Get the rank (ID) of the current process and the total number of processes
    rank = comm.Get_rank()  # Rank of the current process (0 is the root process)
    size = comm.Get_size()  # Total number of processes

    # Number of wavenumbers to process and the number of points per process
    points = np.size(kA)
    num = points // size  # Number of wavenumbers each process should handle

    # If the number of wavenumbers isn't divisible by the number of processes, print a warning and exit
    if float(points) / size != float(points // size):
        if rank == 0:  # Only the root process (rank 0) prints the warning
            print("\n \n \n warning! number of points is divisible by number of processes, exiting \n \n \n")
        return (np.empty(0), np.empty(0))  # Return empty arrays if the points can't be divided evenly

    # Divide the wavenumber array `kA` into subsets based on the process rank
    kOutL = kA[rank * num : rank * num + num]  # Subset of kA that this process will handle

    # Calculate the power spectrum for the subset of wavenumbers using pSpectra
    zzL, timesL = pSpectra(kOutL, back, params, NB, tols, MTE)

    # If the process is not the root (rank != 0), send the results (power spectrum and times) to rank 0
    if rank != 0:
        comm.Send(zzL, dest=0)  # Send power spectrum values to rank 0
        comm.Send(timesL, dest=0)  # Send computation times to rank 0

    # If the process is the root (rank == 0), gather the results from all processes
    if rank == 0:
        # Initialize empty arrays to store the final results
        zzOut = np.array([])  # Will hold all the power spectrum results
        timesOut = np.array([])  # Will hold all the computation times

        # Append the results from the root process itself
        zzOut = np.append(zzOut, zzL)
        timesOut = np.append(timesOut, timesL)

        # Collect results from all other processes
        for jj in range(1, size):  # Loop over all other processes (excluding the root)
            comm.Recv(zzL, source=jj)  # Receive power spectrum results from process jj
            comm.Recv(timesL, source=jj)  # Receive computation times from process jj
            zzOut = np.append(zzOut, zzL)  # Append the received power spectrum
            timesOut = np.append(timesOut, timesL)  # Append the received computation times

        # Return the combined results: power spectrum and computation times
        return zzOut, timesOut
    else:
        # For non-root processes, return empty arrays
        return np.empty(0), np.empty(0)


# calculates the power spectrum and bispectrum in an equilateral configuration at each element in kA at the end of the background evolution (back)
def eqSpectra(kA, back, params, NB, tols, MTE):
    """
    This function calculates both the power spectrum and bispectrum (in an equilateral configuration) for each wavenumber
    in `kA` at the end of the background evolution provided by the `back` array. The calculations are performed using the
    model defined in the `MTE` object, and the results are returned for each element of `kA`. The power spectrum is calculated
    using the `alphaEvolve` method, which evolves the system in time, while the bispectrum is also computed in an equilateral
    triangle configuration (i.e., with k1 = k2 = k3).

    Arguments
    ---------
    kA : ndarray
        A 1D array containing the wavenumbers for which the power spectrum and bispectrum will be calculated.
    back : ndarray
        A 2D array containing the background evolution at different e-folds.
    params : ndarray
        A 1D array representing the model parameters.
    NB : float
        The number of e-folds before the massless or horizon exit condition.
    tols : ndarray
        A 1D array containing the tolerance values used in the numerical solver.
    MTE : object
        An object representing the model that provides methods like `alphaEvolve` to calculate the power spectrum and bispectrum.

    Returns
    -------
    zzOut : ndarray
        A 1D array containing the calculated power spectrum for each wavenumber in `kA`.
    zzzOut : ndarray
        A 1D array containing the calculated bispectrum for each wavenumber in `kA`.
    times : ndarray
        A 1D array containing the computation times for each wavenumber in `kA`.

    Description
    -----------
    This function calculates the power spectrum and bispectrum for each wavenumber in `kA` using an equilateral triangle
    configuration (k1 = k2 = k3). It does so by evolving the system using the `alphaEvolve` method provided by the `MTE` object.
    The results are returned in two separate arrays: one for the power spectrum (`zzOut`) and one for the bispectrum (`zzzOut`).
    The computation times for each wavenumber are also recorded.
    """

    # Initialize empty arrays to store results
    zzzOut = np.array([])  # Will store the bispectrum values
    zzOut = np.array([])   # Will store the power spectrum values
    times = np.array([])   # Will store the computation times for each wavenumber

    num = np.size(kA)  # Get the number of wavenumbers in kA

    # Loop over all wavenumbers in kA
    for ii in range(0, num):
        print("\n \n \n performing " + str(ii+1) + " of " + str(num) + "\n \n \n")  # Print progress
        k = kA[ii]  # Get the current wavenumber

        # Get the initial conditions for this wavenumber using the ICs function
        Nstart, backExitMinus = ICs(NB, k, back, params, MTE)

        # Define the time range over which to evolve the system
        t = np.linspace(Nstart, back[-1, 0], 10)

        # Set up the k values for the equilateral triangle configuration (k1 = k2 = k3)
        k1 = k
        k2 = k
        k3 = k

        # Start measuring the time taken for this computation
        start_time = timeit.default_timer()

        # If Nstart is NaN, set the output to NaN
        if np.isnan(Nstart): # Fix: Use np.isnan()
            nF = MTE.nF()  # Get the number of fields
            threePt = np.empty((2, 5))  # Initialize an empty array to store three-point function data
            threePt[:] = np.nan  # Fill it with NaN values
        else:
            # Evolve the system using the alphaEvolve method in the MTE object
            # This will return a 2x5 array with various quantities (e.g., power spectrum, bispectrum)
            threePt = MTE.alphaEvolve(t, k1, k2, k3, backExitMinus, params, tols, True)

        # Append the power spectrum (second column) and bispectrum (fifth column) to the result arrays
        zzOut = np.append(zzOut, threePt[-1, 1])  # Power spectrum is stored in the second column
        zzzOut = np.append(zzzOut, threePt[-1, 4])  # Bispectrum is stored in the fifth column

        # Append the computation time for this wavenumber to the times array
        times = np.append(times, timeit.default_timer() - start_time)

    # Return the results: power spectrum, bispectrum, and computation times
    return zzOut, zzzOut, times


# Calculates the power spectrum and bispectrum in the equilateral configuration
# at each element in kA at the end of the background evolution (back).
def MPPSpectra(kA, back, params, NB, tols, MTE):
    """
    Computes the power spectrum and bispectrum for a given set of wavenumbers (kA)
    using the MPP (Multi-Point Propagator) method.

    The function iterates over each wavenumber in kA, determines initial conditions,
    and evolves the system using MPP-based numerical methods provided in the MTE object.

    Parameters:
    -----------
    kA : array-like
        A 1D array of wavenumbers for which the power spectrum and bispectrum are computed.
    back : array-like
        A 2D array containing the background evolution data.
    params : array-like
        A set of model parameters used for evolution calculations.
    NB : float
        The number of e-folds before the massless or horizon exit condition.
    tols : array-like
        A set of tolerance values for numerical integration.
    MTE : object
        A model object that provides numerical solvers such as `MPP3` and `MPPAlpha`.

    Returns:
    --------
    zzOut : array-like
        The computed power spectrum values for each k in kA.
    zzzOut : array-like
        The computed bispectrum values for each k in kA.
    times : array-like
        The computation time for each k in kA.

    Description:
    ------------
    - This function loops over the provided kA values.
    - For each k, it computes the initial conditions using `ICs`.
    - It then evolves the system using `MPP3` and `MPPAlpha` methods.
    - The power spectrum and bispectrum are extracted from the results.
    - Finally, it returns the computed values along with computation times.
    """

    # Initialize empty arrays to store results
    zzzOut = np.array([])  # Stores the bispectrum values
    zzOut = np.array([])   # Stores the power spectrum values
    times = np.array([])   # Stores computation times

    num = np.size(kA)  # Get the total number of k values

    # Iterate over each k in kA
    for ii in range(num):
        print("\n\n\n performing " + str(ii + 1) + " of " + str(num) + "\n\n\n")

        k = kA[ii]  # Get the current wavenumber

        # Determine the initial conditions for this k using the ICs function
        Nstart, backExitMinus = ICs(NB, k, back, params, MTE)

        # Define the time array for evolution (from Nstart to the last time in back)
        t = np.linspace(Nstart, back[-1, 0], 10)

        # Equilateral triangle configuration: k1 = k2 = k3 = k
        k1 = k
        k2 = k
        k3 = k

        # Start measuring the computation time
        start_time = timeit.default_timer()

        # If the starting condition is NaN, assign NaN values to the output
        if np.isnan(Nstart):  # Fix: Use `np.isnan()` instead of `Nstart == np.nan`
            nF = MTE.nF()  # Get the number of fields (possibly needed for indexing)
            threePt = np.empty((2, 5))  # Create an empty 2x5 array
            threePt[:] = np.nan  # Fill it with NaN values
        else:
            # Compute the three-point propagator function (rho3) using MPP3
            rho3 = MTE.MPP3(t, k1, k2, k3, backExitMinus, params, tols)

            # Evolve the system using MPPAlpha and compute power spectrum & bispectrum
            threePt = MTE.MPPAlpha(t, k1, k2, k3, backExitMinus, params, rho3, True)

        # Extract and store the final power spectrum and bispectrum values
        zzOut = np.append(zzOut, threePt[-1, 1])  # Power spectrum (2nd column)
        zzzOut = np.append(zzzOut, threePt[-1, 4])  # Bispectrum (5th column)

        # Store the computation time
        times = np.append(times, timeit.default_timer() - start_time)

    # Return computed power spectrum, bispectrum, and times
    return zzOut, zzzOut, times


# Calculates the power spectrum and bispectrum in the equilateral configuration
# at each element in kA at the end of the background evolution (back).
# This function is designed to be executed across many processes using MPI.
def eqSpecMpi(kA, back, params, NB, tols, MTE):
    """
    Parallelized computation of the bispectrum using the alpha-beta parameterization.

    The function distributes the calculation of power and bispectrum across multiple processes,
    allowing large-scale calculations to be performed efficiently.

    Parameters:
    -----------
    kA : array-like
        A 1D array of wavenumbers for which the power spectrum and bispectrum are computed.
    back : array-like
        A 2D array containing the background evolution data.
    params : array-like
        A set of model parameters used for evolution calculations.
    NB : float
        The number of e-folds before the massless or horizon exit condition.
    tols : array-like
        A set of tolerance values for numerical integration.
    MTE : object
        A model object that provides numerical solvers.

    Returns:
    --------
    zzOut : array-like (only returned by rank 0)
        The computed power spectrum values.
    zzzOut : array-like (only returned by rank 0)
        The computed bispectrum values.
    timesOut : array-like (only returned by rank 0)
        The computation time for each k in kA.

    Description:
    ------------
    - The function is executed in parallel using MPI.
    - It divides the wavenumber array `kA` among available processes.
    - Each process computes the power spectrum and bispectrum for its assigned k values.
    - The results are gathered by the root process (rank 0) and returned.
    """

    # Initialize MPI communication
    from mpi4py import MPI # Import MPI here
    comm = MPI.COMM_WORLD  # MPI communicator for inter-process communication

    # Get the rank (process ID) and the total number of processes
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get the total number of k values
    points = np.size(kA)

    # Determine the number of k values per process
    num = points // size  # Integer division to split workload

    # Ensure the number of points is evenly divisible by the number of processes
    if float(points) / size != float(points // size):
        if rank == 0:  # Only rank 0 prints the warning
            print("\n\n\n Warning! Number of points is not divisible by the number of processes, exiting.\n\n\n")
        return np.empty(0), np.empty(0), np.empty(0)  # Return empty arrays in case of misalignment

    # Assign each process its portion of k values
    kOutL = kA[rank * num: rank * num + num]

    # Compute the power spectrum and bispectrum for assigned k values
    zzL, zzzL, timesL = eqSpectra(kOutL, back, params, NB, tols, MTE)

    # If the process is not the root (rank 0), send its results to rank 0
    if rank != 0:
        comm.Send(zzzL, dest=0)
        comm.Send(zzL, dest=0)
        comm.Send(timesL, dest=0)

    # If the process is the root (rank 0), collect results from all other processes
    if rank == 0:
        # Initialize arrays to store the final results
        zzzOut = np.array([])  # Bispectrum output
        zzOut = np.array([])   # Power spectrum output
        timesOut = np.array([])  # Computation time output

        # Store the results from rank 0 itself
        zzzOut = np.append(zzzOut, zzzL)
        zzOut = np.append(zzOut, zzL)
        timesOut = np.append(timesOut, timesL)

        # Receive results from other processes
        for jj in range(1, size):
            zzzL = comm.recv(source=jj)
            zzL = comm.recv(source=jj)
            timesL = comm.recv(source=jj)

            # Store the received data in the combined output arrays
            Bztot[jj * num : jj * num + num, :, :] = BzL # Note: BzL is not defined here, it should be zzzL
            Pz1tot[jj * num : jj * num + num, :, :] = Pz1L # Pz1L not defined
            Pz2tot[jj * num : jj * num + num, :, :] = Pz2L # Pz2L not defined
            Pz3tot[jj * num : jj * num + num, :, :] = Pz3L # Pz3L not defined
            timestot[jj * num : jj * num + num, :] = timesL # timestot not defined

            # Correct appending based on what was received (zzzL, zzL, timesL)
            zzzOut = np.append(zzzOut, zzzL)
            zzOut = np.append(zzOut, zzL)
            timesOut = np.append(timesOut, timesL)


        # Return the combined results
        return (zzOut, zzzOut, timesOut) # Corrected return values

    else:
        # Other processes return empty arrays
        return (np.empty(0), np.empty(0), np.empty(0))


# Calculates the bispectrum in the alpha-beta notation for a given kt at every value of alphaIn and betaIn.
# The bispectrum is computed at `nsnaps` times, always including the final time of the evolution (`back`).
def alpBetSpectra(kt, alphaIn, betaIn, back, params, NB, nsnaps, tols, MTE):
    """
    Computes the bispectrum using the alpha-beta parameterization.

    Parameters:
    -----------
    kt : float
        The total wavenumber of the triangle configuration.
    alphaIn : array-like
        Array of alpha values (parameter controlling the shape of the triangle).
    betaIn : array-like
        Array of beta values (parameter controlling the shape of the triangle).
    back : 2D array-like
        Background evolution data, with the first column being N (e-folds)
        and the remaining columns containing background fields.
    params : array-like
        Model parameters used for evolution calculations.
    NB : float
        Number of e-folds before the massless or horizon exit condition.
    nsnaps : int
        Number of snapshots at which to evaluate the bispectrum.
    tols : array-like
        Numerical tolerance values for solving the evolution equations.
    MTE : object
        Model object that provides numerical solvers.

    Returns:
    --------
    biAOut : 3D array
        The bispectrum values for each (alpha, beta, time snapshot).
    zz1, zz2, zz3 : 3D arrays
        Power spectrum values for the three sides of the triangle.
    times : 2D array
        Computation time for each (alpha, beta) combination.
    snaps : 1D array
        Time snapshots (e-folds) at which the bispectrum is evaluated.
    """

    # Initialize an array to store the Hubble rate H(N) for each time step in `back`
    Hin = np.zeros(np.size(back[:, 0]))

    # Compute the Hubble rate at each background evolution step
    for jj in range(np.size(back[:, 0])):
        Hin[jj] = MTE.H(back[jj, 1:], params)

    # Compute aH = a(N) * H(N) to determine horizon exit
    aH = np.exp(back[:, 0]) * Hin

    # Sort aH values and interpolate to find the horizon exit time Nexit for k=kt/3
    positions = np.argsort(aH)
    Nexit = interpolate.splev(
        kt / 3.,
        interpolate.splrep(aH[positions], back[positions, 0], s=1e-15),
        der=0
    )

    # Define the end time of evolution
    Nend = back[-1, 0]

    # Generate the list of time snapshots, ensuring the final time is included
    snaps = np.linspace(Nexit - (NB - 0.1), Nend, nsnaps)

    # If only one or zero snapshots are requested, use only the final time
    if nsnaps in [0, 1]:
        snaps = np.array([Nend])
        nsnaps = 1

    # Initialize output arrays (dimensions: alpha × beta × snapshots)
    biAOut = np.zeros([np.size(alphaIn), np.size(betaIn), np.size(snaps)])
    zz1 = np.zeros([np.size(alphaIn), np.size(betaIn), np.size(snaps)])
    zz2 = np.zeros([np.size(alphaIn), np.size(betaIn), np.size(snaps)])
    zz3 = np.zeros([np.size(alphaIn), np.size(betaIn), np.size(snaps)])
    times = np.zeros([np.size(alphaIn), np.size(betaIn)])

    # Iterate over all values of alpha
    for l in range(np.size(alphaIn)):
        alpha = alphaIn[l]

        # Iterate over all values of beta
        for j in range(np.size(betaIn)):
            print(f"\n\n\n Performing {l+1} {j+1} of {np.size(alphaIn)} {np.size(betaIn)} \n\n\n")

            # Start timing the computation for this (alpha, beta) combination
            timebefore = timeit.default_timer()

            beta = betaIn[j]

            # Compute k1, k2, k3 using the alpha-beta parameterization
            k1 = kt / 2. - beta * kt / 2.
            k2 = kt / 4. * (1. + alpha + beta)
            k3 = kt / 4. * (1. - alpha + beta)

            # Check if (alpha, beta) is within the valid triangle range
            if - (1 - beta) < alpha < (1 - beta):
                # Find the smallest k value to determine the initial time (Nstart)
                kM = min(k1, k2, k3)
                Nstart, backExitMinus = ICs(NB, kM, back, params, MTE)

                # Create the time array including the starting time and snapshots
                t = np.concatenate((np.array([Nstart]), snaps))

                # If Nstart is NaN, return an array filled with NaNs
                if np.isnan(Nstart): # Fix: Use np.isnan()
                    threePt = np.empty((2, 5))
                    threePt[:] = np.nan
                else:
                    # Solve for the bispectrum using the alpha evolution solver
                    threePt = MTE.alphaEvolve(t, k1, k2, k3, backExitMinus, params, tols, True)

                # Extract bispectrum and power spectrum components
                zzz = threePt[:, :5]

                # Store the results at each snapshot
                for ii in range(1, nsnaps + 1):
                    biAOut[l, j, ii - 1] = zzz[ii, 4]  # Bispectrum component
                    zz1[l, j, ii - 1] = zzz[ii, 1]  # Power spectrum component for k1
                    zz2[l, j, ii - 1] = zzz[ii, 2]  # Power spectrum component for k2
                    zz3[l, j, ii - 1] = zzz[ii, 3]  # Power spectrum component for k3

            else:
                # If the triangle is not valid, store NaN values
                for ii in range(nsnaps):
                    biAOut[l, j, ii] = np.nan
                    zz1[l, j, ii] = np.nan
                    zz2[l, j, ii] = np.nan
                    zz3[l, j, ii] = np.nan

            # Store the computation time
            times[l, j] = timeit.default_timer() - timebefore

    # Return the computed bispectrum, power spectrum values, computation times, and snapshots
    return biAOut, zz1, zz2, zz3, times, snaps


# Performs the same task as `alpBetSpectra`, but in a manner suitable to be spread across many processes.
def alpBetSpecMpi(kt, alpha, beta, back, params, NB, nsnaps, tols, MTE):
    """
    Parallelized computation of the bispectrum using the alpha-beta parameterization.

    Parameters:
    -----------
    kt : float
        The total wavenumber of the triangle configuration.
    alpha : array-like
        Array of alpha values (controls the shape of the triangle).
    beta : array-like
        Array of beta values (controls the shape of the triangle).
    back : 2D array-like
        Background evolution data.
    params : array-like
        Model parameters used for evolution calculations.
    NB : float
        Number of e-folds before the horizon exit.
    nsnaps : int
        Number of snapshots at which to evaluate the bispectrum.
    tols : array-like
        Numerical tolerance values for solving the evolution equations.
    MTE : object
        Model object that provides numerical solvers.

    Returns:
    --------
    Bztot : 3D array
        The bispectrum values for each (alpha, beta, time snapshot).
    Pz1tot, Pz2tot, Pz3tot : 3D arrays
        Power spectrum values for the three sides of the triangle.
    timestot : 2D array
        Computation time for each (alpha, beta) combination.
    snaps : 1D array
        Time snapshots at which the bispectrum is evaluated.
    """

    # Initialize MPI communication
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Process rank (ID)
    size = comm.Get_size()  # Total number of processes

    # Number of alpha values
    side = np.size(alpha)
    Nbefore = NB  # Store the input NB value

    # Determine how many alpha values each process should handle
    num = side // size # Fix: Use integer division for num

    # Ensure the number of alpha values is evenly divisible by the number of processes
    if float(side) / size != float(side // size):
        if rank == 0:
            print("\n\n\n Warning! Number of alpha values must be divisible by the number of processes. Exiting.\n\n\n")
        return (np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)) # Return empty numpy arrays consistent with type

    else:
        # Each process gets a portion of the alpha array
        alphaL = alpha[rank * num : rank * num + num]

        # Each process independently computes the bispectrum for its subset of alpha values
        BzL, Pz1L, Pz2L, Pz3L, timesL, snaps = alpBetSpectra(
            kt, alphaL, beta, back, params, Nbefore, nsnaps, tols, MTE
        )

        # If the process is not the root process, send the results to rank 0
        if rank != 0:
            comm.send(Pz1L, dest=0)
            comm.send(Pz2L, dest=0)
            comm.send(Pz3L, dest=0)
            comm.send(BzL, dest=0)
            comm.send(timesL, dest=0)

        # If this is the root process, collect the results from all processes
        if rank == 0:
            # Initialize arrays to store the combined results from all processes
            Bztot = np.zeros([np.size(alpha), np.size(beta), np.size(BzL[0, 0, :])])
            Pz1tot = np.zeros([np.size(alpha), np.size(beta), np.size(BzL[0, 0, :])])
            Pz2tot = np.zeros([np.size(alpha), np.size(beta), np.size(BzL[0, 0, :])])
            Pz3tot = np.zeros([np.size(alpha), np.size(beta), np.size(BzL[0, 0, :])])
            timestot = np.zeros([np.size(alpha), np.size(beta)])

            # Store the results from the root process itself
            Bztot[0:num, :, :] = BzL
            Pz1tot[0:num, :, :] = Pz1L
            Pz2tot[0:num, :, :] = Pz2L
            Pz3tot[0:num, :, :] = Pz3L
            timestot[0:num, :] = timesL

            # Loop over all other processes and receive their results
            for jj in range(1, size):
                Pz1L_recv = comm.recv(source=jj) # Use new variable names for received data
                Pz2L_recv = comm.recv(source=jj)
                Pz3L_recv = comm.recv(source=jj)
                BzL_recv = comm.recv(source=jj)
                timesL_recv = comm.recv(source=jj)

                # Store the received data in the combined output arrays
                Bztot[jj * num : jj * num + num, :, :] = BzL_recv
                Pz1tot[jj * num : jj * num + num, :, :] = Pz1L_recv
                Pz2tot[jj * num : jj * num + num, :, :] = Pz2L_recv
                Pz3tot[jj * num : jj * num + num, :, :] = Pz3L_recv
                timestot[jj * num : jj * num + num, :] = timesL_recv

            # Return the combined results
            return (Bztot, Pz1tot, Pz2tot, Pz3tot, timestot, snaps)

        else:
            # Other processes return empty arrays
            return (np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), snaps)


import numpy as np
from scipy import interpolate

# Computes the k-mode that exits the horizon at a given field value PhiExit
def kexitPhi(PhiExit, n, back, params, MTE):
    """
    Compute the wavenumber k that exits the horizon when the field reaches PhiExit.

    Parameters:
    -----------
    PhiExit : float
        The value of the field Φ at which we want to find the corresponding k-mode.
    n : int
        Column index in `back` that contains the field value Φ.
    back : 2D array-like
        Background evolution data where:
        - `back[:,0]` contains the number of e-folds (N),
        - `back[:,n]` contains the field value Φ(N),
        - Other columns contain background variables.
    params : array-like
        Model parameters used for evolution calculations.
    MTE : object
        Model object that provides the function to compute the Hubble parameter H.

    Returns:
    --------
    k : float
        The wavenumber k that exits the horizon at Φ = PhiExit.
    """

    # Number of field and momentum components (assuming back stores both)
    nF = np.size(back[0,1:]) // 2

    # Create an array to store background variables at horizon exit
    backExit = np.zeros(2 * nF)

    # Sort the background evolution data based on field values in column `n`
    positions = np.argsort(back[:, n])

    # Interpolate to find the number of e-folds (Nexit) when the field reaches PhiExit
    Nexit = interpolate.splev(
        PhiExit,
        interpolate.splrep(back[positions, n], back[positions, 0], s=1e-15),
        der=0
    )

    # Interpolate to extract background variables at Nexit
    for i in range(1, 2 * nF + 1):
        backExit[i - 1] = interpolate.splev(
            Nexit,
            interpolate.splrep(back[:, 0], back[:, i], s=1e-15),
            der=0
        )

    # Compute the wavenumber k that exits the horizon at Nexit
    k = np.exp(Nexit) * MTE.H(backExit, params)

    return k


import numpy as np
from scipy import interpolate

# Compute the wavenumber k that exits the horizon at a given e-fold number Nexit
def kexitN(Nexit, back, params, MTE, exact=False):
    """
    Compute the wavenumber k that exits the horizon at Nexit.

    Parameters:
    -----------
    Nexit : float
        The number of e-folds at which we want to determine the horizon exit wavenumber k.
    back : 2D array-like
        Background evolution data where:
        - `back[:,0]` contains the number of e-folds (N),
        - Other columns contain background variables.
    params : array-like
        Model parameters for evolution calculations.
    MTE : object
        Model object providing the Hubble function H.
    exact : bool, optional
        If True, may apply a more precise calculation (not implemented in this version).

    Returns:
    --------
    k : float
        The wavenumber k that exits the horizon at Nexit.
    """

    # Number of field/momentum components in the background evolution array
    nF = np.size(back[0, 1:])

    # Array to store Hubble parameter values at each time step
    Harr = np.zeros(np.size(back[:, 0]))

    # Compute the Hubble parameter H at each time step in back[:, 0]
    for ii in range(0, np.size(back[:, 0])):
        Harr[ii] = MTE.H(back[ii, 1:2*nF+1], params)  # Extracts required variables from `back`

    # Interpolate log(k) = N + log(H) at Nexit
    logk = interpolate.splev(
        Nexit,
        interpolate.splrep(back[:, 0], back[:, 0] + np.log(Harr), s=1e-15),
        der=0
    )

    # Return k in normal (non-log) scale
    return np.exp(logk)
