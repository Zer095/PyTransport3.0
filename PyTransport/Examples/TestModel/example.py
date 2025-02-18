####################################### Example of Standard PyTransport #######################################
'''
In this example, we'll show how to use PyTransport to compute inflationary observables.
We'll use the model defined in xSetup.py. Remember to setup the model by running the command: python xSetup.py

The fundamental steps are:
    - Import packages;
    - Import PyTransport;
    - Initialize model.
These steps are needed for every script.

Then we'll show:
    - How to run a background evolution, and how to plot it.
      This is needed for any further calculation.
    - How to run a phase-space 2pt evolution, 
      how to compute the curvature perturbation power spectrum,
      how to plot it. 
    - How to run a tensor 2pt evolution,
    - How to run a phase-space 3pt evolution,
      how to compute the reduced bispectrum f_NL and how to plot it.

In this example, we we'll use the Transport equation. 
For an example on how to use the MPP formalism, see MPPExample.py.
'''

################################################### Import ###################################################
from matplotlib import pyplot as plt     # Standard package for plotting
from pylab import *                      # contains useful stuff for plotting
import time                              # Package to track the running time of our algorithm
import numpy as np                       # Package needed to define arrays
import sys                               # This one and the next packages are needed to correctly import PyTransport
import os                                
from pathlib import Path
###############################################################################################################

############################################## Import PyTransport ##############################################
# To better understand this passage, see xSetup.py or our guide
location = Path(os.path.dirname(__file__))
location = os.path.join(location.resolve().parents[1], 'PyTransPy')
sys.path.append(location)
import PyTransSetup
PyTransSetup.pathSet()

import PyTransTest as PyT               # Import the model. Be sure the name is the same that appears in .compileName3 in xSetup.py
                                        # Do not worry if your editor could not resolve the import. Worry if it shows an error when you
                                        # run the script.  
import PyTransScripts as PyS            # Import the PyTransScript module
###############################################################################################################

############################################# Initialize the model #############################################
# - Fields
nF = PyT.nF()                           # Get the number of fields, usefull for set up and check
fields = np.zeros(nF)                   # Create the array
fields[0] = 12.0; fields[1] = 12.0      # Set up the initial values.
fields = np.array([12.0, 12.0])         # Another way to set up a numpy array which contains the initial values of the fields.

# - Parameters
nP = PyT.nP()                           # Get the number of parameters, usefull for set up and check
params = np.zeros(nP)                   # Create the array
params[0]=1.702050*10.0**(-6.0)         # Set up the values 
params[1]=9.0*params[0]                 # Here we used the first method to setup the array, 
                                        # since the second parameter depends on the first one
# params[2] = 0; params[3] = 0            # In xSetup.py we used four parameter to show how the setup works 
                                        # with a non-canonical field metric. They are not needed here and 
                                        # therefore setted equal to zero.

# - Velocities
V = PyT.V(fields, params)               # Define the potential V
dV = PyT.dV(fields, params)             # Define the derivatives of the potential 

initialCs = np.concatenate((fields, -dV/np.sqrt(3*V))) # Set up the array containing the field values and their derivatives,
                                                       # using the slow roll equation

###############################################################################################################

################################## Run and plot the background fiducial run ###################################
# - Set up the time values
Nstart = 0.0  # The initial value of the time in e-folds. It's usually set to zero. 
Nend = 80.0   # The final value of the time in e-folds. It should be adjusted in
              # order for it to be slightly bigger than the end of inflation. See below.
Nstep = 1000  # The number of time-steps.
t = np.linspace(Nstart, Nend, Nstep) # Array of time-steps.

tols = np.array([10**-17, 10**-17])  # Array with the absolute (tols[0]) and relative (tols[1]) tolerance. 
                                     # These tolerances are needed for the integrator. 

# - Background evolution
back = PyT.backEvolve(t, initialCs, params, tols, True)  # Finally, we run the background evolution.
                                                         # The last argument is a boolean. 
                                                         # If set to true, it stops the run when inflation ends,
                                                         # if set to false, run the evolution up to Nend. 
                                                         # To maximize performances, set Nend as close as possible to 
                                                         # the end of inflation, and set the boolean to True.
                                                         # This may require several attempts. 

Nend_inflation = back[-1, 0]                             # This value represents the time-step at which inflation ends. 
                                                         # If Nend_inflation = Nend, we may not have reached the end of
                                                         # inflation yet.
# The array back is a 2D array. Each column contains the following values: time value, fields (1:nF), velocities(nF:2nF)
# Each row represent a time-step.

# - Plot
fig1 = plt.figure(1)
# We plot the field evolution, as a function of time. 
plt.plot(back[:,0], back[:,1], 'g') # The first argument is the x-axis (here time), the second argument is the y-axis (here field value)
plt.plot(back[:,0], back[:,2], 'r') # Here we have just two fields. 
# Plot settings
title(r'Background evolution',fontsize=15); grid(True); plt.legend(fontsize=15); ylabel(r'Fields', fontsize=20); xlabel(r'$N$', fontsize=15)
grid(True); plt.legend(fontsize=15); plt.savefig("TransTemplate_background.png")

# This is just an example of a plot. Discussing further how to make a good plot is beyond the scope of this example. 
# Some details we'll be given on the guide. 

###############################################################################################################

############################################### Example 2pt run ###############################################

# - Set a scale

# Set a pivot scale which exits after NExit number of e-folds. Here, we chose a scale compatible with CMB experiments. 
# This scale is the one we will use to compute the 2pt phase-space correlation function. It is used to find field and 
# velocity values after NExit number of e-folds. This gives H, then k = aH i.e. the pivot scale. 

NExit = round(back[-1, 0] - 50.0)           # Number of e-folds elapsed at the horizon crossing.
k = PyS.kexitN(NExit, back, params, PyT)    # This is a function in PyTransScript that computes the k value that exits
                                            # the horizon at NExit, given the background evolution. 

NB = 6.0                                    # Number of e-folds before the horizon crossing.
Nstart, backExitMinus = PyS.ICsBE(NB, k, back, params, PyT) # Computes the condition NB e-folds before the horizon crossing of k mode. 

tsig = np.linspace(Nstart, back[-1,0], Nstep) # Array of Nstep time-steps, from NB e-folds befor the horizon crossing of the mode k,
                                              # up to the end of inflation. 

# - Run 2pt evolution

twoPt = PyT.sigEvolve(tsig, k, backExitMinus, params, tols, True)  # This is the PyTransport function that computes the two-point
                                                                     # correlation function. It returns a 2D array with Nstep rows
                                                                     # and 1 + 1 + 2*nF + (2*nF)**2 columns.

# Let's discuss further the content of twoPt
timeS = twoPt[:, 0]                    # This array contains the values of time-steps. It should be equal to tsig.
zz = twoPt[:, 1]                       # This array contains the value of P_\zeta(k) at each time-step
fieldS = twoPt[:, 2:nF]                # This array contains the value of the fields at each time-step
velS = twoPt[:, nF:2*nF]               # This array contains the value of the velocities at each time-step
sigma = twoPt[:, 1 + 1 + 2*nF:]        # This array contains the value of the phase-space two point correlation function.

# - Plot
fig2 = plt.figure(2)
# Here we will plot only the field-field components of the 2pt function
for ii in range(0,2):
    for jj in range(0,2):
        plt.plot(timeS, np.absolute(sigma[:, ii + 2*nF*jj]))
# Plot setting
title(r'$\Sigma$ evolution',fontsize=15); grid(True); plt.legend(fontsize=15); ylabel(r'Aboslute 2pt field correlations', fontsize=20); 
xlabel(r'$N$', fontsize=15); grid(True); yscale('log'); plt.legend(fontsize=15); plt.savefig("TransTemplate_sigma.png")

###############################################################################################################

############################################ Example Tensor 2pt run ############################################





###############################################################################################################                                  

############################################### Example 3pt run ###############################################
# - Find scales and define time-steps

# For the 3pt run (or Bispectrum run) we need three scales. These scales will be defined with respect to k.
# We'll use the two parameters (alpha, beta) to find the triangular configuration of the three k's. 
# For didactical purpose, we choose the equilateral configuration (0, 1/3). 
alpha = 0.0
beta = 1./3. 
k1 = (k/2.) - beta*(k/2.) ; k2 = (k/4.)*(1+alpha+beta) ; k3 = (k/4.)*(1-alpha+beta)  # To find different configurations, change the values of alpha and beta, not this line.

kM = np.min(np.array([k1,k2,k3]))                   # We want to find the smallest scale, since it'll be the first one to exit the horizon.

NstartA, backExitMinusA = PyS.ICsBM(NB, kM, back, params, PyT)  # We find the condition NB e-folds before the horizon crossing of kM
tAlp = np.linspace(NstartA, back[-1,0], Nstep)                  # Array of time-steps for the 3pt evolution. 

# - Run 3pt evolution
threePt = PyT.alphaEvolve(tAlp, k1, k2, k3, backExitMinusA, params, tols, True)   # This is the PyTransport function that computes the three-point
                                                                                  # correlation function. It returns a 2D array with Nstep rows
                                                                                  # and 1 + 3 + 1 + 2*nF + (2*nF)**3 columns.

# Let's discuss further the content of threePt
time_alpha = threePt[:,0]                           # This array contains the values of time-steps. It shoul be equal to tAlp
zz1 = threePt[:,1]                                  # This array contains P_\zeta(k1) at each time-step
zz2 = threePt[:,2]                                  # This array contains P_\zeta(k2) at each time-step
zz3 = threePt[:,3]                                  # This array contains P_\zeta(k3) at each time-step
zzz = threePt[:,4]                                  # This array contains B_\zeta(k1,k2,k3) at each time-step
fieldsAlpha = threePt[:,5:nF]                       # This array contains the field evolution
veloctiyAlpha = threePt[:,nF:2*nF]                  # This array contains the velocity evolution
alpha = threePt[:, 5 + 2*nF:]                       # This array contains the phase-space three-point correlation function

# - Computation of f_NL

fnl = 5.0/6.0 * zzz/(zz2*zz3 + zz1*zz2 + zz1*zz3)   # This array contains fnl at each time step

# - Plot
fig4 = plt.figure(4)
for ii in range(0,2):
    for jj in range(0,2):
        for kk in range(0,2):
            plt.plot(time_alpha, np.abs(alpha[:,ii + 2*nF*jj + 2*nF*2*nF*kk]))
title(r'$\alpha$ evolution',fontsize=15); grid(True); plt.legend(fontsize=15); ylabel(r'Absolute 3pt field correlations', fontsize=20); 
xlabel(r'$N$', fontsize=15); grid(True); yscale('log'); plt.legend(fontsize=15); plt.savefig("DQ4.png")

###############################################################################################################

plt.show()