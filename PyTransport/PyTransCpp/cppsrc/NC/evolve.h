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

// This file contains a prototype of the potential.h file of PyTransport -- it is edited by the PyTransScripts module

//####################### File contains the evolution equations for the background and transport system (with non-canonical field-metric) in the correct form to be used by the integrateor ##########################

#include <iostream>
#include "moments.h"
#include "../potential.h"
#include "model.h"
#include <math.h>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <time.h>

#include "../fieldmetric.h"

using namespace std;

// takes in current state of background system, y -- fields and field derivatives -- calculates dy/dN for
// models with field space metric
void evolveB(double N, double yin[], double yp[], double paramsIn[])
{
    /*----------------------------------------------------------------------------------------------------------------------------------------------
    Update the input array yp with the derivative of fields and velocities, given the current state of the background system.

    Arguments
    ---------
    N : double
        time-step at which we are evaluating the derivatives;
    yin : array of doubles
        array storing the values of field and velocities;
    yp : array of doubles
        array storing the values of the derivatives;
    paramsIn: array of doubles
        array storing the parameters of the model;

    Returns
    -------
    None

    Description
    -----------
    This function compute the derivative of the fields and velocities, and update the input array yp with the derivative of fields and velocities.
    It takes as argument the time-step N at which we are computing the derivatives, the current values of fields and velocities, 
    an array storing the derivatives to update, the parameters of the model.
------------------------------------------------------------------------------------------------------------------------------------------------*/
    model m;                                    // Define the model
    int nF = m.getnF();                         // Get number of fields
    int nP = m.getnP();                         // Get number of parameters
    vector<double> p(paramsIn,paramsIn+nP);     // Vector with parameters
    vector<double> fields(yin, yin+2*nF);       // Vector with fields and velocities
    vector<double> ypin = m.u(fields,p);        // Compute derivatives with the metod model.u(fields, parameters)
    // Store derivatives in yp
    for (int i=0;i<2*nF;i++)
    {
       yp[i] = ypin[i];
    }
}

// defines the connection (Gamma) contracted with the velocity (v) , and made into a 2nF x 2nF matrix of form ( [[ v Gamma, 0], [0, v Gamma]] )
//as needed for the equations of motion below.
vector<double> vG(vector<double> f,vector<double> p, double N)
	{
/*----------------------------------------------------------------------------------------------------------------------------------------------
Defines the connection (Gamma) contracted with the velocity (v) as a 2nF X 2nF matrix.

Arguments
---------
f : vector<double>
    vector with values of the field and of the derivatives of the fields;
p : vector<double>
    vector with values of the parameters;
N : double
    value of time (in e-folds) at which we are computing the connections.

Returns
-------

vG : vector<double>
    vector (2nF x 2nF) storing the values of the connections in the form ([ v Gamma, 0], [0, v Gamma]). 

Description
-----------
Defines the connection (Gamma) contracted with the velocity (v) , and made into a 2nF x 2nF matrix of form ( [[ v Gamma, 0], [0, v Gamma]] )
as needed for the equations of motion below.


------------------------------------------------------------------------------------------------------------------------------------------------*/        
		model m;                                        // Define the model object
		int nF = m.getnF();                             // Get the number of fields
		vector<double> vGout(2*nF*2*nF);                // Define the output vector
		fieldmetric fmet;                               // Define the field metric object
		double Hi=m.H(f,p);                             // Define and compute the Hubble parameter     

		vector<double> FMi; FMi = fmet.fmetric(f,p);    // Define and get the field metric
		vector<double> CHR; CHR = fmet.Chroff(f,p);     // Define and get the Christoffel
		double sum1=0.0;
		for(int i=0;i<nF;i++){
            for(int j=0;j<nF;j++)
            {	
                sum1=0.0;
                for(int m=0;m<nF;m++){
                    for(int l=0;l<nF;l++){}
                    sum1= sum1 + CHR[(2*nF)*(2*nF)*(i)+(2*nF)*(j+nF)+m+nF]*f[nF+m];
                }
                
                vGout[i+ j*2*nF]=sum1/Hi;
                vGout[i+nF+(j)*2*nF]=0.0;
                vGout[i+(j+nF)*2*nF]=0.0;
                vGout[i+nF+(j+nF)*2*nF]=sum1/Hi;
            }
		}
		return vGout;
	}	


//takes in current state of background and 2pt transport system, y -- calculates dy/dN for
//models with field space metric

void evolveSig( double N, double yin[], double yp[], double paramsIn[])
{
    /*----------------------------------------------------------------------------------------------------------------------------------------------
    Update the input array yp with the derivative of (fields, velocities) and 2pt function Sigma, given the current state of the background system.

    Arguments
    ---------
    N : double
        time-step at which we are evaluating the derivatives;
    yin : array of doubles
        array storing the initial values of (field,velocities) and sigma;
    yp : array of doubles
        array storing the values of the derivatives;
    paramsIn: array of doubles
        array storing the parameters of the model,
        the last entrance is the value of the k-mode;

    Returns
    -------
    None

    Description
    -----------
    This function compute the derivative of (fields, velocities), and Sigma. Then it updates the input array yp with the derivatives.
    It takes as argument the time-step N at which we are computing the derivatives, the current values of fields and velocities, 
    an array storing the derivatives to update, the parameters of the model.
    ------------------------------------------------------------------------------------------------------------------------------------------------*/
    model m;                                                        // Define model object           
    int nP = m.getnP();                                             // Get the number of parameters
    vector<double> p(paramsIn,paramsIn+nP);                         // Vector with parameters values
    double k; k=paramsIn[nP];                                       // Define the k-mode of Sigma                                    
    int nF=m.getnF();                                               // Get the number of fields
    vector<double> fields(yin, yin+2*nF);                           // Vector with values of fields and derivatives
    vector<double> u1=m.u(fields,p);                                // Compute u1
    vector<double> u2=m.u(fields,p,k,N);                            // Compute u2
	vector<double> vGi; vGi=vG(fields,p,N);                         // Define and compute the values of the connection
    
    // Redefine u2
    for (int ii=0;ii<2*nF*2*nF;ii++){u2[ii] = u2[ii] - vGi[ii];}
	// Store derivatives of the background
	for(int i=0;i<2*nF;i++){yp[i] = u1[i];}
    // Store derivatives of sigma
    for(int i=0;i<2*nF;i++){
        for(int j=0;j<2*nF;j++)
        {
            double sum=0.0;
            for(int m=0;m<2*nF;m++)
            {
                sum = sum + u2[i + m*2*nF]*yin[2*nF + m + 2*nF*j] + u2[j + m*2*nF]*yin[2*nF + m + 2*nF*i];
            }
            yp[2*nF + i + 2*nF*j]=sum ;
        }
    }
}
 
//takes in current state of background and 2pt Gamma transport system, y -- calculates dy/dN for
//models with field space metric

void evolveGam( double N, double yin[], double yp[], double paramsIn[])
{
    /*----------------------------------------------------------------------------------------------------------------------------------------------
    Update the input array yp with the derivative of (fields, velocities) and 2pt tensor function Gamma, given the current state of the background system.

    Arguments
    ---------
    N : double
        time-step at which we are evaluating the derivatives;
    yin : array of doubles
        array storing the initial values of (field,velocities) and gamma;
    yp : array of doubles
        array storing the values of the derivatives;
    paramsIn: array of doubles
        array storing the parameters of the model,
        the last entrance is the value of the k-mode;

    Returns
    -------
    None

    Description
    -----------
    This function compute the derivative of (fields, velocities), and gamma. Then it updates the input array yp with the derivatives.
    It takes as argument the time-step N at which we are computing the derivatives, the current values of fields and velocities, 
    an array storing the derivatives to update, the parameters of the model.
    ------------------------------------------------------------------------------------------------------------------------------------------------*/
    model m;                                        // Define model
    int nP = m.getnP();                             // Get number of parameters
	int nF = m.getnF();                             // Get number of fields
    vector<double> p(paramsIn,paramsIn+nP);         // Vector with parameters values
    double k;k=paramsIn[nP];                        // Define and store value of k-mode
	int nT=1;
    vector<double> fields(yin, yin+2*nF);           // Vector with fields values
    vector<double> u1=m.u(fields,p);                // Compute u1
    vector<double> w2=m.w(fields,p,k,N);            // Compute u2

    // Store derivatives of background
	for(int i=0;i<2*nF;i++){yp[i] = u1[i];}
    // Store derivatives of Gamma
    for(int i=0;i<2*nT;i++){for(int j=0;j<2*nT;j++)
        {
            double sum=0.0;
            for(int m=0;m<2*nT;m++)
            {
                sum = sum + w2[i+m*2*nT]*yin[2*nF+m+2*nT*j] + w2[j+m*2*nT]*yin[2*nF+i+2*nT*m];
            }
            yp[2*nF+i+2*nT*j]=sum ;

        }
    }
}


//takes in current state of background, the three 2pts functions needed to evolve the 3pt, and the 3pt, y, and calculates dy/dN
//models with field space metric

void evolveAlp(double N,  double yin[], double yp[], double paramsIn[])
{
    /*------------------------------------------------------------------------------------------------------------------------------------------------
    Update the input array yp with the derivative of (fields, velocities), 3 real sigmas, 3 imaginary sigmas, 
    alpha, given the current state of the background system.

    Arguments
    ---------
    N : double
        time-step at which we are evaluating the derivatives;
    yin : array of doubles
        array storing the initial values of (field,velocities) and 3 real sigmas, 3 imaginary sigmas, alpha;
    yp : array of doubles
        array storing the values of the derivatives;
    paramsIn: array of doubles
        array storing the parameters of the model,
        the last 3 entrances are the value of the k-modes k1, k2, k3;

    Returns
    -------
    None

    Description
    -----------
    This function compute the derivative of (fields, velocities), 3 real sigmas, 3 imaginary sigmas and alpha. 
    Then it updates the input array yp with the derivatives.
    It takes as argument the time-step N at which we are computing the derivatives, the current values of (fields,velocities),
    the current values of: 3 real sigmas (1 for each k), 3 imaginary sigmas (1 for each k), and alpha.
    an array storing the derivatives to update, the parameters of the model.
    ------------------------------------------------------------------------------------------------------------------------------------------------*/
    model m;                                            // Define model object
	fieldmetric fmet;                                   // Define fieldmetric object
    int nF=m.getnF(); int nP = m.getnP();               // Get number of fields and parameters
    vector<double> p(paramsIn,paramsIn+nP);             // Vector with parameters values
    double k1, k2, k3;                                  // Define k-modes
    k1=paramsIn[nP];k2=paramsIn[nP+1];k3=paramsIn[nP+2];// Store k-modes
    vector<double> fields(yin, yin+2*nF);               // Vector with fields values
    vector<double> u1, u2a, u2b, u2c, u3a, u3b, u3c;    // Define u-tensors vector
	vector<double> FMi; FMi = fmet.fmetric(fields,p);   // Define and get fieldmetric
    vector<double> CHR; CHR = fmet.Chroff(fields,p);    // Define and get Christoffel
    vector<double> vGi; vGi=vG(fields,p,N);             // Define and get Connections
    u1=m.u(fields,p);                                   // Compute u1
    u2a=m.u(fields,p,k1,N);                             // Compute u2a
    // Redifine u2a
    for (int ii=0;ii<2*nF*2*nF;ii++){u2a[ii] = u2a[ii] - vGi[ii];}
    u2b=m.u(fields,p,k2,N);                             // Compute u2b
    // Redifine u2b
    for (int ii=0;ii<2*nF*2*nF;ii++){u2b[ii] = u2b[ii] - vGi[ii];}
    u2c=m.u(fields,p,k3,N);                             // Compute u2c
    // Redifine u2c
    for (int ii=0;ii<2*nF*2*nF;ii++){u2c[ii] = u2c[ii] - vGi[ii];}
    u3a=m.u(fields,p,k1, k2, k3, N);                    // Compute u3a
    u3b=m.u(fields,p,k2, k1, k3, N);                    // Compute u3b
    u3c=m.u(fields,p,k3, k1, k2, N);                    // Compute u3c
	// Store fields derivatives
    for(int i=0; i<2*nF; i++){yp[i] = u1[i];}
    // Compute sigma real (k1) derivatives
    for(int i=0; i<2*nF; i++){
        for(int j=0;j<2*nF;j++)
        {
            double sum=0.0;
            for(int m=0;m<2*nF;m++)
            {
                sum = sum + u2a[i+m*2*nF]*yin[2*nF+m+2*nF*j] + u2a[j+m*2*nF]*yin[2*nF+m+2*nF*i];
            }
            yp[2*nF+i+2*nF*j]=sum;
        }
    }
    // Compute sigma real (k2) derivatives
    for(int i=0;i<2*nF;i++){for(int j=0;j<2*nF;j++)
    {
        double sum=0.0;
        for(int m=0;m<2*nF;m++)
        {
            sum = sum + u2b[i+m*2*nF]*yin[2*nF + (2*nF*2*nF) + m+2*nF*j] + u2b[j+m*2*nF]*yin[2*nF + (2*nF*2*nF) + m+2*nF*i];
        }
        yp[2*nF + (2*nF*2*nF) + i+2*nF*j]=sum;
    }}
    // Compute sigma real (k3) derivatives
    for(int i=0;i<2*nF;i++){for(int j=0;j<2*nF;j++)
    {
        double sum=0.0;
        for(int m=0;m<2*nF;m++)
        {
            sum = sum + u2c[i+m*2*nF]*yin[2*nF + 2*(2*nF*2*nF) +  m+2*nF*j] + u2c[j+m*2*nF]*yin[2*nF + 2*(2*nF*2*nF) + m+2*nF*i];
        }
        yp[2*nF + 2*(2*nF*2*nF) +i+2*nF*j]=sum;
    }}
    // Compute sigma imaginary (k1) derivatives
    for(int i=0;i<2*nF;i++){for(int j=0;j<2*nF;j++)
    {
        double sum=0.0;
        for(int m=0;m<2*nF;m++)
        {
            sum = sum + u2a[i+m*2*nF]*yin[2*nF + 3*(2*nF*2*nF) +  m+2*nF*j] + u2a[j+m*2*nF]*yin[2*nF + 3*(2*nF*2*nF) + i+2*nF*m];
        }
        yp[2*nF + 3*(2*nF*2*nF) +i+2*nF*j]=sum;
    }}
    // Compute sigma imaginary (k2) derivatives
    for(int i=0;i<2*nF;i++){for(int j=0;j<2*nF;j++)
    {
        double sum=0.0;
        for(int m=0;m<2*nF;m++)
        {
            sum = sum + u2b[i+m*2*nF]*yin[2*nF + 4*(2*nF*2*nF) +  m+2*nF*j] + u2b[j+m*2*nF]*yin[2*nF + 4*(2*nF*2*nF) + i+2*nF*m];
        }
        yp[2*nF + 4*(2*nF*2*nF) +i+2*nF*j]=sum;
    }}
    // Compute sigma imaginary (k3) derivatives
    for(int i=0;i<2*nF;i++){for(int j=0;j<2*nF;j++)
    {
        double sum=0.0;
        for(int m=0;m<2*nF;m++)
        {
            sum = sum + u2c[i+m*2*nF]*yin[2*nF + 5*(2*nF*2*nF) +  m+2*nF*j] + u2c[j+m*2*nF]*yin[2*nF + 5*(2*nF*2*nF) + i+2*nF*m];
        }
        yp[2*nF + 5*(2*nF*2*nF) +i+2*nF*j]=sum;
    }}
    // Compute alpha derivatives
    for(int i=0;i<2*nF;i++){for(int j=0;j<2*nF;j++){for(int k=0;k<2*nF;k++)
    {
        double sum=0.0;
        double sum2=0.0;
        for(int m=0;m<2*nF;m++)
        {
            sum = sum + u2a[i+m*2*nF]*yin[2*nF + 6*(2*nF*2*nF)  +  m + j*2*nF + k*2*nF*2*nF] + u2b[j+m*2*nF]*yin[2*nF + 6*(2*nF*2*nF)
                                                                                                                 +  i + m*2*nF + k*2*nF*2*nF] + u2c[k+m*2*nF]*yin[2*nF + 6*(2*nF*2*nF)  +  i + j*2*nF + m*2*nF*2*nF];
            for(int n=0;n<2*nF;n++){
                sum2 = sum2 + u3a[i+ n*2*nF + m*2*nF*2*nF ]*yin[2*nF + 1*(2*nF*2*nF) + j + n*2*nF]*yin[2*nF +2* (2*nF*2*nF) + k + m*2*nF]
                + u3b[j+ n*2*nF + m*2*nF*2*nF ]*yin[2*nF + 0*(2*nF*2*nF) + n + i*2*nF]*yin[2*nF +2* (2*nF*2*nF) + m + k*2*nF]
                + u3c[k+ n*2*nF + m*2*nF*2*nF ]*yin[2*nF + 0*(2*nF*2*nF) + n+ i*2*nF]*yin[2*nF +1* (2*nF*2*nF) + j + m*2*nF]
                - 1.*u3a[i+ n*2*nF + m*2*nF*2*nF ]*yin[2*nF + 4*(2*nF*2*nF) + n + j*2*nF]*yin[2*nF +5* (2*nF*2*nF) + m + k*2*nF]
                - 1.*u3b[j+ n*2*nF + m*2*nF*2*nF ]*yin[2*nF + 3*(2*nF*2*nF) + i + n*2*nF]*yin[2*nF +5* (2*nF*2*nF) + m + k*2*nF]
                - 1.*u3c[k+ n*2*nF + m*2*nF*2*nF ]*yin[2*nF + 3*(2*nF*2*nF) + i + n*2*nF]*yin[2*nF +4* (2*nF*2*nF) + j + m*2*nF];
            }}
        yp[2*nF + 6*(2*nF*2*nF)  +  i +2*nF*j+k*2*nF*2*nF]=sum+sum2;
        
    }}}  
}
//####################### File contains the evolution equations for the MPP formalism (with non canonical field metric) in the correct form to be used by the integrator ##########################

void evolveRho1(double N, double yin[], double yp[], double paramsIn[])
{
    /*----------------------------------------------------------------------------------------------------------------------------------------------------
    Update the input array yp with the derivative of (fields, velocities) and MPP matrix with 2 indices, given the current state of the background system.

    Arguments
    ---------
    N : double
        time-step at which we are evaluating the derivatives;
    yin : array of doubles
        array storing the initial values of (field,velocities) and MPP2;
    yp : array of doubles
        array storing the values of the derivatives;
    paramsIn: array of doubles
        array storing the parameters of the model,
        the last entrance is the value of the k-mode;

    Returns
    -------
    None

    Description
    -----------
    This function compute the derivative of (fields, velocities), and sigma. Then it updates the input array yp with the derivatives.
    It takes as argument the time-step N at which we are computing the derivatives, the current values of fields and velocities, 
    an array storing the derivatives to update, the parameters of the model.
    ----------------------------------------------------------------------------------------------------------------------------------------------------*/
    model m;                                                    // Define model
    int nP = m.getnP(); int nF = m.getnF();                     // Get number of fields and parameters
    vector<double> p(paramsIn,paramsIn+nP);                     // Vector with parameters values
    double k; k = paramsIn[nP];                                 // Define and store k-mode
    vector<double> fields(yin, yin+2*nF);                       // Vector with fields values
    vector<double> u1=m.u(fields,p);                            // Compute u1
    vector<double> u2=m.u(fields, p, k, N);                     // Compute u2
    vector<double> vGi; vGi=vG(fields,p,N);                     // Define and store connections
    // Redifine u2
    for(int ii=0; ii<2*nF*2*nF;ii++){u2[ii] = u2[ii]-vGi[ii];}
    // Store fields derivatives
    for(int i=0; i < 2*nF; i++){yp[i] = u1[i];}
    // Store MPP2 derivatives
    for(int i=0; i < 2*nF; i++)
    {
        for(int j=0; j < 2*nF; j++)
        {
            double sum = 0.0;
            for(int m=0; m < 2*nF; m++)
            {
                sum = sum + u2[i + 2*nF*m]*yin[2*nF + j + 2*nF*m];
                // sum = sum + u2[i + 2*nF*m]*yin[2*nF + j + 2*nF*m] + vGi[2*nF*j + m]*yin[2*nF + 2*nF*i + m];
            }
            yp[2*nF + 2*nF*i + j] = sum ;
        }
    }
}

void evolveRho2(double N, double yin[], double yp[], double paramsIn[])
{
    /*----------------------------------------------------------------------------------------------------------------------------------------------------
    Update the input array yp with the derivative of (fields, velocities), 3 MPP2, 3 MPP3, 
    given the current state of the background system.

    Arguments
    ---------
    N : double
        time-step at which we are evaluating the derivatives;
    yin : array of doubles
        array storing the initial values of (field,velocities), 3 MPP2, and 3 MPP3;
    yp : array of doubles
        array storing the values of the derivatives;
    paramsIn: array of doubles
        array storing the parameters of the model,
        the last 3 entrances are the value of the k-modes k1, k2, k3;

    Returns
    -------
    None

    Description
    -----------
    This function compute the derivative of (fields, velocities), 3 MPP2, 3 MPP3. 
    Then it updates the input array yp with the derivatives.
    It takes as argument the time-step N at which we are computing the derivatives, the current values of (fields,velocities),
    the current values of: 3 MPP2 (1 for each k), 3 MPP3 (1 for each permutation of k's), and alpha.
    an array storing the derivatives to update, the parameters of the model.
    ------------------------------------------------------------------------------------------------------------------------------------------------*/
    model m;                                                    // Define model object
    fieldmetric fmet;                                           // Define fieldmetric object
    int nF=m.getnF(); int nP = m.getnP();                       // Get number of fields and parameters
    vector<double> fields(yin, yin+2*nF);                       // Vector with fields
    vector<double> p(paramsIn,paramsIn+nP);                     // Vector with parameters
    double k1, k2, k3;                                          // Define k-modes
    k1=paramsIn[nP]; k2=paramsIn[nP+1]; k3=paramsIn[nP+2];      // Store k-modes
    vector<double> u1, u21, u22, u23, u31, u32, u33;            // Vectors with u-tensors
    vector<double> CHR; CHR = fmet.Chroff(fields,p);            // Define and get Christoffel
    vector<double> FMi; FMi = fmet.fmetric(fields,p);           // Define and get Field-metric
    vector<double> vGi; vGi=vG(fields,p,N);                     // Define and get connections
    u1 = m.u(fields, p);                                        // Compute u1
    u21 = m.u(fields, p, k1, N);                                // Compute u21
    // Redifine u21
    for (int ii=0;ii<2*nF*2*nF;ii++){u21[ii] = u21[ii] - vGi[ii];}
    u22 = m.u(fields, p, k2, N);                                // Compute u22
    // Redifine u22
    for (int ii=0;ii<2*nF*2*nF;ii++){u22[ii] = u22[ii] - vGi[ii];}
    u23 = m.u(fields, p, k3, N);                                // Compute u23
    // Redifine u23
    for (int ii=0;ii<2*nF*2*nF;ii++){u23[ii] = u23[ii] - vGi[ii];}
    u31 = m.u(fields, p, k1, k2, k3, N);                        // Compute u31
    u32 = m.u(fields, p, k2, k1, k3, N);                        // Compute u32
    u33 = m.u(fields, p, k3, k1, k2, N);                        // Compute u33


    // Derivative of the fields
    for(int i=0; i < 2*nF; i++){yp[i] = u1[i];}
    // Derivative of Rho1(k1)
    for(int i=0; i < 2*nF; i++)
    {
        for(int j=0; j < 2*nF; j++)
        {
            double sum = 0.0;
            for(int m = 0; m < 2*nF; m++)
            {
                sum = sum + u21[i + 2*nF*m]*yin[2*nF + j + 2*nF*m];
                // sum = sum + u21[i + 2*nF*m]*yin[2*nF + j + 2*nF*m] + vGi[2*nF*j + m]*yin[2*nF + 2*nF*i + m];
            }
            yp[2*nF + 2*nF*i + j] = sum;
        }
    }
    // Derivative of Rho1(k2)
    for(int i=0; i < 2*nF; i++)
    {
        for(int j=0; j < 2*nF; j++)
        {
            double sum = 0.0;
            for(int m=0; m < 2*nF; m++)
            {
                sum = sum + u22[i + 2*nF*m]*yin[2*nF + (2*nF*2*nF) + j + 2*nF*m];
                // sum = sum + u22[i + 2*nF*m]*yin[2*nF + (2*nF*2*nF) + j + 2*nF*m] + vGi[2*nF*j + m]*yin[2*nF + (2*nF*2*nF) + 2*nF*i + m];
            }
            yp[2*nF + (2*nF*2*nF) + 2*nF*i + j] = sum;
        }
    } 
    // Derivative of Rho1(k3)
    for(int i=0; i < 2*nF; i++)
    {
        for(int j=0; j < 2*nF; j++)
        {
            double sum = 0.0;
            for(int m=0; m < 2*nF; m++)
            {
                sum = sum + u23[i + 2*nF*m]*yin[2*nF + 2*(2*nF*2*nF) + j + 2*nF*m];
                // sum = sum + u23[i + 2*nF*m]*yin[2*nF + 2*(2*nF*2*nF) + j + 2*nF*m]+ vGi[2*nF*j + m]*yin[2*nF + 2*(2*nF*2*nF) + 2*nF*i + m];
            }
            yp[2*nF + 2*(2*nF*2*nF) + 2*nF*i + j] = sum;
        }
    }
    // Derivative of Rho2(k1,k2,k3)
    double t1 = 0.;
    double t2 = 0.;
    for(int i=0; i < 2*nF; i++)
    {
        for(int j=0; j < 2*nF; j++)
        {
            for(int k = 0; k < 2*nF; k++)
            {
                t1 = 0.;
                t2 = 0.;
                for(int m = 0; m < 2*nF; m++)
                {
                    for(int n = 0; n < 2*nF; n++)
                    {
                        t2 = t2 + u31[i + m*(2*nF) + n*(2*nF*2*nF)]*yin[2*nF + (2*nF*2*nF) + 2*nF*m + j]*yin[2*nF + 2*(2*nF*2*nF) + 2*nF*n + k];
                    }
                    t1 = t1 + u21[i + 2*nF*m]*yin[2*nF + 3*(2*nF*2*nF) + (2*nF*2*nF)*m + 2*nF*j + k];
                }
                yp[2*nF + 3*(2*nF*2*nF) + (2*nF*2*nF)*i + 2*nF*j + k] = t1 + t2;
            }
        }
    }
    // Derivative of Rho2(k2,k1,k3)
    for(int i=0; i < 2*nF; i++)
    {
        for(int j=0; j < 2*nF; j++)
        {
            for(int k = 0; k < 2*nF; k++)
            {
                t1 = 0.;
                t2 = 0.;
                for(int m = 0; m < 2*nF; m++)
                {
                    for(int n = 0; n < 2*nF; n++)
                    {
                        t2 = t2 + u32[i+ m*(2*nF) + n*(2*nF*2*nF)]*yin[2*nF + 2*nF*m + j]*yin[2*nF + 2*(2*nF*2*nF) + 2*nF*n + k];
                    }
                    t1 = t1 + u22[i + 2*nF*m]*yin[2*nF + 3*(2*nF*2*nF) + (2*nF*2*nF*2*nF) + (2*nF*2*nF)*m + 2*nF*j + k];
                }
                yp[2*nF + 3*(2*nF*2*nF) + (2*nF*2*nF*2*nF) + (2*nF*2*nF)*i + 2*nF*j + k] = t1 + t2; 
            }
        }
    }
    // Derivative of Rho2(k3,k1,k2)
    for(int i=0; i < 2*nF; i++)
    {
        for(int j=0; j < 2*nF; j++)
        {
            for(int k = 0; k < 2*nF; k++)
            {
                t1 = 0.;
                t2 = 0.;
                for(int m = 0; m < 2*nF; m++)
                {
                    for(int n = 0; n < 2*nF; n++)
                    {
                        t2 = t2 + u33[i+ m*(2*nF) + n*(2*nF*2*nF)]*yin[2*nF + 2*nF*m + j]*yin[2*nF + (2*nF*2*nF) + 2*nF*n + k];
                    }
                    t1 = t1 + u23[i + 2*nF*m]*yin[2*nF + 3*(2*nF*2*nF) + 2*(2*nF*2*nF*2*nF) + (2*nF*2*nF)*m + 2*nF*j + k];
                }
                yp[2*nF + 3*(2*nF*2*nF) + 2*(2*nF*2*nF*2*nF) + (2*nF*2*nF)*i + 2*nF*j + k] = t1 + t2; 
            }
        }
    }
}