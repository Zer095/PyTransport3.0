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



//####################### File contains the evolution equations for the background and transport system in the correct form to be used by the integrateor ##########################

#include <iostream>
#include "moments.h"
#include "potential.h"
#include "model.h"
#include <math.h>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <time.h>

using namespace std;

//takes in current state of background system, y -- fields and field derivatives -- calculates dy/dN
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
        // Store the derivatives of the background quantities
       yp[i] = ypin[i];
    }
}

//takes in current state of background and 2pt transport system, y -- calculates dy/dN
void evolveSig( double N,  double yin[], double yp[], double paramsIn[])
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
    This function compute the derivative of (fields, velocities), and sigma. Then it updates the input array yp with the derivatives.
    It takes as argument the time-step N at which we are computing the derivatives, the current values of fields and velocities, 
    an array storing the derivatives to update, the parameters of the model.
    ------------------------------------------------------------------------------------------------------------------------------------------------*/
    model m;                                        // Define model
    double k;                                       // k-mode of Sigma
    int nP = m.getnP();                             // Get number of parameters
    int nF=m.getnF();                               // Get number of fields
    vector<double> p(paramsIn,paramsIn+nP);         // Vector with parameters values
    k=paramsIn[nP];                                 // Get k-value
    vector<double> fields(yin, yin+2*nF);           // Vector with fields
    vector<double> u1=m.u(fields,p);                // Compute u1 with the method model.u(fields, params)
    vector<double> u2=m.u(fields,p,k,N);            // Compute u2 with the method model.u(fields, params, k, N)
    
    // Store derivatives of fields and velocities
    for(int i=0;i<2*nF;i++){yp[i] = u1[i];}
    // Store derivatives of the 2pt function
    for(int i=0;i<2*nF;i++){
        for(int j=0;j<2*nF;j++)
        {
            double sum=0.0;
            for(int m=0;m<2*nF;m++)
            {
                sum = sum + u2[i+m*2*nF]*yin[2*nF+m+2*nF*j] + u2[j+m*2*nF]*yin[2*nF+m+2*nF*i];
            }
            yp[2*nF+i+2*nF*j]=sum;
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
    model m;                                            // Define the model
    double k;                                           // Define the k-mode
    int nP = m.getnP();                                 // Get the number of parameters
	int nF = m.getnF();                                 // Get the number of fields
    vector<double> p(paramsIn,paramsIn+nP);             // Vector with parameters
    k=paramsIn[nP];                                     // Get value of the k-mode
	int nT=1;                                           // Number of T
    vector<double> fields(yin, yin+2*nF);               // Vector with fields
    vector<double> u1=m.u(fields,p);                    // Compute u with method model.u(fields, parameters)
    vector<double> w2=m.w(fields,p,k,N);                // Compute w with method model.w(fields, parameters, k, N)
    // Store the derivatives of fields and velocities
	for(int i=0;i<2*nF;i++){yp[i] = u1[i];}
    // Store the derivative of Gamma
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
    model m;                                            // Define model
    double k1, k2, k3;                                  // Define k-modes
    int nF=m.getnF();                                   // Get number of fields
    int nP = m.getnP();                                 // Get number of parameters
    vector<double> p(paramsIn,paramsIn+nP);             // Vector with parameters
    k1=paramsIn[nP];                                    // Store k1
    k2=paramsIn[nP+1];                                  // Store k2
    k3=paramsIn[nP+2];                                  // Store k3
    vector<double> fields(yin, yin+2*nF);               // Vector with fields
    vector<double> u1, u2a, u2b, u2c, u3a, u3b, u3c;    // Vectors to store u's

    u1=m.u(fields,p);                                   // Compute u for fields and velocity derivatives
    u2a=m.u(fields,p,k1,N);                             // Compute u2(k1)
    u2b=m.u(fields,p,k2,N);                             // Compute u2(k2)
    u2c=m.u(fields,p,k3,N);                             // Compute u2(k3)
    u3a=m.u(fields,p,k1, k2, k3, N);                    // Compute u3(k1,k2,k3)
    u3b=m.u(fields,p,k2, k1, k3, N);                    // Compute u3(k2,k1,k3)
    u3c=m.u(fields,p,k3, k1, k2, N);                    // Compute u3(k3,k1,k2)
    
    // Compute and store (fields, velocities) derivatives
    for(int i=0; i<2*nF; i++){yp[i] = u1[i];}
    // Compute and store sigma real (k1)
    for(int i=0; i<2*nF; i++){for(int j=0;j<2*nF;j++)
        {
            double sum=0.0;
            
            for(int m=0;m<2*nF;m++)
            {
                sum = sum + u2a[i+m*2*nF]*yin[2*nF+m+2*nF*j] + u2a[j+m*2*nF]*yin[2*nF+m+2*nF*i];
            }
            
            yp[2*nF+i+2*nF*j]=sum;
        }}
    // Compute and store sigma real (k2)    
    for(int i=0;i<2*nF;i++){for(int j=0;j<2*nF;j++)
        {
            double sum=0.0;
            for(int m=0;m<2*nF;m++)
            {
                sum = sum + u2b[i+m*2*nF]*yin[2*nF + (2*nF*2*nF) + m+2*nF*j] + u2b[j+m*2*nF]*yin[2*nF + (2*nF*2*nF) + m+2*nF*i];
            }
            yp[2*nF + (2*nF*2*nF) + i+2*nF*j]=sum;
        }}
    // Compute and store sigma real (k3)    
    for(int i=0;i<2*nF;i++){for(int j=0;j<2*nF;j++)
        {
            double sum=0.0;
            for(int m=0;m<2*nF;m++)
            {
                sum = sum + u2c[i+m*2*nF]*yin[2*nF + 2*(2*nF*2*nF) +  m+2*nF*j] + u2c[j+m*2*nF]*yin[2*nF + 2*(2*nF*2*nF) + m+2*nF*i];
            }
            yp[2*nF + 2*(2*nF*2*nF) +i+2*nF*j]=sum;
        }}
    // Compute and store sigma imaginary (k1)
    for(int i=0;i<2*nF;i++){for(int j=0;j<2*nF;j++)
        {
            double sum=0.0;
            for(int m=0;m<2*nF;m++)
            {
                sum = sum + u2a[i+m*2*nF]*yin[2*nF + 3*(2*nF*2*nF) +  m+2*nF*j] + u2a[j+m*2*nF]*yin[2*nF + 3*(2*nF*2*nF) + i+2*nF*m];
            }
            yp[2*nF + 3*(2*nF*2*nF) +i+2*nF*j]=sum;
        }}
    // Compute and store sigma imaginary (k2)
    for(int i=0;i<2*nF;i++){
        for(int j=0;j<2*nF;j++)
        {
            double sum=0.0;
            for(int m=0;m<2*nF;m++)
            {
                sum = sum + u2b[i+m*2*nF]*yin[2*nF + 4*(2*nF*2*nF) +  m+2*nF*j] + u2b[j+m*2*nF]*yin[2*nF + 4*(2*nF*2*nF) + i+2*nF*m];
            }
            yp[2*nF + 4*(2*nF*2*nF) +i+2*nF*j]=sum;
        }}
    // Compute and store sigma imaginary (k3)
    for(int i=0;i<2*nF;i++){for(int j=0;j<2*nF;j++)
        {
            double sum=0.0;
            for(int m=0;m<2*nF;m++)
            {
                sum = sum + u2c[i+m*2*nF]*yin[2*nF + 5*(2*nF*2*nF) +  m+2*nF*j] + u2c[j+m*2*nF]*yin[2*nF + 5*(2*nF*2*nF) + i+2*nF*m];
            }
            yp[2*nF + 5*(2*nF*2*nF) +i+2*nF*j]=sum;
        }}
    // Compute and store alpha(k1,k2,k3)
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
            yp[2*nF + 6*(2*nF*2*nF)  +  i+2*nF*j+k*2*nF*2*nF]=sum+sum2;    
        }
    }}
}

//####################### File contains the evolution equations for the MPP formalism in the correct form to be used by the integrator ##########################

// Compute the derivatives of MPP2 
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
    model m;                                            // Define model
    double k;                                           // Define k-mode
    int nP = m.getnP();                                 // Get number of parameters
    vector<double> p(paramsIn,paramsIn+nP);             // Vector with parameters
    k = paramsIn[nP];                                   // Store k value
    int nF = m.getnF();                                 // Get number of fields
    vector<double> fields(yin, yin+2*nF);               // Vector with fields
    vector<double> u1=m.u(fields,p);                    // Compute u
    vector<double> u2=m.u(fields, p, k, N);             // Compute u2
    // Store derivatives of (fields,velocities)
    for(int i=0; i < 2*nF; i++){yp[i] = u1[i];}
    // Compute and store derivatives of MPP2
    for(int i=0; i < 2*nF; i++)
    {
        for(int j=0; j < 2*nF; j++)
        {
            double sum = 0.0;
            for(int m=0; m < 2*nF; m++)
            {
                sum = sum + u2[i + 2*nF*m]*yin[2*nF + j + 2*nF*m];
            }
            yp[2*nF + 2*nF*i + j] = sum;
        }
    }
}
// Compute the derivatives of (fields,velocities), 3 MPP2 and 3 MPP3
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
    model m;                                                // Define model
    double k1, k2, k3;                                      // Define k-modes
    int nF=m.getnF();                                       // Get number of fields
    int nP = m.getnP();                                     // Get number of parameters
    vector<double> p(paramsIn,paramsIn+nP);                 // Vector with parameters
    k1=paramsIn[nP];                                        // Get k1 value
    k2=paramsIn[nP+1];                                      // Get k2 value
    k3=paramsIn[nP+2];                                      // Get k3 value
    vector<double> fields(yin, yin+2*nF);                   // Vector with fields
    vector<double> u1, u21, u22, u23, u31, u32, u33;        // Vector storing u

    u1 = m.u(fields, p);                                    // Calculate u
    u21 = m.u(fields, p, k1, N);                            // Calculate u2(k1)
    u22 = m.u(fields, p, k2, N);                            // Calculate u2(k2)
    u23 = m.u(fields, p, k3, N);                            // Calculate u2(k3)
    u31 = m.u(fields, p, k1, k2, k3, N);                    // Calculate u3(k1,k2,k3)
    u32 = m.u(fields, p, k2, k1, k3, N);                    // Calculate u3(k2,k1,k3)
    u33 = m.u(fields, p, k3, k1, k2, N);                    // Calculate u3(k3,k1,k2)

    // Compute and store derivative of the fields
    for(int i=0; i < 2*nF; i++){yp[i] = u1[i];}
    // Compute and store derivative of MMP2(k1)
    for(int i=0; i < 2*nF; i++)
    {
        for(int j=0; j < 2*nF; j++)
        {
            double sum = 0.0;
            for(int m = 0; m < 2*nF; m++)
            {
                sum = sum + u21[i + 2*nF*m]*yin[2*nF + 0*(2*nF*2*nF) + j + 2*nF*m];
            }
            yp[2*nF + 0*(2*nF*2*nF) + 2*nF*i + j] = sum;
        }
    }
    // Compute and store derivative of MMP2(k2)
    for(int i=0; i < 2*nF; i++)
    {
        for(int j=0; j < 2*nF; j++)
        {
            double sum = 0.0;
            for(int m=0; m < 2*nF; m++)
            {
                sum = sum + u22[i + 2*nF*m]*yin[2*nF + 1*(2*nF*2*nF) + j + 2*nF*m];
            }
            yp[2*nF + 1*(2*nF*2*nF) + 2*nF*i + j] = sum;
        }
    }
    // Compute and store derivative of MMP2(k3)
    for(int i=0; i < 2*nF; i++)
    {
        for(int j=0; j < 2*nF; j++)
        {
            double sum = 0.0;
            for(int m=0; m < 2*nF; m++)
            {
                sum = sum + u23[i + 2*nF*m]*yin[2*nF + 2*(2*nF*2*nF) + j + 2*nF*m];
            }
            yp[2*nF + 2*(2*nF*2*nF) + 2*nF*i + j] = sum;
        }
    }
    // Compute and store derivative of MPP3(k1,k2,k3)
    double t1 = 0.;             // Temp variable
    double t2 = 0.;             // Temp variable
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
                        t2 = t2 + u31[i + m*(2*nF) + n*(2*nF*2*nF)]*yin[2*nF + 1*(2*nF*2*nF) + (2*nF)*m + j]*yin[2*nF + 2*(2*nF*2*nF) + (2*nF)*n + k];
                    }
                    t1 = t1 + u21[i + (2*nF)*m]*yin[2*nF + 3*(2*nF*2*nF) + (2*nF*2*nF)*m + (2*nF)*j + k];
                }
                yp[2*nF + 3*(2*nF*2*nF) + 0*(2*nF*2*nF*2*nF) +  (2*nF*2*nF)*i + (2*nF)*j + k] = t1 + t2;
            }
        }
    }
    // Compute and store derivative of Rho2(k2,k1,k3)
    for(int i=0; i < 2*nF; i++)
    {
        for(int j=0; j < 2*nF; j++)
        {
            for(int k = 0; k < 2*nF; k++)
            {
                t1 = 0.;    // Temp variable
                t2 = 0.;    // temp variable
                for(int m = 0; m < 2*nF; m++)
                {
                    for(int n = 0; n < 2*nF; n++)
                    {
                        t2 = t2 + u32[i+ m*(2*nF) + n*(2*nF*2*nF)]*yin[2*nF + 2*nF*m + j]*yin[2*nF + 2*(2*nF*2*nF) + 2*nF*n + k];
                    }
                    t1 = t1 + u22[i + 2*nF*m]*yin[2*nF + 3*(2*nF*2*nF) + 1*(2*nF*2*nF*2*nF) + (2*nF*2*nF)*m + 2*nF*j + k];
                }
                yp[2*nF + 3*(2*nF*2*nF) + 1*(2*nF*2*nF*2*nF) + (2*nF*2*nF)*i + (2*nF)*j + k] = t1 + t2; 
            }
        }
    }
    // Compute and store derivative of MPP3(k3,k1,k2)
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
                yp[2*nF + 3*(2*nF*2*nF) + 2*(2*nF*2*nF*2*nF) + (2*nF*2*nF)*i + (2*nF)*j + k] = t1 + t2; 
            }
        }
    }
}
